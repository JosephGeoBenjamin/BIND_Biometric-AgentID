"""
aligner.py
----------
Thin wrapper around the CVLFace DFA aligner model that handles:
  - Device placement (CPU / CUDA)
  - Single-image inference with full error handling
  - Returning a structured result so the caller can decide what to do on failure

The CVLFace aligner returns:
    aligned_x     (1, 3, 112, 112) tensor  ← what we save
    orig_ldmks    (1, 5, 2) landmarks in original image space
    aligned_ldmks (1, 5, 2) landmarks in 112x112 aligned space
    score         (1,) confidence / quality score
    thetas        (1, 2, 3) affine transformation matrix
    bbox          (1, 4) bounding box in original image

We only *need* aligned_x, but we surface all fields for debugging.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import argparse
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from cvlface_loader import load_aligner

from lib.aligner import FaceAligner as FAM_FaceAligner

##------------------------------------------------------------------------------


# Extensions we consider "image files"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# CVLFace normalization (mean/std = 0.5 for all channels → maps [0,1] → [-1,1])
_TRANSFORM = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



##------------------------------------------------------------------------------

def collect_image_paths(data_root: Path) -> list[Path]:
    """
    Recursively collect all supported image files under data_root.

    Returns a sorted list of absolute Paths.
    """
    paths = sorted(
        p for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"Found {len(paths)} image(s) under {data_root}")
    return paths


def mirror_path( image_path: Path, data_root: Path, save_root: Path) -> Path:
    """
    Compute the output path that mirrors the source directory structure.

    Example:
        image_path = /data/id001/cam1/face.jpg
        data_root  = /data
        save_root  = /aligned
        → returns   /aligned/id001/cam1/face.jpg
    """
    relative = image_path.relative_to(data_root)
    return save_root / relative



def load_image_as_tensor(image_path: Path) -> Optional[torch.Tensor]:
    """
    Load an image file and return a (1, 3, H, W) float32 tensor in [-1, 1].

    Returns None if the file cannot be opened.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"Cannot open image {image_path}: {exc}")
        return None

    tensor = _TRANSFORM(img).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a (1, 3, H, W) or (3, H, W) float32 tensor in [-1, 1] to a PIL RGB image.
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # → (3, H, W)

    # De-normalize: [-1, 1] → [0, 255]
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    arr = (arr * 0.5 + 0.5) * 255.0
    arr = arr.clip(0, 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def save_aligned_image(
    aligned_buffer,
    save_path: Path,
) -> None:
    """
    Save an aligned face (torch tensor OR numpy array) to disk.

    Args:
        aligned_buffer: torch.Tensor (1,3,112,112) OR np.ndarray
        save_path:      Full destination path
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Handle input type ---
    if isinstance(aligned_buffer, torch.Tensor):
        pil_img = tensor_to_pil(aligned_buffer)

    elif isinstance(aligned_buffer, np.ndarray):
        # Expected shapes: (H,W,3) or (3,H,W)
        if aligned_buffer.ndim == 3 and aligned_buffer.shape[0] == 3:
            aligned_buffer = np.transpose(aligned_buffer, (1, 2, 0))

        # Normalize if needed (assume float [0,1] or [0,255])
        if aligned_buffer.dtype != np.uint8:
            aligned_buffer = (aligned_buffer * 255).clip(0, 255).astype(np.uint8)

        pil_img = Image.fromarray(aligned_buffer)

    else:
        raise TypeError("Input must be torch.Tensor or np.ndarray")

    # --- Save logic (unchanged) ---
    suffix = save_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        pil_img.save(save_path, format="JPEG", quality=95)
    elif suffix == ".png":
        pil_img.save(save_path, format="PNG")
    else:
        pil_img.save(save_path)


##==============================================================================

@dataclass
class AlignmentResult:
    """Structured result for a single image alignment attempt."""
    success: bool
    aligned_tensor: Optional[torch.Tensor] = None   # (1, 3, 112, 112) or None
    score: Optional[float] = None                   # confidence score or None
    error_message: Optional[str] = None             # set on failure


class CVLFaceAligner:
    """
    Wraps the CVLFace DFA model for safe, device-aware single-image alignment.

    Usage:
        aligner = FaceAligner(model, device="cuda")
        result  = aligner.align(input_tensor)   # input_tensor: (1, 3, H, W)
    """

    def __init__(self, aligner_id=None, model=None, device: str = "cpu"):
        """
        Args:
            model:  The loaded CVLFace aligner (nn.Module from model_loader).
            device: 'cpu', 'cuda', or 'cuda:N'.
        """
        self.device = torch.device(device)
        if aligner_id:
            self.model = load_aligner(aligner_id=str(aligner_id)).to(self.device)
        elif model:
            self.model = model.to(self.device)

        self.model.eval()
        print(f"FaceAligner ready on device: {self.device}")

    @torch.no_grad()
    def align(self, input_tensor: torch.Tensor) -> AlignmentResult:
        """
        Run alignment on a single image tensor.

        Args:
            input_tensor: (1, 3, H, W) float32 in [-1, 1], RGB.

        Returns:
            AlignmentResult with success=True and aligned_tensor on success,
            or success=False with error_message on failure.
        """
        tensor = input_tensor.to(self.device)

        try:
            aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = self.model(tensor)
        except Exception as exc:
            # Common causes: no face detected, image too small/dark, CUDA OOM
            msg = f"{type(exc).__name__}: {exc}"
            print(f"Alignment failed — {msg}")
            return AlignmentResult(success=False, error_message=msg)

        # Validate output shape (sanity check)
        if aligned_x is None or aligned_x.shape[-2:] != (112, 112):
            msg = f"Unexpected output shape: {getattr(aligned_x, 'shape', None)}"
            print(f"Alignment produced unexpected output — {msg}")
            return AlignmentResult(success=False, error_message=msg)

        confidence = float(score[0]) if score is not None else None
        print(f"Alignment successful (score={confidence:.4f})" if confidence else "Alignment successful")

        return AlignmentResult(
            success=True,
            aligned_tensor=aligned_x.cpu(),   # move back to CPU for saving
            score=confidence,
        )

    @staticmethod
    def best_device() -> str:
        """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
        return "cuda" if torch.cuda.is_available() else "cpu"


    def process_single_image(self,
        image_path: Path,
        save_path: Path,
    ) -> tuple[str, Optional[str]]:
        """
        Process one image: load → align → save.

        Returns:
            (status, error_message) where status is 'skipped', 'success', or 'failed'.
        """
        # --- Load image ---
        tensor = load_image_as_tensor(image_path)
        if tensor is None:
            return "failed", "Could not open image file"

        # --- Align ---
        result: AlignmentResult = self.align(tensor)
        if not result.success:
            return "failed", result.error_message

        image_buffer = result.aligned_tensor

        # --- Save ---
        try:
            save_aligned_image(image_buffer, save_path)
        except Exception as exc:
            return "failed", f"Save error: {exc}"

        return "success", None


##==============================================================================

# ---------------------------------------------------------------------------
# Main batch pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_root: Path,
    save_root: Path,
    face_aligner: CVLFaceAligner
    ):
    """
    Run face alignment on all images under data_root, mirroring the
    directory structure into save_root.

    Args:
        data_root:     Source image directory (scanned recursively).
        save_root:     Destination root (created automatically).
        face_aligner:  Initialized FaceAligner instance.
        skip_existing: If True, skip images whose output already exists.
        log_every:     Print a progress line every N images.

    Returns:
        PipelineStats with counts and list of failed paths.
    """

    image_paths = collect_image_paths(data_root)

    success_count = 0
    failed_count = 0

    save_root.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(tqdm(image_paths), start=1):
        save_path = mirror_path(image_path, data_root, save_root)
        status, error = face_aligner.process_single_image(
            image_path, save_path,
        )

        if status == "success":
            success_count += 1
        else:
            failed_count += 1

    print(f"| {success_count} ok, {failed_count} failed")

    return None



def parse_args():
    parser = argparse.ArgumentParser(
        description="CVLFace face alignment — batch process images preserving directory structure."
    )
    parser.add_argument( "--data-root", type=Path, required=True, help="Root directory containing source images (searched recursively).")
    parser.add_argument( "--save-root", type=Path, required=True, help="Root directory where aligned images will be saved (structure mirrored).")
    parser.add_argument( "--aligner-id", type=str, default="minchul/cvlface_DFA_mobilenet",
            help=( "HuggingFace repo id for the aligner. Options: minchul/cvlface_DFA_mobilenet (default, fast) or minchul/cvlface_DFA_resnet50 (more accurate)." ))
    parser.add_argument( "--device", type=str, default=None, help="Compute device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detects if not set.")

    return parser.parse_args()



if __name__ == "__main__":


    args = parse_args()

    # Resolve device
    device = args.device if args.device else CVLFaceAligner.best_device()

    ## CVLab Face - Aligner
    face_aligner = CVLFaceAligner(aligner_id=str(args.aligner_id), device=device)

    # ## Aligner used in FaceAnonyMixer
    # face_aligner = FAM_FaceAligner(device=device)


    run_pipeline(
        data_root=args.data_root,
        save_root=args.save_root,
        face_aligner=face_aligner
    )



