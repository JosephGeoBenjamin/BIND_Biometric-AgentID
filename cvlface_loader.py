"""
CVLface model loader — works for ALL models (AdaFace, ArcFace, ViT variants).
Models are downloaded from HuggingFace and cached locally.
No CVLface repo install needed.
"""

import os, sys, shutil, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel
from torchvision import transforms as tv_transforms

from PIL import Image

CACHE_DIR = "/egr/research-sprintai/benja161/.cache/cvlface_cache/"

# ── Download helpers ─────────────────────────────────────────────────────────

def _download_repo(repo_id: str, local_path: str, hf_token: str | None = None):
    """Download all files listed in files.txt plus the standard model files."""
    os.makedirs(local_path, exist_ok=True)
    files_txt = os.path.join(local_path, "files.txt")

    if not os.path.exists(files_txt):
        hf_hub_download(repo_id, "files.txt", token=hf_token,
                        local_dir=local_path, local_dir_use_symlinks=False)

    with open(files_txt) as f:
        extra_files = [l.strip() for l in f.read().split("\n") if l.strip()]

    must_have = ["config.json", "wrapper.py", "model.safetensors"]
    for fname in extra_files + must_have:
        dest = os.path.join(local_path, fname)
        if not os.path.exists(dest):
            print(f"  downloading {fname} …")
            hf_hub_download(repo_id, fname, token=hf_token,
                            local_dir=local_path, local_dir_use_symlinks=False)


def _load_from_local(local_path: str, hf_token: str | None = None):
    """Load an AutoModel from a locally-downloaded CVLface repo folder."""
    cwd = os.getcwd()
    os.chdir(local_path)
    sys.path.insert(0, local_path)
    try:
        model = AutoModel.from_pretrained(
            local_path, trust_remote_code=True, token=hf_token
        )
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    return model


def load_cvlface_model(
    repo_id: str,
    cache_dir: str = CACHE_DIR,
    hf_token: str | None = None,
    force_download: bool = False,
) -> torch.nn.Module:
    """
    Main entry point. Downloads (once) and returns the model ready for eval.

    Args:
        repo_id      : HuggingFace repo, e.g. 'minchul/cvlface_adaface_ir101_webface4m'
        cache_dir    : local cache root
        hf_token     : HF token (only needed if repo is gated; these are public)
        force_download: wipe cache and re-download
    """
    local_path = os.path.join(cache_dir, repo_id.replace("/", "_"))

    if force_download and os.path.exists(local_path):
        shutil.rmtree(local_path)

    print(f"Loading: {repo_id}")
    _download_repo(repo_id, local_path, hf_token)
    model = _load_from_local(local_path, hf_token)
    model.eval()
    return model



def load_aligner(
    aligner_id: str = "minchul/cvlface_DFA_mobilenet",
    cache_dir: str = CACHE_DIR,
    hf_token: str | None = None,
    force_download: bool = False,
):
    """
    Public entry point: download (if needed) and return the aligner model.

    Args:
        aligner_id:  HuggingFace repo id, e.g. 'minchul/cvlface_DFA_mobilenet'
        cache_dir:   Local directory to cache the downloaded model files.
        hf_token:    Optional HuggingFace token for private repos.

    Returns:
        aligner model (nn.Module), ready for inference (eval mode, on CPU).
    """
    # Use a flat subfolder name derived from the repo id
    model_folder_name = aligner_id.replace("/", "__")
    local_path = os.path.join(cache_dir, model_folder_name)

    if force_download and os.path.exists(local_path):
        shutil.rmtree(local_path)

    _download_repo(aligner_id, local_path, hf_token)
    aligner = _load_from_local(local_path, hf_token)
    return aligner


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load a face image (already aligned/cropped to ~112×112) and return
    a (1, 3, 112, 112) tensor ready for inference.
    Input image should already be face-aligned. If not, align first.
    """
    transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(image_path).convert("RGB").resize((112, 112))
    return transform(img).unsqueeze(0)



def get_cvlface_transform(input_size=None, tensorize=True):
    """
    CVLface preprocessing for ArcFace IR101, AdaFace IR101, and ViT KPRPE.
    All three models share identical preprocessing: 112×112, RGB, normalized to [-1, 1].

    Args:
        input_size: if your VGGFace2 dataloader hasn't resized yet, pass (H, W) or int.
                    If already 112×112, leave as None.
    """
    ops = []
    if input_size is not None:
        size = (input_size, input_size) if isinstance(input_size, int) else input_size
        ops.append(tv_transforms.Resize(size))

    if tensorize:
        ops += [tv_transforms.ToTensor(),                             # [0,255] HWC → [0,1] CHW
                ]

    ops += [ tv_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),         # [0,1] → [-1, 1]
            ]

    return tv_transforms.Compose(ops)


##=============================================================================
# ── ViTKPRPE feature extractor ────────────────────────────────────────────────
#
# Wraps the recognition model + DFA aligner into one callable object.
#
# Usage pattern (see extract_features.py):
#
#   extractor = KPRPEExtractor(device=device)
#   feats = extractor(imgs)          # imgs: (B,3,112,112) float [0,1] tensor
##=============================================================================

# Fallback canonical 5-point landmarks for a 112×112 arcface-aligned crop,
# used when the aligner fails to detect a face in a given image.
_CANONICAL_KPS = torch.tensor([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=torch.float32)   # (5, 2)


class KPRPEExtractor(torch.nn.Module):
    """
    Self-contained extractor for the CVLface ViT-KPRPE recognition model.

    Bundles the DFA aligner (for landmark detection on already-aligned 112×112
    crops) and the KPRPE recognition model into one callable.  Landmarks that
    the aligner fails to detect fall back to canonical positions.

    Args:
        recognition_id : HuggingFace repo for the KPRPE recognition model.
        aligner_id     : HuggingFace repo for the DFA landmark aligner.
        cache_dir      : local model cache root.
        hf_token       : optional HuggingFace token.
        device         : 'cuda' or 'cpu'.

    Example::

        extractor = KPRPEExtractor(device='cuda')
        extractor.to('cuda')

        # imgs: (B, 3, 112, 112) float32 in [0, 1]  (raw dataloader ToTensor output)
        feats = extractor(imgs)   # → (B, D) on CPU
    """

    def __init__(
        self,
        recognition_id: str = "minchul/cvlface_adaface_vit_base_kprpe_webface4m",
        aligner_id:     str = "minchul/cvlface_DFA_mobilenet",
        cache_dir:      str = CACHE_DIR,
        hf_token:       str | None = None,
        device:         str = "cuda",
    ):
        super().__init__()
        self.device    = device

        print("Loading KPRPE recognition model …")
        self.recognizer = load_cvlface_model(recognition_id, cache_dir, hf_token)
        self.recognizer.eval().to(device)

        print("Loading DFA aligner …")
        self.aligner = load_aligner(aligner_id, cache_dir, hf_token)
        self.aligner.eval().to(device)

    @torch.no_grad()
    def get_keypoints(self, imgs_norm: torch.Tensor) -> torch.Tensor:
        """
        Run the DFA aligner on a batch of normalised 112×112 crops and return
        the detected landmarks.  Per-image fallback to canonical kps on failure.

        Args:
            imgs_norm : (B, 3, 112, 112) float32 in [-1, 1], on self.device.

        Returns:
            keypoints : (B, 5, 2) float32 on self.device.
        """
        _, orig_ldmks, _, scores, _, _ = self.aligner(imgs_norm)
        # orig_ldmks: (B, 5, 2);  scores: (B,)
        # Replace per-image keypoints where the aligner had low confidence.
        canonical = _CANONICAL_KPS.unsqueeze(0).expand_as(orig_ldmks).to(self.device)
        failed    = (scores < 0.5).view(-1, 1, 1).expand_as(orig_ldmks)
        keypoints = torch.where(failed, canonical, orig_ldmks)
        return keypoints

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract KPRPE features for a batch of pre-aligned 112×112 images.

        Args:
            imgs : (B, 3, 112, 112) float32 in [0, 1]  (raw ToTensor output).

        Returns:
            (B, D) feature tensor on CPU.
        """
        keypoints = self.get_keypoints(imgs)              # (B,5,2)
        feats     = self.recognizer(imgs, keypoints=keypoints)
        return feats.cpu()