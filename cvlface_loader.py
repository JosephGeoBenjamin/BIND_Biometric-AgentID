"""
CVLface model loader — works for ALL models (AdaFace, ArcFace, ViT variants).
Models are downloaded from HuggingFace and cached locally.
No CVLface repo install needed.
"""

import os, sys, shutil, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel
# from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms as tv_transforms

from PIL import Image


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
    cache_dir: str = "/egr/research-sprintai/benja161/.cache/cvlface_cache/",
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
    cache_dir: str = "/egr/research-sprintai/benja161/.cache/cvlface_cache/",
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




##==============================================================================

## Untested TODO: check this
# ── Setup landmark detector (ViTKPRPE) ───────────────────────────────────────

# import insightface

# detector = insightface.app.FaceAnalysis(allowed_modules=['detection'])
# detector.prepare(ctx_id=0, det_size=(112, 112))  # ctx_id=0 for GPU, -1 for CPU

# def get_keypoints(imgs_uint8_numpy: list[np.ndarray]) -> torch.Tensor:
#     """
#     Args:
#         imgs_uint8_numpy: list of B numpy arrays, each (112, 112, 3) uint8 BGR
#                           (insightface expects BGR, OpenCV format)
#     Returns:
#         keypoints: (B, 5, 2) float32 tensor  [x, y] per landmark
#     """
#     kps_batch = []
#     for img in imgs_uint8_numpy:
#         faces = detector.get(img)
#         if len(faces) == 0 or faces[0].kps is None:
#             # fallback: use canonical 112×112 landmark positions
#             kps = torch.tensor([
#                 [38.2946, 51.6963],   # left eye
#                 [73.5318, 51.5014],   # right eye
#                 [56.0252, 71.7366],   # nose tip
#                 [41.5493, 92.3655],   # left mouth
#                 [70.7299, 92.2041],   # right mouth
#             ], dtype=torch.float32)
#         else:
#             kps = torch.from_numpy(faces[0].kps).float()   # (5, 2)
#         kps_batch.append(kps)
#     return torch.stack(kps_batch)   # (B, 5, 2)




# ── In your dataloader loop ───────────────────────────────────────────────────

# imgs      : (B, 3, 112, 112) float tensor on device, already normalized
# imgs_path : list of image paths (or keep raw numpy around)

# Convert your normalized tensor back to uint8 BGR numpy for the detector
# def tensor_to_bgr_numpy(imgs_tensor: torch.Tensor) -> list[np.ndarray]:
#     """(B,3,112,112) float [-1,1] → list of (112,112,3) uint8 BGR"""
#     imgs_np = ((imgs_tensor.cpu() * 0.5 + 0.5) * 255).byte().numpy()  # (B,3,H,W) uint8
#     return [cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for img in imgs_np]


# # Forward pass with keypoints
# imgs_bgr = tensor_to_bgr_numpy(imgs)                        # list of numpy BGR
# keypoints = get_keypoints(imgs_bgr).to(device)              # (B, 5, 2)

# with torch.no_grad():
#     feats = cvl_vitkprpe_model(
#         cvl_vitkprpe_tf(imgs.to(device)),
#         keypoints=keypoints                                  # ← this is what was missing
#     ).cpu()

# cvl_vitkprpe_feats.append(feats)
