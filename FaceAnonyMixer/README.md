# FaceAnonyMixer: Cancelable Faces via Identity Consistent Latent Space Mixing

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2508.05636)
[![Conference](https://img.shields.io/badge/IJCB-2025-blue)](https://ijcb2025.ieee-biometrics.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Mohammed Talha Alam¹, Fahad Shamshad¹, Fakhri Karray¹², Karthik Nandakumar¹³**
>
> ¹ Mohamed Bin Zayed University of Artificial Intelligence, UAE  
> ² University of Waterloo, Canada · ³ Michigan State University, USA  
> `{mohammed.alam, fahad.shamshad, fakhri.karray, karthik.nandakumar}@mbzuai.ac.ae`

**Accepted at IEEE International Joint Conference on Biometrics (IJCB) 2025**

---

## Overview

FaceAnonyMixer generates privacy-preserving face images by **mixing a real face's W+ latent code with a key-derived synthetic code** inside StyleGAN2's latent space. It satisfies all four ISO/IEC 24745 requirements simultaneously:

| Requirement | What it means |
|---|---|
| **Revocability** | Change the key to revoke and replace any compromised template |
| **Unlinkability** | Templates from different keys cannot be cross-matched |
| **Irreversibility** | Original identity cannot be recovered even if the key and template are both known |
| **Performance Preservation** | Recognition accuracy on protected faces matches unprotected baselines |

> **Note** — this is a vendored copy of the upstream FaceAnonyMixer release,
> adapted for use inside the Bind-AgentID project. Dataset roots point at local
> paths, a face-pose filtering stage has been added, and the pipeline is driven
> by `run_pipelines.sh`. See the parent `../README.md` for how the protected
> images feed back into the binding protocol.

## Repository Structure

```
FaceAnonyMixer/
│
├── anonymize.py              # Core: latent mixing + multi-loss optimization
├── invert.py                 # GAN inversion via e4e + Pivot Tuning
├── create_fake_dataset.py    # Generate fake StyleGAN2 image pool
├── extract_features.py       # Extract CLIP/FaRL/DINO/ArcFace features for real images
├── pair_unique.py            # Pair each real identity to a unique fake identity ← use this
├── pair_nn.py                # Random per-image pairing (ablation only)
├── download_pretrained.py    # Download all pretrained weights
├── run_pipelines.sh          # End-to-end pipeline invocations
│
├── lib/
│   ├── __init__.py
│   ├── config.py             # Dataset paths + model URLs  ← update DATASETS here
│   ├── celebahq.py           # CelebA-HQ dataset class
│   ├── vggface2.py           # VGGFace2 dataset class
│   ├── latent_code.py        # LatentCode nn.Module
│   ├── id_loss.py            # Anonymity loss (ArcFace cosine)
│   ├── attr_loss.py          # Attribute loss (FaRL/CLIP/DINO)
│   ├── cons_loss.py          # Identity preservation loss
│   ├── augmentations.py      # ImageAugmenter
│   ├── aligner.py            # Face alignment (face_alignment library)
│   ├── arcface.py            # ArcFace feature extractor
│   ├── facepose.py           # MediaPipe frontal-pose filter (CLI)
│   ├── collate_fn.py         # DataLoader collate helper
│   └── aux.py                # tensor2image, anon_exp_dir, DataParallelPassthrough
│
├── models/
│   ├── load_generator.py     # Build + load StyleGAN2 from GenForce
│   ├── psp.py                # e4e / pSp encoder wrapper
│   ├── encoders/
│   │   ├── helpers.py        # Shared IR bottleneck blocks
│   │   ├── model_irse.py     # IR-SE backbone
│   │   └── psp_encoders.py   # GradualStyleEncoder, Encoder4Editing
│   ├── stylegan2/            # StyleGAN2 ops (incl. CUDA extensions under op/)
│   ├── genforce/             # GenForce model definitions
│   └── pretrained/           # Downloaded weights (e4e, farl, sfd, genforce)
│
├── utils1/
│   ├── __init__.py
│   ├── ImagesDataset.py      # Simple flat image dataset
│   └── data_utils.py         # make_dataset helper
│
├── datasets/                 # Pipeline outputs (inv, features, fake, anonymised)
└── README.md
```

---

## Installation

Dependencies are installed from the parent project's `../requirements.txt`,
which covers this pipeline as well as the binding protocol.

```bash
# 1. Install Python dependencies (from the parent directory)
pip install -r ../requirements.txt

# 2. Download pretrained weights
#    (StyleGAN2-FFHQ-1024/512, e4e, ArcFace, FaRL ep16+ep64, SFD detector)
python download_pretrained.py
```

Weights land under `models/pretrained/`, and the GenForce model definitions are
already vendored under `models/genforce/`.

`lib/facepose.py` uses MediaPipe. The parent `requirements.txt` notes
`mediapipe==0.10.14` as the working version for this stage.

Set `PYTHONPATH` to this directory before running any script, as
`run_pipelines.sh` does:

```bash
export PYTHONPATH="/path/to/Bind-AgentID/FaceAnonyMixer/:$PYTHONPATH"
```

---

## Dataset Preparation

Datasets must be organised as **one sub-folder per identity**:

```
/path/to/dataset/
└── train/
    ├── n000001/
    │   ├── 0001.jpg
    │   └── 0002.jpg
    └── n000002/
        └── 0001.jpg
```

Edit `DATASETS` in `lib/config.py` to point to your local paths. The keys
defined there are the values accepted by every script's `--dataset` flag:

```python
DATASETS = {
    'celebahq':       '/path/to/CelebA-HQ-images/',
    'celebahq-front': '/path/to/CelebA-HQ-frontal',
    'celebahq-trial': '/path/to/SubSet-for-test/',
}
```

`celebahq-trial` is a small subset useful for verifying the pipeline end to end
before committing to a full run.

---

## Step-by-Step Usage

`run_pipelines.sh` holds the full sequence with working arguments. The steps
below mirror it.

### Step 1 — GAN Inversion

Project real faces into W+ space using e4e + Pivot Tuning Inversion.

```bash
CUDA_VISIBLE_DEVICES=0 python invert.py \
    --dataset celebahq-front \
    --batch-size 1 \
    --save-reconstructed-images \
    --save-aligned-images \
    --cuda --verbose
```

Output: `datasets/inv/<dataset>/`, alongside `alignment_errors.txt` and
`face_detection_errors.txt` listing any images that could not be processed.

---

### Step 2 — Extract Real Dataset Features

```bash
CUDA_VISIBLE_DEVICES=0 python extract_features.py \
    --dataset celebahq-front \
    --batch-size 128 \
    --cuda --verbose
```

Output: `datasets/features/<dataset>/` — one `.pt` per feature space
(`clip`, `farl`, `dino`, `arcface`) plus `image_filenames.txt`.

Individual feature spaces can be skipped with `--no-clip`, `--no-farl`,
`--no-dino`, or `--no-arcface`.

---

### Step 3 — Generate Fake Dataset

Sample a pool of synthetic faces with matching feature embeddings.

```bash
CUDA_VISIBLE_DEVICES=0 python create_fake_dataset.py \
    --gan stylegan2_ffhq1024 \
    --num-samples 1000 \
    --truncation 0.7 \
    --cuda --verbose
```

Output: `datasets/fake/fake_dataset_stylegan2_ffhq1024/` — one directory per
sample, keyed by latent-code hash, each holding `image.jpg`, `latent_code_w+.pt`,
`latent_code_s.pt`, and the four feature tensors.

---

### Step 4 — Filter by Pose

Keep only near-frontal synthetic faces, so the key identities are well posed.

```bash
python lib/facepose.py \
    --inp "datasets/fake/fake_dataset_stylegan2_ffhq1024/*/*.jpg" \
    --out datasets/fake/Fake_Filtered/
```

Output: `datasets/fake/Fake_Filtered/` carrying forward the same per-sample
directory layout. Quote the `--inp` glob so the shell passes it through intact.

---

### Step 5 — Pair Identities

Assign each real identity a unique fake identity. The paired fake latent is the
revocable key — changing it revokes and replaces the protected template.

```bash
python pair_unique.py \
    --real-dataset celebahq-front \
    --fake-dataset-root datasets/fake/Fake_Filtered \
    --verbose
```

Output: `random_nn_map_<dataset>.json` inside the fake dataset directory.
`--seed` (default 42) makes the assignment reproducible.

`pair_nn.py` offers random per-image pairing instead, for ablation only.

---

### Step 6 — Anonymize

```bash
CUDA_VISIBLE_DEVICES=0 python anonymize.py \
    --dataset celebahq-front \
    --fake-nn-map datasets/fake/Fake_Filtered/random_nn_map_celebahq-front.json \
    --latent-space W+ \
    --epochs 50 \
    --lr 0.01 \
    --lambda-id 10.0 \
    --lambda-attr 0.15 \
    --lambda-consistency 10.0 \
    --cuda --gpu-id 0 --verbose
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Optimisation steps per identity group |
| `--lr` | 0.01 | Learning rate |
| `--optim` | adam | Optimiser |
| `--lambda-id` | 10.0 | Weight for `L_anon` |
| `--lambda-attr` | 0.15 | Weight for `L_attr` |
| `--lambda-consistency` | 10.0 | Weight for `L_idp` |
| `--id-margin` | 0.0 | Cosine margin in `L_anon` (0 = max anonymisation) |
| `--latent-space` | W+ | Latent space used for mixing |

Output: `datasets/anonymised/<dataset>/`, with an `args.json` recording the
settings used for the run.

## Citation

```bibtex
@article{alam2025faceanonymixer,
  title={FaceAnonyMixer: Cancelable Faces via Identity Consistent Latent Space Mixing},
  author={Alam, Mohammed Talha and Shamshad, Fahad and Karray, Fakhri and Nandakumar, Karthik},
  journal={arXiv preprint arXiv:2508.05636},
  year={2025}
}
```

---

## Acknowledgements

Built on [FALCO](https://github.com/chi0tzp/FALCO) · GAN inversion via [e4e](https://github.com/omertov/encoder4editing) · Generator from [GenForce/StyleGAN2](https://github.com/genforce/genforce) · Identity features from [ArcFace](https://github.com/deepinsight/insightface) · Attribute features from [FaRL](https://github.com/FacePerceiver/FaRL)

---

## License

MIT License, per the [upstream release](https://github.com/talha-alam/faceanonymixer).
The vendored GenForce components under `models/genforce/` carry their own
licence in `models/genforce/LICENSE`.
