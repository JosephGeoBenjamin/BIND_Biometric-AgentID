# BIND — Binding Biometrics with AI Agent Identifiers for Delegation of Authority

[![Paper](https://img.shields.io/badge/Paper-arXiv:2608.04292-red)](https://arxiv.org/abs/2608.04292)

> **Joseph Geo Benjamin, Anil K. Jain, Karthik Nandakumar**

Accepted at the IEEE/IAPR International Joint Conference on Biometrics (IJCB) Sessions 2026

Reference implementation of the **BIND** framework.

---

## Abstract

The proliferation of agentic artificial intelligence (AI) systems has raised
serious questions about the accountability for tasks performed by AI agents.
Ideally, an AI agent must not be allowed to perform critical tasks without
explicit authorization by a human operator. Since biometric recognition is one
of the most reliable approaches for authenticating individuals, it has the
potential to enable authenticated delegation of authority to AI agents. In this
work, we present a framework called **BIND**, which leverages ideas from the
field of biometric cryptosystems, to securely bind biometric data of the human
user to the AI agent identity (ID) and authority scope (task-specific
constraints) at the time of agent authorization. This token/identifier can be
presented by the AI agent to an Identity Auditor, who simultaneously performs
biometric authentication and recovers the agent ID and scope, thereby enabling
real-time user authentication and establishing a non-repudiable proof of human
control and delegation of authority. We also provide a practical implementation
of the proposed BIND framework based on face features extracted using standard
deep neural network models. To facilitate this implementation, we propose a
feature adaptation module that transforms real-valued feature embeddings into
fixed-length binary representations suitable for a fuzzy commitment construct
based on turbo error correcting codes. Experiments demonstrate the practical
feasibility of the proposed face cryptosystem, achieving a True Match Rate of
96% at zero False Match Rate and supporting 1024-bit agent tokens.

### Framework

![BIND framework block diagram](https://github.com/JosephGeoBenjamin/BIND_Biometric-AgentID/releases/download/v0.1/flow-diagram.png)

---

## Pipeline

```
images ──► align ──► extract features ──► bind token ──► recover + score
         (Step 1)      (Step 2)              (Step 3)
```

---

## Repository layout

| File | Role |
|---|---|
| `cvlface_align_faces.py` | Batch face alignment to 112×112 crops (CLI) |
| `facefeature_extractor.py` | Batch feature extraction across recognition models (CLI) |
| `cvlface_loader.py` | Loads CVLface recognition + alignment models from HuggingFace |
| `binder-protocol.py` | Feature adaptation, fuzzy commitment, experiment driver, scoring |
| `turbocode.py` | Standalone NumPy turbo encoder/decoder with rate extensions |
| `run_standalone.sh` | Runs Steps 1 and 2 across datasets |

CVLface models are fetched on first use and cached under `CACHE_DIR` in
`cvlface_loader.py`.

### Notebooks

Analysis notebooks and supplementary code are published with the
[v0.1 release](https://github.com/JosephGeoBenjamin/BIND_Biometric-AgentID/releases/tag/v0.1).

---

## Installation

```bash
conda create -n agentid python=3.14
conda activate agentid

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

`sionna-no-rt` supplies the turbo codec; `timm` / `transformers` /
`huggingface-hub` cover the CVLface models. The reported experiments were run
against `sionna-no-rt` 2.0.1 and torch 2.11.0+cu128.

---

## Configuring paths

Dataset and output roots are set in the source rather than a config file.
Before running on a new machine, update:

| Location | Constant |
|---|---|
| `facefeature_extractor.py` | `DATASETS` — one entry per dataset root |
| `binder-protocol.py` (`__main__`) | `ROOT_PATH` — where extracted features live |
| `binder-protocol.py` (`__main__`) | `SAVE_PATH` — where results are written |
| `cvlface_loader.py` | `CACHE_DIR` — HuggingFace model cache |

Each dataset root holds an `aligned/` sub-folder, one directory per identity:

```
<dataset-root>/
└── aligned/
    ├── <identity_001>/
    │   ├── img_01.jpg
    │   └── img_02.jpg
    └── <identity_002>/
        └── img_01.jpg
```

Identity is read from the parent folder name, which is what makes genuine and
impostor pairing possible downstream.

---

## Step 1 — Align

Crops faces to the 112×112 geometry the recognition models expect, mirroring
the input directory structure into the output root.

```bash
CUDA_VISIBLE_DEVICES=0 python cvlface_align_faces.py \
    --data-root /path/to/dataset/images/ \
    --save-root /path/to/dataset/aligned/ \
    --aligner-id minchul/cvlface_DFA_mobilenet
```

| Argument | Default | Notes |
|---|---|---|
| `--data-root` | required | Searched recursively for images |
| `--save-root` | required | Directory structure is mirrored |
| `--aligner-id` | `minchul/cvlface_DFA_mobilenet` | `minchul/cvlface_DFA_resnet50` is more accurate, slower |
| `--device` | auto | Falls back to CPU when CUDA is unavailable |

Images that fail alignment are counted and reported in the closing summary.

---

## Step 2 — Extract features

Runs one or more recognition backbones over an aligned dataset and saves an
embedding matrix plus a parallel filename list.

```bash
CUDA_VISIBLE_DEVICES=0 python facefeature_extractor.py \
    --dataset lfw-a \
    --output-root /path/to/data_extracts/face_features/ \
    --batch-size 128 \
    --use-adaface-cvl \
    --cuda --verbose
```

| Flag | Model |
|---|---|
| `--use-adaface-cvl` | AdaFace IResNet-101 (WebFace4M) |
| `--use-arcface-cvl` | ArcFace IResNet-101 (WebFace4M) |
| `--use-vitkprpe-cvl` | KPRPE-AdaFace ViT-Base (WebFace4M) |

Flags combine, so several backbones can be extracted in one pass.

Output per dataset, under `<output-root>/<dataset>/`:

```
cvl_adaface_features.pt     # [N, 512] embedding matrix
image_filenames.txt         # N lines, "<identity>/<image>", row-aligned to the .pt
```

The two files are positional — row *i* of the tensor corresponds to line *i* of
the text file, and Step 3 relies on this to group embeddings by identity.

### Running both steps from the shell script

`run_standalone.sh` wraps Steps 1 and 2, resolving each dataset's image root
and passing it through explicitly:

```bash
./run_standalone.sh              # extract features for every dataset
./run_standalone.sh align        # align one dataset
./run_standalone.sh all          # align, then extract
```

Settings are overridable from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `GPU` | `0` | Value for `CUDA_VISIBLE_DEVICES` |
| `MODEL` | `--use-vitkprpe-cvl` | Recognition backbone flag |
| `DATASETS` | all five | Space-separated dataset keys to extract |
| `BATCH_SIZE` | `128` | Extraction batch size |
| `OUTPUT_ROOT` | `<project>/datasets/data_extracts/face_features` | Where features are written |
| `ALIGN_DATASET` | `celeba-hq-frontal` | Dataset to align in `align` mode |

```bash
GPU=3 MODEL=--use-adaface-cvl DATASETS="lfw-a cfp-frontal" ./run_standalone.sh
```

A dataset missing its `aligned/` folder is skipped with a warning, and one
failure does not abort the remaining datasets — the run ends with a summary of
which keys failed and a non-zero exit status.

---

## Step 3 — Bind and score

`binder-protocol.py` is the experiment driver. It is configured through
constants in its `__main__` block and run directly:

```bash
CUDA_VISIBLE_DEVICES=0 python binder-protocol.py
```

Before launching, populate `LMBDA_DICT` in `__main__` — it ships with its
entries commented out, so the run loop does nothing until it has at least one.
Each key reads `<model>_<dataset>_<rate>_<setting>`, and `ROOT_PATH` /
`SAVE_PATH` set the input features and output root.

### Outputs

Per configuration, under `<SAVE_PATH>/<dataset>/<config>/<bits>-bits/`:

```
<model>-message_match_error_sum.pkl   # {model: [genuine_errors, impostor_errors]}
<model>-biom_analysis.pkl             # per-stage similarity scores
binded-msg-recovery-sum.png           # token error histogram, TMR/FNMR/FMR
biometric-similarities.png            # similarity histograms, TAR@FAR table
```

Token scores are Hamming *error counts*: zero means the token was recovered
exactly. True Match Rate is the fraction of genuine attempts scoring zero,
False Match Rate the fraction of impostor attempts scoring zero.

The output directory is created with `exist_ok=False`, so re-running a
configuration that has already been written needs a fresh `SAVE_PATH`.

---

## Turbo codes

`turbocode.py` is a standalone NumPy turbo encoder/decoder. Run it directly for
the demos:

```bash
python turbocode.py
```

---

## Citation

```bibtex
@article{benjamin2026bind,
  title   = {Binding Biometrics with AI Agent Identifiers for Delegation of Authority},
  author  = {Benjamin, Joseph Geo and Jain, Anil K. and Nandakumar, Karthik},
  journal = {arXiv preprint arXiv:2608.04292},
  year    = {2026}
}
```

---

## Acknowledgements

- Recognition and alignment models from [CVLface](https://github.com/mk-minchul/CVLface)
- Turbo encoder/decoder from [NVIDIA Sionna](https://github.com/NVlabs/sionna)
