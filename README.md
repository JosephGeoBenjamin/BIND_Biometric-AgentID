# Bind-AgentID

Binding a cryptographic message to a face template, so that the message can be
recovered from a fresh capture of the same person and not from anyone else.

The repository covers the full path from raw face images to a measured
message-recovery rate:

```
images ──► align ──► extract features ──► bind message ──► recover + score
```

A face embedding is projected into a binary string, that string is XOR-combined
with an error-correcting-code encoding of a secret message, and the result is
stored. Presenting a second image of the same identity reproduces a nearby
binary string; the error-correcting decoder absorbs the difference and returns
the original message. A different identity produces a distant string and the
decoder fails, which is the intended behaviour.

---

## Repository layout

| File | Role |
|---|---|
| `cvlface_align_faces.py` | Batch face alignment to 112×112 crops (CLI) |
| `facefeature_extractor.py` | Batch feature extraction across recognition models (CLI) |
| `cvlface_loader.py` | Loads CVLface recognition + alignment models from HuggingFace |
| `lafs_loader.py` | Loads LAFS Part-fViT models from a local checkpoint |
| `binder-protocol.py` | The binding protocol, experiment driver, and scoring |
| `turbocode.py` | Standalone NumPy Turbo encoder/decoder with rate extensions |
| `ldpcode_sionna.py` | LDPC encoder/decoder over a custom parity-check matrix |
| `train_features_adaptation.py` | Autoencoder study on embedding noise tolerance |
| `run_standalone.sh` | Recorded alignment and extraction invocations |
| `FaceAnonyMixer/` | Cancelable-face generation (see its own README) |

### Supporting directories

- `pretrained/` — local checkpoints, currently the LAFS fine-tuned weights.
- CVLface models are fetched on first use and cached under the path set by
  `CACHE_DIR` in `cvlface_loader.py`.

---

## Installation

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

`requirements.txt` pulls in the recognition and coding stack: `sionna-no-rt`
for the Turbo and LDPC codecs, `timm` / `transformers` / `huggingface-hub` for
the CVLface models, `face-alignment` and `clip` for the auxiliary feature
spaces, plus the usual scientific Python set.

---

## Configuring paths

Dataset and output roots are set directly in the source rather than through a
config file. Before running on a new machine, update:

| Location | Constant |
|---|---|
| `facefeature_extractor.py` | `DATASETS` — one entry per dataset root |
| `binder-protocol.py` (`__main__`) | `ROOT_PATH` — where extracted features live |
| `binder-protocol.py` (`__main__`) | `SAVE_PATH` — where results are written |
| `train_features_adaptation.py` (`main`) | `SAVEPATH` |
| `cvlface_loader.py` | `CACHE_DIR` — HuggingFace model cache |

Each dataset root is expected to hold an `aligned/` sub-folder organised as one
directory per identity:

```
<dataset-root>/
└── aligned/
    ├── <identity_001>/
    │   ├── img_01.jpg
    │   └── img_02.jpg
    └── <identity_002>/
        └── img_01.jpg
```

Identity is read from the parent folder name, so this layout is what makes
genuine and impostor pairing possible downstream.

---

## Step 1 — Align

Detects and crops faces to the 112×112 geometry the recognition models expect,
mirroring the input directory structure into the output root.

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

Backbone flags, any combination:

| Flag | Model |
|---|---|
| `--use-adaface-cvl` | AdaFace IResNet-101 (WebFace4M) |
| `--use-arcface-cvl` | ArcFace IResNet-101 (WebFace4M) |
| `--use-vitkprpe-cvl` | KPRPE-AdaFace ViT-Base (WebFace4M) |
| `--use-lafs` | LAFS Part-fViT, from `pretrained/` |
| `--use-clip` | CLIP ViT-B/16 |
| `--use-dino` | DINO ViT-B/16 |

Output per dataset, under `<output-root>/<dataset>/`:

```
cvl_adaface_features.pt     # [N, 512] embedding matrix
image_filenames.txt         # N lines, "<identity>/<image>", row-aligned to the .pt
```

The two files are positional — row *i* of the tensor corresponds to line *i* of
the text file. `vectors_loader()` in `binder-protocol.py` relies on this to
group embeddings by identity.

The KPRPE backbone additionally runs the DFA aligner to obtain landmarks, and
falls back to canonical 5-point positions when detection confidence is low.

---

## Step 3 — Bind and score

`binder-protocol.py` is the experiment driver. Rather than a CLI, it is
configured through constants in its `__main__` block and run directly:

```bash
CUDA_VISIBLE_DEVICES=0 python binder-protocol.py
```

### How the binding works

`BiomBinder_Method` holds the protocol.

**Key generation** (`generate_cypherkey`) draws a random projection matrix, an
optional bias vector scaled by `lmbda`, and a random index permutation. This
triple is the revocable key — regenerating it yields an unlinkable template
from the same face.

**Embedding to bits** (`proc_to_bits`) normalises the embedding to a fixed
norm, adds the bias, projects it into a 2×B space, gathers the permuted
indices, and binarises by comparing the two halves. The result is a B-bit
string.

**Binding** (`compile_message`) Turbo-encodes the secret message to B bits and
XORs it with the biometric string. **Recovery** (`extract_message`) XORs a
fresh biometric string back out and hands the result to the Turbo decoder as
LLRs.

### Configuration

Set in `__main__`:

| Constant | Meaning |
|---|---|
| `LMBDA_DICT` | Experiment set: one entry per `<model>_<dataset>_<rate>_<setting>`, valued by the bias scale λ |
| `keys_mapper` | Maps the short codes in those keys to feature filenames, dataset folders, and rate denominators |
| `model_filter` | Feature files to load, mapped to display names for plots |
| `bc` loop | Message sizes in bits (e.g. `[4096]`) |
| `ROOT_PATH` / `SAVE_PATH` | Input features and output root |

A key such as `kprpe_pie_T_09` reads as KPRPE-AdaFace on Multi-PIE, rate `T`
(1/2), setting `09`. The message size and the rate denominator together fix the
biometric string length: `biom_bit_size = msg_bit_size × rate_denom`.

`LMBDA_DICT` ships with its entries commented out, so populate it with the
configurations to run before launching.

### Sampling controls

| Parameter | Effect |
|---|---|
| `genuine_1VSrest` | `False` pairs every image against every other within an identity; `True` uses one anchor against the rest |
| `max_tnsr_capacity` | Per-identity image cap; identities above it switch to anchor-vs-rest to bound memory |
| `imposter_max` | Images taken per identity when forming impostor pairs |

Multi-PIE carries roughly 520 images per identity, so the capacity cap governs
there and keeps the comparison count tractable. LFW-a and CFP-frontal sit
mostly below the cap.

Identities with only one image are dropped during loading, since a genuine pair
cannot be formed from them.

### Outputs

Per configuration, under `<SAVE_PATH>/<dataset>/<config>/<bits>-bits/`:

```
<model>-message_match_error_sum.pkl   # {model: [genuine_errors, impostor_errors]}
<model>-biom_analysis.pkl             # per-stage similarity scores
binded-msg-recovery-sum.png           # message error histogram, TAR/FRR/FAR
biometric-similarities.png            # similarity histograms, TAR@FAR table
```

The `biom_analysis` pickle tracks similarity at each stage of the transform —
`raw`, `norm`, `bias`, `proj`, `binary` — which makes it possible to see where
in the pipeline genuine and impostor distributions separate.

Message scores are Hamming *error counts*: zero means the message was
recovered exactly. TAR is therefore the fraction of genuine attempts scoring
zero, and FAR the fraction of impostor attempts scoring zero.

Note that `os.makedirs(..., exist_ok=False)` guards the output directory, so a
configuration that has already been written will need a fresh `SAVE_PATH`.

---

## Coding schemes

Two error-correcting backends are available.

**Turbo** — the binder uses Sionna's `TurboEncoder` / `TurboDecoder` at rate
1/2 or 1/3, constraint length 4, 6 decoder iterations, hard output.
`turbocode.py` is a self-contained NumPy implementation of the same family,
with repetition and puncturing layers that reach rates from 1/2 down to 1/30.
Run it directly to see the demos:

```bash
python turbocode.py
```

**LDPC** — `ldpcode_sionna.py` builds a sparse Gallager-style parity-check
matrix, derives a systematic generator over GF(2), and decodes with Sionna's
belief-propagation decoder. It carries its own test suite:

```bash
python ldpcode_sionna.py
```

This runs orthogonality, systematic-position, valid-codeword, and
noise-tolerance checks, then a rate-1/10 demo at 25% bit-flip.

---

## Feature adaptation study

`train_features_adaptation.py` trains an autoencoder over face embeddings under
graded input noise, measuring how cosine similarity and reconstruction error
degrade. It informs how much perturbation the binding stage can absorb before
message recovery fails.

```bash
python train_features_adaptation.py
```

Configured through constants in `main()`: bottleneck shape, training steps,
batch size, learning rate, and the noise sweep.

---

## FaceAnonyMixer

`FaceAnonyMixer/` is a vendored copy of the IJCB 2025 cancelable-face method,
which generates privacy-preserving face images by mixing a real W+ latent code
with a key-derived synthetic one. It runs as an independent pipeline with its
own dataset roots and pretrained weights — see `FaceAnonyMixer/README.md`.

Within this project it supplies protected face images that can be fed back
through alignment and extraction, letting the binding protocol be measured on
cancelable templates as well as on unprotected ones.

---

## References

- CVLface recognition and alignment models — https://github.com/mk-minchul/CVLface
- NVIDIA Sionna — https://github.com/NVlabs/sionna
- FaceAnonyMixer, IJCB 2025 — https://arxiv.org/abs/2508.05636
