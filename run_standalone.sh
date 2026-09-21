### ALIGNER

# CUDA_VISIBLE_DEVICES=2 python cvlface_align_faces.py \
#     --data-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/imagesets/CelebA-HQ-frontal/images/ \
#     --save-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/imagesets/CelebA-HQ-frontal/aligned/ \
#     --aligner-id minchul/cvlface_DFA_mobilenet


### FEATURE EXTRACTOR

# --use-adaface-cvl \
# --use-arcface-cvl \
# --use-lafs \

# multipie, lfw-a, cfp-frontal, celeba-hq-frontal, casia-face

# CUDA_VISIBLE_DEVICES=1 python facefeature_extractor.py \
#     --dataset casia-face \
#     --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/face_features/ \
#     --batch-size 128 \
#     --use-adaface-cvl \
#     --cuda --verbose

# CUDA_VISIBLE_DEVICES=1 python facefeature_extractor.py \
#     --dataset casia-face \
#     --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/face_features/ \
#     --batch-size 128 \
#     --use-arcface-cvl \
#     --cuda --verbose

CUDA_VISIBLE_DEVICES=5 python facefeature_extractor.py \
    --dataset lfw-a \
    --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/face_features/ \
    --batch-size 128 \
    --use-vitkprpe-cvl \
    --cuda --verbose



CUDA_VISIBLE_DEVICES=5 python facefeature_extractor.py \
    --dataset cfp-frontal \
    --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/face_features/ \
    --batch-size 128 \
    --use-vitkprpe-cvl \
    --cuda --verbose

CUDA_VISIBLE_DEVICES=5 python facefeature_extractor.py \
    --dataset multipie \
    --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/face_features/ \
    --batch-size 128 \
    --use-vitkprpe-cvl \
    --cuda --verbose

CUDA_VISIBLE_DEVICES=5 python facefeature_extractor.py \
    --dataset celeba-hq-frontal \
    --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/face_features/ \
    --batch-size 128 \
    --use-vitkprpe-cvl \
    --cuda --verbose


CUDA_VISIBLE_DEVICES=5 python facefeature_extractor.py \
    --dataset casia-face \
    --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/face_features/ \
    --batch-size 128 \
    --use-vitkprpe-cvl \
    --cuda --verbose
