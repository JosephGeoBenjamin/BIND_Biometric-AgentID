### FEATURE EXTRACTOR

# --no-clip --no-farl \
# --no-dino --no-arcface \
# --no-arcface-cvl  \
# --no-adaface-cvl \

CUDA_VISIBLE_DEVICES=5 python extract_features.py \
    --dataset celebahq-front \
    --dataset-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/FaceAnonyMixer/datasets/anonymised/celebahq-front \
    --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/features_Anonym/ \
    --batch-size 128 \
    --no-clip --no-farl \
    --no-dino --no-arcface \
    --no-arcface-cvl \
    --cuda --verbose

CUDA_VISIBLE_DEVICES=5 python extract_features.py \
    --dataset celebahq-front \
    --dataset-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/FaceAnonyMixer/datasets/anonymised/celebahq-front \
    --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/features_Anonym/ \
    --batch-size 128 \
    --no-clip --no-farl \
    --no-dino --no-arcface \
    --no-adaface-cvl \
    --cuda --verbose


CUDA_VISIBLE_DEVICES=5 python extract_features.py \
    --dataset celebahq-front \
    --dataset-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/FaceAnonyMixer/datasets/anonymised/celebahq-front \
    --output-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/features_Anonym/ \
    --batch-size 128 \
    --no-arcface-cvl \
    --no-adaface-cvl \
    --cuda --verbose



### ALIGNER

# CUDA_VISIBLE_DEVICES=7 python cvlface_align_faces.py \
#     --data-root /egr/research-sprintai/shared/Datasets-Biometrics/FaceRecognition/CelebA-HQ-Faces/CelebA-HQ-images/ \
#     --save-root /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/CelebA-HQ-AnonyAligned/ \
#     --aligner-id minchul/cvlface_DFA_mobilenet



### FILTER GENERATED-FACES


# python lib/facepose.py  --inp "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/FaceAnonyMixer/datasets/fake/fake_dataset_stylegan2_ffhq1024-0.7-30000-CLIP-FaRL-DINO-ArcFace/*/*.jpg" \
#                         --out /egr/research-sprintai/benja161/BioMetron/agentic_idOBO/FaceAnonyMixer/datasets/fake/Fake_Filtered/

