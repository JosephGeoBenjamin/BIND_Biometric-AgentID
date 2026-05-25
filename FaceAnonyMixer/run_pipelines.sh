export PYTHONPATH="/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/Bind-AgentID/FaceAnonyMixer/":$PYTHONPATH


CUDA_VISIBLE_DEVICES=7 python invert.py \
    --dataset celebahq-front \
    --batch-size 1  \
    --save-reconstructed-images \
    --save-aligned-images \
    --cuda --verbose


CUDA_VISIBLE_DEVICES=7 python extract_features.py \
    --dataset celebahq-front \
    --batch-size 128 \
    --no-adaface-cvl  \
    --no-arcface-cvl \
    --cuda --verbose


CUDA_VISIBLE_DEVICES=7 python create_fake_dataset.py \
    --gan stylegan2_ffhq1024 \
    --num-samples 1000 \
    --truncation 0.7 \
    --cuda --verbose


## Filter Generated face with Good Pose

python lib/facepose.py  --inp "datasets/fake/fake_dataset_stylegan2_ffhq1024/*/*.jpg" \
                        --out datasets/fake/Fake_Filtered/



python pair_unique.py \
    --real-dataset celebahq-front \
    --fake-dataset-root  datasets/fake/Fake_Filtered \
    --verbose


CUDA_VISIBLE_DEVICES=7 python anonymize.py \
    --dataset celebahq-front \
    --fake-nn-map datasets/fake/Fake_Filtered/random_nn_map_celebahq-front.json \
    --latent-space W+ \
    --epochs 50 \
    --lr 0.01 \
    --lambda-id 10.0 \
    --lambda-attr 0.15 \
    --lambda-consistency 10.0 \
    --cuda --gpu-id 0 --verbose
