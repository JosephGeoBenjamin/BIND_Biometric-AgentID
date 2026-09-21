"""Extract feature embeddings for every image in a real dataset.

Supported feature spaces: CLIP, FaRL, DINO, ArcFace.
Results are saved under ``datasets/features/<dataset>/``.

Usage
-----
    python extract_features.py \\
        --dataset sample_IJB-C \\
        --batch-size 128 \\
        --cuda --verbose
"""

import argparse
import os
import os.path as osp

import clip
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# from lib import DATASETS, FARL_PRETRAIN_MODEL, VGGFace2

import cvlface_loader as cvlface
import lafs_loader as lafs_fr


DATASETS = {
    "cfp-frontal" : "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/imagesets/CFP-frontal/",
    "lfw-a" : "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/imagesets/LFW-a-set/",
    "multipie" : "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/imagesets/Multi_PIE/",
    "celeba-hq": "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/imagesets/CelebA-HQ-set/",
    "celeba-hq-frontal": "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/imagesets/CelebA-HQ-frontal/",
    "casia-face": "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/imagesets/CASIA-webface/"
}

class VGGFaceDataset(data.Dataset):
    """
    Expects the dataset root to contain a ``images/`` sub-folder organised as
    one directory per identity::

        <root_dir>/aligned/<identity>/<img>.jpg

    Missing optional items are returned as zero tensors.

    Args:
        root_dir (str): dataset root directory.
        transform: torchvision transform applied to every image.
    """

    def __init__(self,
                 root_dir: str,
                 transform=None):
        self.root_dir  = root_dir
        # self.train_dir = osp.join(root_dir, 'train')
        self.train_dir = osp.join(root_dir, 'aligned')

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self._prepare_identity_based_data()

    def _prepare_identity_based_data(self):
        identity_folders = sorted(
            f for f in os.listdir(self.train_dir)
            if osp.isdir(osp.join(self.train_dir, f))
        )
        self.images, self.labels = [], []
        for label, folder in enumerate(identity_folders):
            folder_path = osp.join(self.train_dir, folder)
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(osp.join(folder_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path      = self.images[idx]
        img_basename  = osp.basename(img_path)
        identity_folder = osp.basename(osp.dirname(img_path))
        stem          = img_basename.split('.')[0]

        img_orig = self.transform(Image.open(img_path).convert('RGB'))

        return [img_orig, img_path]



# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Extract CLIP/FaRL/DINO/ArcFace features for a real dataset.')

    p.add_argument('-v', '--verbose',  action='store_true')
    p.add_argument('--dataset',        type=str, required=True,
                    choices=list(DATASETS.keys()))
    p.add_argument('--dataset-root',   type=str, default=None)
    p.add_argument('--output-root',   type=str, default=None)
    p.add_argument('--batch-size',     type=int, default=128)

    p.add_argument('--use-clip',            action='store_true')
    p.add_argument('--use-dino',            action='store_true')
    p.add_argument('--use-lafs',            action='store_true')

    p.add_argument('--use-arcface-cvl',     action='store_true')
    p.add_argument('--use-adaface-cvl',     action='store_true')
    p.add_argument('--use-vitkprpe-cvl',     action='store_true')

    p.add_argument('--cuda',           dest='cuda', action='store_true')

    p.set_defaults(cuda=True)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    if args.output_root:
        out_dir = osp.join(args.output_root, args.dataset)
    else:
        out_dir = osp.join('datasets', 'features', args.dataset)

    os.makedirs(out_dir, exist_ok=True)

    ### -- Skip already-computed features
    # if osp.exists(osp.join(out_dir, 'clip_features.pt')):    args.use_clip    = True
    # if osp.exists(osp.join(out_dir, 'farl_features.pt')):    args.no_farl    = True
    # if osp.exists(osp.join(out_dir, 'dino_features.pt')):    args.use_dino    = True
    # if osp.exists(osp.join(out_dir, 'arcface_features.pt')): args.no_arcface = True

    # if all([args.use_clip, args.no_farl, args.use_dino, args.no_arcface]):
    #     print(f'All features already computed under {out_dir}. Nothing to do.')
    #     return


    # ── build models ──────────────────────────────────────────────────────────
    clip_model   = dino_model = lafs_model = None
    cvl_arcface_model  =  cvl_adaface_model  =  cvl_vitkprpe_model = None


    if args.use_clip:
        clip_model, _ = clip.load('ViT-B/16', device=device, jit=False)
        clip_model.float().eval()
        clip_tf = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                  (0.26862954, 0.26130258, 0.27577711)),
        ])

    if args.use_dino:
        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        dino_model.float().eval().to(device)
        dino_tf = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if args.use_lafs:
        lafs_repo  = "pretrained/lafs_webface_finetune_withaugmentation.pth"
        lafs_model = lafs_fr.load_lafs_model(lafs_repo).to(device)
        lafs_model.float().eval().to(device)
        lafs_tf    = lafs_fr.get_lafs_transform(input_size=112, tensorize=False)


    ## CVLface
    if args.use_arcface_cvl:
        cvl_arcface_repo  = "minchul/cvlface_arcface_ir101_webface4m"
        cvl_arcface_model = cvlface.load_cvlface_model(cvl_arcface_repo)
        cvl_arcface_model.float().eval().to(device)
        cvl_arcface_tf    = cvlface.get_cvlface_transform(input_size=112, tensorize=False)

    if args.use_adaface_cvl:
        cvl_adaface_repo  = "minchul/cvlface_adaface_ir101_webface4m"
        cvl_adaface_model = cvlface.load_cvlface_model(cvl_adaface_repo).to(device)
        cvl_adaface_model.float().eval().to(device)
        cvl_adaface_tf    = cvlface.get_cvlface_transform(input_size=112, tensorize=False)

    if args.use_vitkprpe_cvl:
        cvl_vitkprpe_repo  = "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
        cvl_aligner_repo   = "minchul/cvlface_DFA_mobilenet"
        cvl_vitkprpe_model = cvlface.KPRPEExtractor(cvl_vitkprpe_repo,
                                                    cvl_aligner_repo,
                                                    device=device)
        cvl_vitkprpe_model.float().eval().to(device)
        cvl_vitkprpe_tf    = cvlface.get_cvlface_transform(input_size=112, tensorize=False)



    # ── dataset ───────────────────────────────────────────────────────────────
    dataset_root = args.dataset_root or DATASETS[args.dataset]
    dataset      = VGGFaceDataset(root_dir=dataset_root, transform=transforms.ToTensor())
    loader       = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ── extraction loop ───────────────────────────────────────────────────────
    filenames     = []
    clip_feats    = []
    dino_feats    = []
    lafs_feats    = []

    cvl_arcface_feats  = []
    cvl_adaface_feats  = []
    cvl_vitkprpe_feats = []


    desc = f'Extracting features [{args.dataset}]' if args.verbose else ''
    for batch in tqdm(loader, desc=desc):
        imgs       = batch[0].to(device)
        # batch_fns  = [osp.basename(p) for p in batch[1]]
        batch_fns  = ["/".join(p.split("/")[-2:]) for p in batch[1]]
        filenames.extend(batch_fns)

        with torch.no_grad():
            if clip_model:
                clip_feats.append(clip_model.encode_image(clip_tf(imgs)).cpu())
            if dino_model:
                dino_feats.append(dino_model(dino_tf(imgs)).cpu())
            if lafs_model:
                lafs_feats.append(lafs_model(lafs_tf(imgs)).cpu())

            if cvl_arcface_model:
                cvl_arcface_feats.append(cvl_arcface_model(cvl_arcface_tf(imgs)).cpu())
            if cvl_adaface_model:
                cvl_adaface_feats.append(cvl_adaface_model(cvl_adaface_tf(imgs)).cpu())
            if cvl_vitkprpe_model:
                cvl_vitkprpe_feats.append(cvl_vitkprpe_model(cvl_vitkprpe_tf(imgs)).cpu())


    # ── save ──────────────────────────────────────────────────────────────────
    with open(osp.join(out_dir, 'image_filenames.txt'), 'w') as f:
        f.writelines(fn + '\n' for fn in filenames)

    def _save(feats, name):
        if feats:
            mat = torch.cat(feats)
            torch.save(mat, osp.join(out_dir, name))
            if args.verbose:
                print(f'  \\__ {name}: {mat.shape}')


    _save(clip_feats,    'clip_features.pt')
    _save(dino_feats,    'dino_features.pt')
    _save(lafs_feats,    'lafs_features.pt')

    _save(cvl_arcface_feats,  'cvl_arcface_features.pt')
    _save(cvl_adaface_feats,  'cvl_adaface_features.pt')
    _save(cvl_vitkprpe_feats, 'cvl_vitkprpe_features.pt')


    if args.verbose:
        print(f'#. Features saved to {out_dir}')


if __name__ == '__main__':
    main()
