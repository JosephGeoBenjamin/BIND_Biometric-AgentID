import os, glob, sys
import time
import json
import csv
import itertools
from collections import defaultdict
from tqdm import tqdm
import pickle

import math
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as torch_F
import sklearn.metrics as sk_metrics

from PIL import Image
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import XKCD_COLORS, ListedColormap
import matplotlib.patches as patches

from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder




##------------------------------------------------------------------------------

def cosine_sim(a, b, dim=-1):
    """
    a, b: (..., D)
    returns: (...,) cosine similarity along last dim
    """
    a = torch_F.normalize(a, dim=dim)
    b = torch_F.normalize(b, dim=dim)
    return (a * b).sum(dim=dim)


def hamming_sim(a, b, dim=-1):
    """
    Returns similarity in [0,1]
    1 = identical, 0 = completely different
    """
    return (a == b).float().mean(dim=dim)


def hamming_error(a, b, dim=-1):
    """
    Returns number of differing elements along last dim
    """
    return (a != b).int().sum(dim=dim)



def binary_encode_bits(bin_idx: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    bin_idx: [N, D], int values
    num_bits: B

    Returns:
        [N, D, B] - binary encoding (MSB first)
    """
    bits = []
    for b in range(num_bits):
        bit = (bin_idx >> (num_bits - 1 - b)) & 1
        bits.append(bit)

    return torch.stack(bits, dim=-1)


def gray_encode_bits(bin_idx: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    bin_idx: [N, D], int values
    num_bits: B

    Returns:
        [N, D, B] - Gray code encoding (MSB first)
    """
    # Step 1: convert to Gray code index
    gray_idx = bin_idx ^ (bin_idx >> 1)

    # Step 2: extract bits (same as binary)
    bits = []
    for b in range(num_bits):
        bit = (gray_idx >> (num_bits - 1 - b)) & 1
        bits.append(bit)

    return torch.stack(bits, dim=-1)



##=====================  Utils  ================================================

def tar_at_far(genuine, impostor, target_far):
    """
    TAR at a specific FAR.
    """
    y_true = np.concatenate([
        np.ones(len(genuine)),
        np.zeros(len(impostor))
    ])

    y_score = np.concatenate([genuine, impostor])

    fpr, tpr, thresholds = sk_metrics.roc_curve(
        y_true, y_score, drop_intermediate=False  # don't skip rare FAR points
    )

    # exact FAR=0 case
    if target_far == 0:
        valid = np.where(fpr == 0)[0]
        if len(valid) == 0:
            return 0.0, thresholds[-1]
        idx = valid[np.argmax(tpr[valid])]
        return tpr[idx], thresholds[idx]

    # snap to closest FAR <= target (conservative, research-standard)
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        return 0.0, thresholds[0]

    idx = valid[np.argmax(tpr[valid])]
    return tpr[idx], thresholds[idx]


def plot_score_hist_grid(score_dict, bins=50, density=True, cols=4,
                        title="", save_path='', enable_roc_metrics=False):
    """
    score_dict: {
        "exp1": [genuine_array, impostor_array],
        "exp2": [genuine_array, impostor_array],
        ...
    }
    """

    n = len(score_dict)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))


    if (rows*cols) <2: axes = [axes]
    else: axes = axes.flatten()


    for ax, (name, (genuine, impostor)) in zip(axes, score_dict.items()):
        name,_,suff = name.partition("--")

        sns.histplot(genuine, bins=bins, stat="density", alpha=0.7,  edgecolor=None, ax=ax)
        sns.kdeplot(genuine, ax=ax, linewidth=2, label="Genuine KDE")

        sns.histplot(impostor, bins=bins, stat="density", alpha=0.7, edgecolor=None, ax=ax)
        sns.kdeplot(impostor, ax=ax, linewidth=2, label="Impostor KDE")

        ax.set_title(f"{model_filter[name]} {suff}")
        ax.set_ylabel("Density" if density else "Count")
        ax.legend()
        ax.grid(True)

        xlabel = "score"
        if enable_roc_metrics:
            tar_far_0 , thrs_0   = tar_at_far(genuine, impostor, 0)
            tar_far_1em6, thrs_1em6 = tar_at_far(genuine, impostor, 0.000001)
            tar_far_0001, thrs_0001 = tar_at_far(genuine, impostor, 0.001)
            tar_far_01, thrs_01   = tar_at_far(genuine, impostor, 0.1)

            xlabel += (
                f"\n{'TAR@FAR=0':<15}: {tar_far_0:.3f}  thresh: {thrs_0:.3f}"
                f"\n{'TAR@FAR=1e-6':<15}: {tar_far_1em6:.3f}  thresh: {thrs_1em6:.3f}"
                f"\n{'TAR@FAR=0.001':<15}: {tar_far_0001:.3f}  thresh: {thrs_0001:.3f}"
                f"\n{'TAR@FAR=0.1':<15}: {tar_far_01:.3f}  thresh: {thrs_01:.3f}"
            )

        ax.set_xlabel(xlabel, fontfamily='monospace')

    # remove unused subplots
    for i in range(len(score_dict), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f"Score Histogram {title}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if not save_path: save_path = f"distribution-plot-{time.time()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # save here





def vectors_loader(txt_path, pt_path=None, vectors=None, device="cuda"):

    # load
    with open(txt_path, "r") as f:
        paths = [line.strip() for line in f]

    if vectors is None:
        vectors = torch.load(pt_path).to(device)  # shape [N, D]
    else:
        vectors = torch.tensor(vectors).to(device)

    # build dict
    vec_dic = defaultdict(dict)

    for i, p in enumerate(paths):
        user_id, img_name = p.split("/")  # adjust if deeper paths
        vec_dic[user_id][img_name] = vectors[i].to(device)

    # convert to normal dict
    vec_dic = dict(vec_dic)
    # FILTER: keep only users with >1 images
    vec_dic = {k: v for k, v in vec_dic.items() if len(v) > 1}

    return vectors, vec_dic




##==============================================================================
##                           Binders
##==============================================================================


class BiomBinder_IoMaxGRP():

    def __init__(self, biom_size=300, msg_size=100, shared_cypher=True,
                 device='cuda'):

        if (biom_size//3) < msg_size:
            raise ValueError("Not Compatible msg and biom size", msg_size, biom_size )
        self.device = device
        self.batch_max  = 1024

        self.M = msg_size
        self.B = msg_size * 3
        self.D = 512
        self.actual_rate = 1.0 / 3

        self.shared_cypher = shared_cypher

        self.encoder = TurboEncoder(
            rate=1/3,              # 1/3 or 1/2 only
            constraint_length=4,
            terminate=False
        )
        self.decoder = TurboDecoder(
            encoder=self.encoder,
            num_iter=6,
            hard_out=True
        )

        print(f"[BiomBinder lowrate] Message={self.M}, "
              f"rate=1/{3} ≈ {self.actual_rate:.4f}, "
              f"transmitted bits per msg = {3 * self.M}")

        self.biom_ref = None


    def generate_cypherkey(self, N_=1):
        # (N, B, 2, D)
        weight_m = torch.randn((N_, self.D, 2, self.B ))
        # (N, D)
        bias_v = 1.63*torch.randn((N_, 1, self.D))

        # (N, 2, B)
        pert_i = torch.randint(self.B, (N_, 2, self.B))

        return (weight_m.to(self.device),
                bias_v.to(self.device),
                pert_i.to(self.device))


    def proc_to_bits(self, vecs_in, cypher, return_intermediate=False):

        # (N, B, 2, D), (N, B), (N, 2, B)
        Rweight, Rbias, Rpert = cypher

        N, D = vecs_in.shape
        N_   = Rweight.shape[0] # can be 1 or N
        B    = self.B

        x_bias = vecs_in + Rbias.view(N_, D)

        # project: ---> (N, 2, B)
        proj = (x_bias[:, None, :] @ Rweight.view(N_, D, 2*B)) # (N, 2B) matmul treats last 2-dim as matrix, rest are batch
        proj = proj.view(N, 2, B) / math.sqrt(D)

        # # binarize projection: (N, B)
        # bits = (proj[:, 0, :] > proj[:, 1, :]).to(torch.int)

        ## #perturb indices (N, 2, B), (N_, 2, P) --> (N, 2, P)
        pert = torch.gather(proj, dim=-1, index=Rpert)

        ## #binarize perturbated: (N, B)
        bits = (pert[:, 0, :] > pert[:, 1, :])

        ret_tuple = (bits.to(torch.int), )

        if return_intermediate:
            ret_tuple = ret_tuple + (x_bias.view(N, D), proj.view(N, 2, B))

        return ret_tuple



    def compile_message(self, biom, msg):

        encoded_msg = self.encoder(msg)

        hashed_msg = torch.logical_xor(encoded_msg, biom)

        return hashed_msg


    def extract_message(self, biom2, hashed_msg):

        encoded_hat = torch.logical_xor(hashed_msg, biom2)

        # Sionna convention: negative LLR = likely 0, positive LLR = likely 1
        # bit=0 → signal +1 → LLR should be negative → -(+1) * scale
        # bit=1 → signal -1 → LLR should be positive → -(-1) * scale
        llr = -(1.0 - 2.0 * encoded_hat) * 10.0

        msg_hat     = self.decoder(llr)

        return msg_hat.to(torch.int)


    def genuine_extraction(self, biom_dict, msg_bits):
        msg_scores = []
        biom_scores = []
        rbiom_scores = []
        brbiom_scores = []; prbiom_scores = [];

        for user_id, imgs in tqdm(biom_dict.items(), position=0):
            vecs = list(imgs.values())
            if len(vecs) < 2: continue


            vecs = torch.vstack(vecs).to(self.device)

            R_CYPER = self.generate_cypherkey(N_=1) # genuine cypher is same always

            anchor_vec = vecs[0:1]
            anchor_bit, anc_bias, anc_proj = self.proc_to_bits(anchor_vec, R_CYPER,
                                                            return_intermediate=True)

            others_vec = vecs[1:]
            others_bit, oth_bias, oth_proj = self.proc_to_bits(others_vec, R_CYPER,
                                                            return_intermediate=True)


            # start_time = time.time()
            hashed_data = self.compile_message(biom=others_bit, msg=msg_bits)
            msg_bits_hat = self.extract_message(biom2=anchor_bit, hashed_msg=hashed_data)
            # print("Hash unhash Time", time.time()- start_time)

            scr_m = hamming_error(msg_bits, msg_bits_hat)
            msg_scores.extend(scr_m.cpu().tolist())

            scr_b = hamming_sim(anchor_bit, others_bit)
            biom_scores.extend(scr_b.cpu().tolist())

            scr_r = cosine_sim(anchor_vec, others_vec)
            rbiom_scores.extend(scr_r.cpu().tolist())

            scr_br = cosine_sim(anc_bias, oth_bias)
            brbiom_scores.extend(scr_br.cpu().tolist())

            scr_pr = cosine_sim(anc_proj, oth_proj).view(-1)
            prbiom_scores.extend(scr_pr.cpu().tolist())

        return msg_scores, biom_scores, rbiom_scores, brbiom_scores, prbiom_scores


    def imposter_extraction(self, biom_dict, msg_bits, sim_func=np.dot):
        msg_scores = []
        biom_scores = []
        rbiom_scores = []
        brbiom_scores = []; prbiom_scores = [];

        max_v = 10

        # first max_v vector per user
        filtered_dict = {u: torch.vstack(
                            list(biom_dict[u].values())[:max_v]
                                ).to(self.device)
                            for u in biom_dict.keys()
                                if len(biom_dict[u]) > 0}

        user_ids = list(filtered_dict.keys())


        for ref_user in tqdm(user_ids):

            R_CYPER = self.generate_cypherkey(N_=1)

            anchor_vec = filtered_dict[ref_user][0:1]
            anchor_bit, anc_bias, anc_proj = self.proc_to_bits(anchor_vec, R_CYPER,
                                                    return_intermediate=True)


            R_CYPER_O = R_CYPER

            others_vec = torch.vstack([v for u, v in filtered_dict.items()
                                        if u != ref_user])
            others_bit, oth_bias, oth_proj = self.proc_to_bits(others_vec, R_CYPER_O,
                                                            return_intermediate=True)

            hashed_data = self.compile_message(biom=others_bit, msg=msg_bits)
            msg_bits_hat = self.extract_message(biom2=anchor_bit, hashed_msg=hashed_data)

            scr_m = hamming_error(msg_bits, msg_bits_hat)
            msg_scores.extend(scr_m.cpu().tolist())

            scr_b = hamming_sim(anchor_bit, others_bit)
            biom_scores.extend(scr_b.cpu().tolist())

            scr_r = cosine_sim(anchor_vec, others_vec)
            rbiom_scores.extend(scr_r.cpu().tolist())

            scr_br = cosine_sim(anc_bias, oth_bias)
            brbiom_scores.extend(scr_br.cpu().tolist())

            scr_pr = cosine_sim(anc_proj, oth_proj).view(-1)
            prbiom_scores.extend(scr_pr.cpu().tolist())

        return msg_scores, biom_scores, rbiom_scores, brbiom_scores, prbiom_scores





##==============================================================================
##                           MAIN
##==============================================================================

def read_modelwise_biometrics(root_path, device_mapping='cuda'):
    feature_files = glob.glob(f"{root_path}/*.pt", recursive=True)
    modelrep_dict = {}

    for ff in feature_files:
        ff_name = ff.split("/")[-1].strip(".pt")

        if ff_name not in model_filter.keys(): continue

        vec , vec_dict = vectors_loader(os.path.join(root_path, "image_filenames.txt"),
                            ff, device=device_mapping)
        modelrep_dict[ff_name] = [vec, vec_dict]

        print(ff_name)

    return modelrep_dict


def binding_debugger():

    biom_size = 512
    msg_size = biom_size // 3
    binder = BiomBinder_IoMaxGRP(biom_size=biom_size, msg_size=msg_size)

    bits = np.random.randint(0, 2, size=msg_size, dtype=np.int32)
    biom = np.random.randint(0, 2, size=biom_size, dtype=np.int32)

    encoded_message = binder.compile_message(biom=biom, msg=bits)

    bits_hat= binder.extract_message(biom2=biom, compiled_tuple=encoded_message)

    error = int(np.abs(bits_hat - bits).sum())

    print(error)



if __name__ == "__main__":
    DEVICE = 'cuda'

    ROOT_PATH = "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/face_features/cfp-frontal/"

    SAVE_PATH = "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/HYPES/trials/e2/"
    os.makedirs(SAVE_PATH, exist_ok=True)


    model_filter = {
    # "clip_features"    : "Clip Zero-Shot ViT-B",
    # "farl_features"    : "FARL Zero-Shot ViT-B",
    # "cvl_arcface_features" : "ArcFace IResNet-101",
    "cvl_adaface_features" : "AdaFace IResNet-101",
    # "dino_features"        : "Dino Zero-shot",
    # "arcface_features"     : "IR50 Arc Face",
    }

    modelrep_dict = read_modelwise_biometrics(ROOT_PATH, DEVICE)


    repeat_rate = 1
    biom_enc_bits = 1
    msg_size = 2048
    biom_size = msg_size*3

    binder = BiomBinder_IoMaxGRP(biom_size=biom_size, msg_size=msg_size, device=DEVICE)

    msg_bits = torch.randint(0, 2, (msg_size,), dtype=torch.int).to(DEVICE)


    msg_matching_dict = {}
    rbiom_matching_dict = {}

    for fk in model_filter.keys():
        print(fk)
        vec_dict = modelrep_dict[fk][1]

        VEC_IN = modelrep_dict[fk][0]

        vec , vec_dict = vectors_loader(os.path.join(ROOT_PATH, "image_filenames.txt"),
                            vectors=VEC_IN)

        # ### REMOVE : debug
        # sample_keys = random.sample(list(vec_dict.keys()), min(10, len(vec_dict)))
        # sub_dict = {k: vec_dict[k] for k in sample_keys}
        # vec_dict = sub_dict
        # #### remove

        ## sequential computation
        (gen_msg_err, gen_biom_sim, gen_rbiom_sim, gen_brbiom_sim, gen_prbiom_sim
                            ) = binder.genuine_extraction(vec_dict, msg_bits)
        (imp_msg_err, imp_biom_sim, imp_rbiom_sim, imp_brbiom_sim, imp_prbiom_sim
                            ) = binder.imposter_extraction(vec_dict, msg_bits)

        ## store stuff
        msg_matching_dict[fk] = [gen_msg_err, imp_msg_err]

        rbiom_matching_dict[f"{fk}--raw"] = [gen_rbiom_sim, imp_rbiom_sim]
        rbiom_matching_dict[f"{fk}--bias"] = [gen_brbiom_sim, imp_brbiom_sim]
        rbiom_matching_dict[f"{fk}--proj"] = [gen_prbiom_sim, imp_prbiom_sim]
        rbiom_matching_dict[f"{fk}--binary"] = [gen_biom_sim, imp_biom_sim]


        dump_dict = {fk: msg_matching_dict[fk]}
        with open(os.path.join(SAVE_PATH, f"{fk}-message_match_error_sum.pkl"), "wb") as f:
            pickle.dump(dump_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        dump_dict = {k:v for k,v in rbiom_matching_dict.items() if fk in k}
        with open(os.path.join(SAVE_PATH, f"{fk}-biom_analysis.pkl"), "wb") as f:
            pickle.dump(dump_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # with open("data.pkl", "rb") as f: loaded_from_disk = pickle.load(f)


        plot_score_hist_grid(msg_matching_dict, title = f"Retrived Message Error", cols=1,
                save_path=os.path.join(SAVE_PATH,"binded-msg-recovery-sum.png"))

        plot_score_hist_grid(rbiom_matching_dict, title = f"Biometrics Similarity for Step-Wise", cols=4,
            save_path=os.path.join(SAVE_PATH,"biometric-similarities.png"))