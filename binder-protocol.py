import os, glob, sys
import json
import csv
import itertools
from collections import defaultdict
from tqdm import tqdm
import pickle

import pandas as pd
import numpy as np
import torch
import math
import random
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import XKCD_COLORS, ListedColormap
import matplotlib
import matplotlib.patches as patches
from PIL import Image
import sklearn.metrics as sk_metrics
from natsort import natsorted


from turbocode import RSCEncoder, TurboEncoder, TurboDecoder, noiseless_llr, repeat_encode, repeat_decode

##------------------------------------------------------------------------------

def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def hamming_sim(a, b):
    """
    Returns value in [0,1]
    0 = identical, 1 = completely different
    np.sum(a != b)  -> not normalised
    """
    return 1 - np.mean(a != b)


def hamming_error(a, b):
    """
    np.sum(a != b)  -> not normalised
    """
    return np.sum(a != b)


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



def percentile_binary_encode(vec: torch.Tensor, num_bits: int, coding="binary") -> torch.Tensor:
    """
    a: Tensor [N, D]
    num_bits: number of bits (B)

    Returns:
        Tensor [N, D * B] with binary encoding (0/1 ints)
    """
    N, D = vec.shape
    num_bins = 2 ** num_bits

    # Step 1: compute thresholds (percentiles)
    percentiles = torch.linspace(0, 100, steps=num_bins + 1, device=vec.device)[1:-1]
    thresholds = torch.quantile(vec, percentiles / 100.0, dim=0)  # [num_bins-1, D]

    # Step 2: assign bin indices
    # bin_idx in [0, num_bins-1]
    bin_idx = torch.zeros_like(vec, dtype=torch.long)

    for i, t in enumerate(thresholds):
        bin_idx += (vec > t).long()

    # Step 3: convert bin index to binary (B bits) [N, D, B]
    if coding == "gray":
        encoded =  gray_encode_bits(bin_idx, num_bits)
    elif coding == "binary":
        encoded =  binary_encode_bits(bin_idx, num_bits)
    elif coding == "bins":
        encoded =  bin_idx

    # Step 4: stack → [N, D, B] → reshape → [N, D*B]
    encoded = encoded.reshape(N, D * num_bits)

    return encoded



##=====================  Utils  ================================================

def plot_score_hist_grid(score_dict, bins=50, density=True, cols=4, title="", save_path=''):
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

        sns.histplot(genuine, bins=bins, stat="density", alpha=0.7,  edgecolor=None, ax=ax)
        sns.kdeplot(genuine, ax=ax, linewidth=2, label="Genuine KDE")

        sns.histplot(impostor, bins=bins, stat="density", alpha=0.7, edgecolor=None, ax=ax)
        sns.kdeplot(impostor, ax=ax, linewidth=2, label="Impostor KDE")

        ax.set_title(model_filter[name])
        ax.set_xlabel("Score")
        ax.set_ylabel("Density" if density else "Count")
        ax.legend()
        ax.grid(True)

    # remove unused subplots
    for i in range(len(score_dict), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f"Score Histogram {title}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if not save_path: save_path = "binded-msg-recovery.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # save here





def vectors_loader(txt_path, pt_path=None, vectors=None):

    # load
    with open(txt_path, "r") as f:
        paths = [line.strip() for line in f]

    if vectors is None:
        vectors = torch.load(pt_path)  # shape [N, D]
    else:
        vectors = vectors

    # build dict
    vec_dic = defaultdict(dict)

    for i, p in enumerate(paths):
        user_id, img_name = p.split("/")  # adjust if deeper paths
        vec_dic[user_id][img_name] = vectors[i].cpu().numpy()

    # convert to normal dict
    vec_dic = dict(vec_dic)
    # FILTER: keep only users with >1 images
    vec_dic = {k: v for k, v in vec_dic.items() if len(v) > 1}

    return vectors, vec_dic




##==============================================================================
##                           Binders
##==============================================================================


class BiomBinder_Naive():
    # poly_a=13, poly_b=15 are the standard LTE/3GPP generator polynomials (octal).
    # num_iterations=6 is a common default for turbo decoding convergence.

    def __init__(self, biom_size=522, msg_size=174):
        self.N  = biom_size
        self.M  = msg_size

        fb, ff = 0o23, 0o35

        self.interleaver = np.random.permutation(self.M)
        self.rsc         = RSCEncoder(g_feedback=fb, g_forward=ff)
        self.encoder     = TurboEncoder(self.rsc, self.interleaver)
        self.decoder     = TurboDecoder(self.rsc, self.interleaver, n_iter=8)

    def compile_message(self, biom, msg):
        sys, par1, par2 = self.encoder.encode(msg)
        # print("Encoded", sys.shape, par1.shape, par2.shape)

        sys_x  = np.bitwise_xor(sys,  biom[:self.M])
        par1_x = np.bitwise_xor(par1, biom[self.M:(self.M*2)])
        par2_x = np.bitwise_xor(par2, biom[(self.M*2):(self.M*3)])

        return (sys_x, par1_x, par2_x)

    def extract_message(self, biom2, compiled_tuple):
        sys, par1, par2 = compiled_tuple
        sys_f  = np.bitwise_xor(sys,  biom2[:self.M])
        par1_f = np.bitwise_xor(par1, biom2[self.M:(self.M*2)])
        par2_f = np.bitwise_xor(par2, biom2[(self.M*2):(self.M*3)])

        bits_hat = self.decoder.decode(
                        noiseless_llr(sys_f,),
                        noiseless_llr(par1_f,),
                        noiseless_llr(par2_f,)
                    )

        return bits_hat


    def genuine_extraction(self, biom_dict, msg_bits, sim_func=np.dot):
        scores = []

        for user_id, imgs in tqdm(biom_dict.items()):
            vecs = list(imgs.values())

            if len(vecs) < 2:
                continue

            anchor = vecs[0]
            enc_tuple = self.compile_message(biom=anchor, msg=msg_bits)
            for v in vecs[1:]:
                msg_bits_hat = self.extract_message(biom2=v, compiled_tuple=enc_tuple)
                scr = sim_func(msg_bits, msg_bits_hat)
                scores.append(scr)


        return np.array(scores)


    def imposter_extraction(self, biom_dict, msg_bits, sim_func=np.dot):
        scores = []
        user_ids = list(biom_dict.keys())

        # first vector per user
        anchors = {u: list(biom_dict[u].values())[0] for u in user_ids if len(biom_dict[u]) > 0}

        users = list(anchors.keys())
        for i in tqdm(range(len(users))):
            enc_tuple = self.compile_message(biom=anchors[users[i]], msg=msg_bits)
            j = random.randrange(len(users)-1)
            if j==i: j=j+1 # to make it not equal to i
            msg_bits_hat = self.extract_message(biom2=anchors[users[j]], compiled_tuple=enc_tuple)

            scr = sim_func(msg_bits, msg_bits_hat)
            scores.append(scr)

        return np.array(scores)



class BiomBinder_RateControl():
    # poly_a=13, poly_b=15 are the standard LTE/3GPP generator polynomials (octal).
    # num_iterations=6 is a common default for turbo decoding convergence.

    def __init__(self, biom_size=512, msg_size=None, R=3):

        if not msg_size: msg_size = biom_size // (3*R)

        if (biom_size//(3*R)) != msg_size:
            raise ValueError("Not Compatible msg and biom size")

        self.N  = biom_size
        self.M  = msg_size
        self.R = R
        self.actual_rate = 1.0 / (3 * R)

        fb, ff = 0o23, 0o35

        self.interleaver = np.random.permutation(self.M)
        self.rsc         = RSCEncoder(g_feedback=fb, g_forward=ff)
        self.encoder     = TurboEncoder(self.rsc, self.interleaver)
        self.decoder     = TurboDecoder(self.rsc, self.interleaver, n_iter=8)

        print(f"[BiomBinder lowrate] N={self.N}, R={R}, "
              f"rate=1/{3*R} ≈ {self.actual_rate:.4f}, "
              f"transmitted bits per msg = {3 * self.M * R}, ",
               f"Message bits = {self.M}" )


    def compile_message(self, biom, msg):

        sys, par1, par2 = self.encoder.encode(msg)
        # print("Encoded", sys.shape, par1.shape, par2.shape)

        sys_rep  =  repeat_encode(sys,  self.R)
        par1_rep =  repeat_encode(par1, self.R)
        par2_rep =  repeat_encode(par2, self.R)

        biom_tiled = biom

        j = self.M * self.R
        sys_x  = np.bitwise_xor(sys_rep,  biom_tiled[:j])
        par1_x = np.bitwise_xor(par1_rep, biom_tiled[j:2*j])
        par2_x = np.bitwise_xor(par2_rep, biom_tiled[2*j:3*j])

        return (sys_x, par1_x, par2_x)

    def extract_message(self, biom2, compiled_tuple):
        sys, par1, par2 = compiled_tuple

        biom2_tiled = biom2

        j = self.M * self.R
        sys_f  = np.bitwise_xor(sys,  biom2_tiled[:j])
        par1_f = np.bitwise_xor(par1, biom2_tiled[j:2*j])
        par2_f = np.bitwise_xor(par2, biom2_tiled[2*j:3*j])

        sys_f  = noiseless_llr(sys_f)
        par1_f = noiseless_llr(par1_f)
        par2_f = noiseless_llr(par2_f)

        sys_rep  = repeat_decode(sys_f,  self.R)
        par1_rep = repeat_decode(par1_f, self.R)
        par2_rep = repeat_decode(par2_f, self.R)

        bits_hat = self.decoder.decode(
                        sys_rep,
                        par1_rep,
                        par2_rep,
                    )

        return bits_hat


    def genuine_extraction(self, biom_dict, msg_bits, sim_func=np.dot):
        scores = []

        for user_id, imgs in tqdm(biom_dict.items()):
            vecs = list(imgs.values())

            if len(vecs) < 2:
                continue

            anchor = vecs[0]
            enc_tuple = self.compile_message(biom=anchor, msg=msg_bits)
            for v in vecs[1:]:
                msg_bits_hat = self.extract_message(biom2=v, compiled_tuple=enc_tuple)
                scr = sim_func(msg_bits, msg_bits_hat)
                scores.append(scr)


        return np.array(scores)


    def imposter_extraction(self, biom_dict, msg_bits, sim_func=np.dot):
        scores = []
        user_ids = list(biom_dict.keys())

        # first vector per user
        anchors = {u: list(biom_dict[u].values())[0] for u in user_ids if len(biom_dict[u]) > 0}

        users = list(anchors.keys())
        for i in tqdm(range(len(users))):
            enc_tuple = self.compile_message(biom=anchors[users[i]], msg=msg_bits)
            for _ in range(2):
                j = random.randrange(len(users))
                if j==i: j=j+1 # to make it not equal to i
                msg_bits_hat = self.extract_message(biom2=anchors[users[j]], compiled_tuple=enc_tuple)

                scr = sim_func(msg_bits, msg_bits_hat)
                scores.append(scr)

        return np.array(scores)



##==============================================================================
##                           MAIN
##==============================================================================

def read_modelwise_biometrics(root_path):
    feature_files = glob.glob(f"{root_path}/*.pt", recursive=True)
    modelrep_dict = {}

    for ff in feature_files:
        ff_name = ff.split("/")[-1].strip(".pt")

        if ff_name not in model_filter.keys(): continue

        vec , vec_dict = vectors_loader(os.path.join(root_path, "image_filenames.txt"),
                            ff)
        modelrep_dict[ff_name] = [vec, vec_dict]

        print(ff_name)

    return modelrep_dict


def binding_debugger():

    biom_size = 512
    msg_size = biom_size // 3
    binder = BiomBinder_RateControl(biom_size=biom_size, msg_size=msg_size)

    bits = np.random.randint(0, 2, size=msg_size, dtype=np.int32)
    biom = np.random.randint(0, 2, size=biom_size, dtype=np.int32)

    encoded_message = binder.compile_message(biom=biom, msg=bits)

    bits_hat= binder.extract_message(biom2=biom, compiled_tuple=encoded_message)

    error = int(np.abs(bits_hat - bits).sum())

    print(error)



if __name__ == "__main__":

    ROOT_PATH = "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/datasets/data_extracts/features_Anonym/celebahq-front/"

    SAVE_PATH = "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/HYPES/ijcb_runs/base-R1-b512/"
    os.makedirs(SAVE_PATH, exist_ok=True)

    model_filter = {
    # "clip_features"    : "Clip Zero-Shot ViT-B",
    # "farl_features"    : "FARL Zero-Shot ViT-B",
    # "cvl_arcface_features" : "ArcFace IR101",
    # "cvl_adaface_features" : "AdaFace IR101",
    # "dino_features"        : "Dino Zero-shot",
    "arcface_features"     : "IR50 Arc Face",
    }
    modelrep_dict = read_modelwise_biometrics(ROOT_PATH)


    repeat_rate = 1
    biom_enc_bits = 1
    biom_size = 512
    msg_size = biom_size // (repeat_rate*3)

    binder = BiomBinder_RateControl(biom_size=biom_size, R=repeat_rate)
    msg_bits = np.random.randint(0, 2, size=msg_size, dtype=np.int32)


    msg_matching_dict = {}
    for fk in model_filter.keys():
        print(fk)
        VEC_IN = percentile_binary_encode(modelrep_dict[fk][0], num_bits=biom_enc_bits, coding='binary')
        vec , vec_dict = vectors_loader(os.path.join(ROOT_PATH, "image_filenames.txt"),
                            vectors=VEC_IN)

        # #REMOVE : debug
        # sample_keys = random.sample(list(vec_dict.keys()), min(6, len(vec_dict)))
        # sub_dict = {k: vec_dict[k] for k in sample_keys}
        # vec_dict = sub_dict
        # #remove

        msg_matching_dict[fk] = [binder.genuine_extraction(vec_dict, msg_bits, sim_func=hamming_error),
                            binder.imposter_extraction(vec_dict, msg_bits, sim_func=hamming_error)]



    with open(os.path.join(SAVE_PATH, "message_match_hist_sum.pkl"), "wb") as f:
        pickle.dump(msg_matching_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("data.pkl", "rb") as f: loaded_from_disk = pickle.load(f)


    plot_score_hist_grid(msg_matching_dict, title = f"Retrived Message Error", cols=1,
            save_path=os.path.join(SAVE_PATH,"binded-msg-recovery-sum.png"))

