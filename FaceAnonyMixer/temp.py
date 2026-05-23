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


from turbocode import RSCEncoder, TurboEncoder, TurboDecoder, noiseless_llr, repeat_encode, repeat_decode # in FaceAnonyMixer

def flip_bits(bits, flip_ratio):
    y = bits.copy()
    mask = np.random.rand(len(bits)) < flip_ratio
    y[mask] = 1 - y[mask]
    return y


class BiomBinder_RateControl():
    # poly_a=13, poly_b=15 are the standard LTE/3GPP generator polynomials (octal).
    # num_iterations=6 is a common default for turbo decoding convergence.

    def __init__(self, msg_size=512, R=3, printer=True):

        self.N  = msg_size

        self.R = R
        self.actual_rate = 1.0 / (3 * R)

        fb, ff = 0o23, 0o35

        self.interleaver = np.random.permutation(self.N)
        self.rsc         = RSCEncoder(g_feedback=fb, g_forward=ff)
        self.encoder     = TurboEncoder(self.rsc, self.interleaver)
        self.decoder     = TurboDecoder(self.rsc, self.interleaver, n_iter=8)

        if printer:
            print(f"[BiomBinder lowrate] N={self.N}, R={R}, "
                f"rate=1/{3*R} ≈ {self.actual_rate:.4f}, "
                f"transmitted bits per msg = {3 * self.N * R}")

    def encode_message(self, msg):

        sys, par1, par2 = self.encoder.encode(msg)
        # print("Encoded", sys.shape, par1.shape, par2.shape)

        sys_rep  =  repeat_encode(sys,  self.R)
        par1_rep =  repeat_encode(par1, self.R)
        par2_rep =  repeat_encode(par2, self.R)

        return (sys_rep, par1_rep, par2_rep)

    def decode_message(self, compiled_tuple):
        sys, par1, par2 = compiled_tuple

        sys_f  = noiseless_llr(sys)
        par1_f = noiseless_llr(par1)
        par2_f = noiseless_llr(par2)

        sys_rep  = repeat_decode(sys_f,  self.R)
        par1_rep = repeat_decode(par1_f, self.R)
        par2_rep = repeat_decode(par2_f, self.R)

        bits_hat = self.decoder.decode(
                        sys_rep,
                        par1_rep,
                        par2_rep,
                    )

        return bits_hat


### --- debugger Binding----
msg_size = 512
binder = BiomBinder_RateControl(msg_size=msg_size, R=3)

bits = np.random.randint(0, 2, size=msg_size, dtype=np.int32)
biom = np.random.randint(0, 2, size=msg_size, dtype=np.int32)

message = binder.encode_message(msg=bits)

bits_hat= binder.decode_message(compiled_tuple=message)

error = int(np.abs(bits_hat - bits).sum())

print("ERROR LEVEL CHECK", error)



def get_rate_str(R):
    return f"Repeat Count = {R} \n Transmitted Bits = {3 * 512 * R} \n Code Rate=1/{3*R} ≈ {1.0/(3 * R):.3f}"


ablate_titles= {
1: get_rate_str(1),
3: get_rate_str(3),
6: get_rate_str(6),
9: get_rate_str(9),
12: get_rate_str(12),
}


axis_titles= {
0.1:  "10%",
0.15: "15%",
0.2:  "20%",
0.22: "22%",
0.25: "25%",
0.27: "27%",
0.3:  "30%",
0.4:  "40%",
}

pickle_savepath = "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/HYPES/PLOTS/repeatVerror.pkl"
##------------

loop_count = 100

tc_ablate = {}

for rval in ablate_titles.keys():
    corr_ablate = {}
    for corr in tqdm(axis_titles.keys(), desc=f"{rval}"):
        msg_size = 512
        count = 0
        binder = BiomBinder_RateControl(msg_size=msg_size, R=rval)

        for l in range(loop_count):
            bits = np.random.randint(0, 2, size=msg_size, dtype=np.int32)
            biom = np.random.randint(0, 2, size=msg_size, dtype=np.int32)

            message = binder.encode_message(msg=bits)
            s = flip_bits(message[0], corr)
            p1 = flip_bits(message[1], corr)
            p2 = flip_bits(message[2], corr)

            bits_hat= binder.decode_message(compiled_tuple=(s, p1, p2))

            error = int(np.abs(bits_hat - bits).sum())
            if error == 0:
                count = count + 1
        corr_ablate[corr] = count

    tc_ablate[rval] = corr_ablate

with open(pickle_savepath, "wb") as f:
    pickle.dump(tc_ablate, f, protocol=pickle.HIGHEST_PROTOCOL)

##-----------------

with open(pickle_savepath, "rb") as f:
    tc_ablate = pickle.load(f)

tc_ablate