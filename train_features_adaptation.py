"""
Boiler plate from Claude
"""
import os, sys
import math
import numpy as np
import torch
import torch.nn.functional as torch_F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt



##==============================================================================

class Measurer():
    def __init__(self, test_steps):
        self.test_steps = test_steps
        self.cosim = []
        self.mserr = []

        self.inp_var = []
        self.out_var = []

    def measure(self, yhat, y):
        self._update_cosim(yhat, y)
        self._update_mserr(yhat, y)
        self._update_variance(yhat, y)

    def _update_cosim(self, yhat, y):
        s = torch.nn.functional.cosine_similarity(y, yhat, dim=1)
        s = s.mean().item()
        self.cosim.append(s)

    def _update_mserr(self, yhat, y):
        s = ((y - yhat) ** 2).mean(dim=1)
        s = s.mean().item()
        self.mserr.append(s)

    def _update_variance(self, yhat, y):
        v = torch.var(yhat, dim=0).mean().item()
        self.out_var.append(v)
        v = torch.var(y, dim=0).mean().item()
        self.inp_var.append(v)


def plot_measurer(measurer, title="", savepath=''):

    r, c = 1, 3
    fig, axes = plt.subplots(r, c, figsize=(c*4, r*4))


    interval = measurer.test_steps

    mserr = measurer.mserr
    x1 = np.arange(len(mserr))

    axes[0].plot(x1, mserr, color='orangered')
    axes[0].set_title('Mean Squared Error')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Error")
    for v in range(0, len(mserr), interval):
        axes[0].axvline(v, linestyle=':', color='gray', linewidth=0.7, alpha=0.6)

    cosim =  measurer.cosim
    x2 = np.arange(len(cosim))

    axes[1].plot(x2, cosim,color='teal')
    axes[1].set_title("Cosine Similarity")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Similarity")
    for v in range(0, len(cosim), interval):
        axes[1].axvline(v, linestyle=':', color='gray', linewidth=0.7, alpha=0.6)

    var1 = measurer.inp_var
    var2 = measurer.out_var
    axes[2].plot(x2, var1, color='goldenrod')
    axes[2].plot(x2, var2, color='yellow')
    axes[2].set_title("variance")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Similarity")
    for v in range(0, len(cosim), interval):
        axes[2].axvline(v, linestyle=':', color='gray', linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(savepath,"test_measurements.png"), dpi=300, bbox_inches="tight")



def plot_training(losstracker, title="", savepath=''):

    plt.figure(figsize=(10, 5))

    for k, v in losstracker.items():
        x = np.arange(len(v))
        plt.plot(x, v, label=str(k))

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "train_curves.png"), dpi=300, bbox_inches="tight")


# ======================== Dataset =============================================

class RandomVectorDataset(Dataset):
    """
    Returns random unit-normed vectors of dim 512.
    Values are both positive and negative (sampled from standard normal,
    then L2-normalized so ||x|| = 1).
    """

    def __init__(self, sample_count: int = 10000, dim: int = 512, noise_eps=0.0001):
        self.size = sample_count
        self.dim = dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        raw = torch.randn(self.dim)
        true = raw / raw.norm()

        noise = torch.randn(self.dim)
        raw_noise = raw+noise
        corr = raw_noise / raw_noise.norm()

        return corr, true



class RandomVectorTestset(Dataset):
    """
    Returns random unit-normed vectors of dim 512.
    Values are both positive and negative (sampled from standard normal,
    then L2-normalized so ||x|| = 1).
    """

    def __init__(self, sample_count: int = 5000, dim: int = 512, noise_eps=0.0001):
        self.size = sample_count
        self.dim = dim
        raw = torch.randn(self.size, dim)

        self.test_true = raw / raw.norm(dim=1, keepdim=True)  # unit norm

        noise = noise_eps * torch.randn(self.size, dim)
        raw_noise = noise + raw
        self.test_corr = raw_noise / raw_noise.norm(dim=1, keepdim=True)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.test_corr[idx], self.test_true[idx]  # shape: (512,)


# ======================== Model ===============================================


class Autoencoder(nn.Module):
    """
    Simple symmetric encoder/decoder.
    Bottleneck is intentionally smaller than input — change freely.
    """

    def __init__(self, input_dim: int = 512, bottlenecks: int = [256, 512, 256]):
        super().__init__()

        self.input_dim = input_dim
        self.bottlenecks = bottlenecks

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottlenecks[0]),
            nn.Tanh(),
            nn.Linear(bottlenecks[0], bottlenecks[1]),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottlenecks[1], bottlenecks[2]),
            nn.Tanh(),
            nn.Linear(bottlenecks[2], input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LossCriterion(nn.Module):
    def __init__(self, mse_weight=1.0, cos_weight=0.005, eps=1e-8):
        super().__init__()
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        self.eps = eps

    def forward(self, pred, target):
        # MSE loss
        mse = torch_F.mse_loss(pred, target)
        mse = self.mse_weight * mse

        # cosine similarity loss (1 - cos sim)
        pred_norm = torch_F.normalize(pred, dim=1, eps=self.eps)
        target_norm = torch_F.normalize(target, dim=1, eps=self.eps)

        cos = (pred_norm * target_norm).sum(dim=1).mean()
        cos_loss = 1.0 - cos
        cos_loss = self.cos_weight * cos_loss


        loss =   mse + cos_loss

        return loss, {'cosim_loss': cos_loss, "mse_loss": mse}

# ====================== Train loop ============================================


def train(model, loader, optimizer, criterion, losstracker, device):
    model.train()

    for inputv, gtruth in tqdm(loader):
        inputv = inputv.to(device)
        gtruth = gtruth.to(device)

        optimizer.zero_grad()
        recon = model(inputv)
        loss, ls_dict = criterion(recon, gtruth)   # target == input for identity/AE
        loss.backward()
        optimizer.step()

        losstracker['total_loss'].append(loss.item())

        for k,l in ls_dict.items(): losstracker[k].append(l.item())

    return np.array(losstracker['total_loss']).mean()



def test(model, loader, criterion, measurer, device):
    model.eval()
    total_loss = []

    with torch.no_grad():
        for inputv, gtruth in loader:
            inputv = inputv.to(device)
            gtruth = gtruth.to(device)

            recon = model(inputv)
            loss, _ = criterion(recon, gtruth)

            total_loss.append(loss.item())
            measurer.measure(recon, gtruth)

    return np.array(total_loss).mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    SAVEPATH = "/egr/research-sprintai/benja161/BioMetron/agentic_idOBO/HYPES/Adaptaion/Exp2/"

    SEED        = 42
    DIM         = 512
    BOTTLENECK  = [512, 768, 512]
    TRAIN_STEP  = 2000000
    TEST_SIZE   = 5000
    BATCH_SIZE  = 128
    LR          = 1e-4
    # EPOCHS      = 10
    NOISES      = [0, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.0035, 0.004]
    NOISES      = np.exp(np.linspace(-5, 0, 10))


    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(SAVEPATH, exist_ok=True)

    # --- model, loss, optimizer ---
    model     = Autoencoder(input_dim=DIM, bottlenecks=BOTTLENECK).to(device)
    criterion = LossCriterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    measurer = Measurer(test_steps = int(math.ceil(TEST_SIZE/BATCH_SIZE)))
    losstracker = defaultdict(list)


    # --- data ---

    test_dataset  = RandomVectorTestset(sample_count=TEST_SIZE,  dim=DIM, noise_eps=1.0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # --- training ---
    for n, noise in enumerate(NOISES):

        train_dataset = RandomVectorDataset(sample_count=TRAIN_STEP, dim=DIM, noise_eps=noise)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        train_loss = train(model, train_loader, optimizer, criterion, losstracker, device)
        test_loss  = test(model,  test_loader,  criterion,  measurer, device)

        plot_training(losstracker, savepath=SAVEPATH)
        plot_measurer(measurer, savepath=SAVEPATH)

        print(f"Epoch {n:>3} / Noise {noise}  train_loss={train_loss:.6f}  test_loss={test_loss:.6f}")


    for i in range(5):
        sample = test_dataset[i][0].unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(sample)
            print(str(recon.tolist()), file=open(os.path.join(SAVEPATH,"temp.txt"), "a"))


if __name__ == "__main__":
    main()