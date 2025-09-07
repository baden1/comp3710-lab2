"""
REF: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
"""

import os
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from vae_utils import Encoder, Decoder, VAE, OasisDataset, transform
from torch.optim import Adam
from pathlib import Path
import matplotlib.pyplot as plt

batch_size = 64
x_dim = 4096
hidden_dim = 256
latent_dim = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

test_dataset = OasisDataset(
    "/home/groups/comp3710/OASIS/keras_png_slices_test", transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

# model = VAE(Encoder=encoder, Decoder=decoder, device=device).to(device)
def load_from_disk(path):
    loaded_model = VAE(Encoder=encoder, Decoder=decoder, device=device)
    loaded_model.load_state_dict(torch.load(path, map_location=device))
    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


model = load_from_disk(str(Path.home() / "comp3710/lab2/recognition/vae.pth"))
model.eval()

# take one test batch, flatten the images into vectors, get reconstructions from trained vae
with torch.no_grad():
    for batch_idx, x in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(device)

        x_hat, _, _ = model(x)

        break

def show_image(x, idx, label):
    x = x.view(batch_size, 64, 64)  # (batch_size) lots of 64x64 images

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())
    plt.savefig(str(Path.home() / f"comp3710/lab2/recognition/vae_images/{label}/{idx}.png"))
    

for i in range(10):
    # show original testset image
    show_image(x, i, label="original")
    # show reconstructed image
    show_image(x_hat, i, label="reconstructed")