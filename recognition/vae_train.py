"""
REF: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from vae_utils import Encoder, Decoder, VAE, OasisDataset, transform
from torch.optim import Adam
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


batch_size = 64

x_dim = 4096
hidden_dim = 256
latent_dim = 64

lr = 1e-3

epochs = 10

train_dataset = OasisDataset(
    "/home/groups/comp3710/OASIS/keras_png_slices_train", transform=transform
)
val_dataset = OasisDataset(
    "/home/groups/comp3710/OASIS/keras_png_slices_validate", transform=transform
)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

model = VAE(Encoder=encoder, Decoder=decoder, device=device).to(device)


def loss_function(x, x_hat, mean, log_var):
    """
    Computes the VAE loss function. comprised of 2 parts:
    1. reconstruction loss - measures how well the decoder can construct an image from latent distribution, using BCE
    2. KL divergence - regularisation term
    """
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, x in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(
        "\tEpoch",
        epoch + 1,
        "complete!",
        "\tAverage Loss: ",
        overall_loss / (batch_idx * batch_size),
    )

# Save model to disk
save_path = str(Path.home() / "comp3710/lab2/recognition/vae.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
