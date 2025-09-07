import torch
from torch.utils.data import Dataset, DataLoader
from vae_utils import Encoder, Decoder, VAE, OasisDataset, transform
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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

latents = []
with torch.no_grad():
    for x in test_loader:
        x = x.to(device)
        x = x.view(x.size(0), -1)  # flatten to vector
        _, mu, _ = model(x)  # take mean vector from encoder
        latents.append(mu.cpu())

latents = torch.cat(latents, dim=0).numpy()  # shape: [N, latent_dim]

# plot the first two principal components of where images land in the latent space
pca = PCA(n_components=2)
latents_2d = pca.fit_transform(latents)

plt.figure(figsize=(8, 6))
plt.scatter(latents_2d[:, 0], latents_2d[:, 1])
plt.title("PCA projection of test data to latent space")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()
plt.savefig(Path.home() / "comp3710/lab2/recognition/vae_images/plots/pca-test-to-latent.png")
