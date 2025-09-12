import torch
from unet_model import UNet
from unet_utils import OasisDataset, transform, mask_transform
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

batch_size = 8

lr = 1e-3
epochs = 4

def train():
    train_masks_dataset = OasisDataset(
        "H:\\comp3710\\oasis\\OASIS\\keras_png_slices_seg_train", transform=mask_transform
    )
    val_masks_dataset = OasisDataset(
        "H:\\comp3710\\oasis\\OASIS\\keras_png_slices_seg_validate", transform=mask_transform
    )
    train_images_dataset = OasisDataset(
        "H:\\comp3710\\oasis\\OASIS\\keras_png_slices_train", transform=transform
    )
    val_images_dataset = OasisDataset(
        "H:\\comp3710\\oasis\\OASIS\\keras_png_slices_validate", transform=transform
    )
    val_masks_loader = DataLoader(val_masks_dataset, batch_size=batch_size, shuffle=False)
    train_masks_loader = DataLoader(train_masks_dataset, batch_size=batch_size, shuffle=True)
    val_images_loader = DataLoader(val_images_dataset, batch_size=batch_size, shuffle=False)
    train_images_loader = DataLoader(train_images_dataset, batch_size=batch_size, shuffle=True)

    
    model = UNet(n_channels=1, n_classes=4, checkpoint=False, bilinear=True).to(device)  # segment the image into 4 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        running_loss = 0.0
        for i, (masks, images) in enumerate(zip(train_masks_loader, train_images_loader)):
            images, masks = images.to(device), masks.to(device)
            # target = torch.argmax(masks, dim=1)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_masks_loader):.4f}")

    torch.save(model.state_dict(), "H:\\comp3710\\comp3710-lab2\\recognition\\unet\\unet.pth")
    print("Model saved.")


if __name__ == '__main__':
    train()