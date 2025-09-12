from unet_model import UNet
from unet_utils import *
import torch

def dice_coeff(pred, target, epsilon=1e-6):
    """
    Compute Dice coefficient for one sample
    pred, target: [C, H, W], one-hot tensors (float)
    """
    assert pred.shape == target.shape
    intersection = (pred * target).sum(dim=(1,2))   # sum over H,W per class
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    dice = (2 * intersection + epsilon) / (union + epsilon)  # per class
    return dice

batch_size = 8

# load model
model = UNet()
state_dict = torch.load('H:\\comp3710\\comp3710-lab2\\recognition\\unet\\unet.pth')
model.load_state_dict(state_dict)
model.eval()

# load test data sets - images and masks
test_masks_dataset = OasisDataset(
    "H:\\comp3710\\oasis\\OASIS\\keras_png_slices_seg_test", transform=mask_transform
)
test_images_dataset = OasisDataset(
    "H:\\comp3710\\oasis\\OASIS\\keras_png_slices_test", transform=transform
)

test_masks_loader = DataLoader(test_masks_dataset, batch_size=batch_size, shuffle=False)
test_images_loader = DataLoader(test_images_dataset, batch_size=batch_size, shuffle=False)

dice_scores = []
for i, (masks, images) in enumerate(zip(test_masks_loader, test_images_loader)):
    # make mask from test image

    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)  # convert logits to probabilities

    # Convert predictions to one-hot: take argmax per pixel
    preds = torch.argmax(probs, dim=1)      # [B,H,W], integer class
    preds_onehot = torch.nn.functional.one_hot(preds, num_classes=4)  # [B,H,W,C]
    preds_onehot = preds_onehot.permute(0,3,1,2).float()  # [B,C,H,W]

    # Compute Dice per sample
    for i in range(preds_onehot.size(0)):
        dice = dice_coeff(preds_onehot[i], masks[i])
        dice_scores.append(dice.item())
