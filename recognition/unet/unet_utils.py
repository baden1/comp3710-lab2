from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os

class MaskToOneHot:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes

    def __call__(self, mask):
        # mask is already a tensor [1, H, W] from ToTensor
        mask = mask.squeeze(0)  # [H, W]
        mask = (mask * (self.num_classes - 1)).long()  # convert 0,1/3,2/3,1 -> 0,1,2,3
        one_hot = F.one_hot(mask, num_classes=self.num_classes)  # [H, W, C]
        one_hot = one_hot.permute(2, 0, 1).float()  # [C, H, W]
        return one_hot

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    MaskToOneHot(num_classes=4),
])

    

class OasisDataset(Dataset):
    """
    Custom dataset for loaing oasis png files. REF: Chatgpt
    https://chatgpt.com/c/68bd6eff-ae04-8321-b273-83610d1e7fa4
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".png")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img
