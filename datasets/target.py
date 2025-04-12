import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import params
from PIL import Image
import os

class TargetImageDataset(Dataset):
    """Custom dataset for target domain with fixed label = 0."""

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # or "L" for grayscale

        if self.transform:
            image = self.transform(image)

        label = 0  # fixed label for target domain
        return image, label

def get_target():
    """Get target dataset loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_dataset = TargetImageDataset(
        image_dir='/home/james/MyFolder/code/GIT/pytorch-adda/data/target',
        transform=transform)

    target_data_loader = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return target_data_loader