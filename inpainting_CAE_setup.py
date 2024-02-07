import torch
import glob
from PIL import Image


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, image_paths=glob.glob("celeba_hq_256/*.jpg"), transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
