import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class EcommerceDataset(Dataset):
    def __init__(self, csv_path, image_root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df["Image"] = self.df["Image"].str.strip()

        self.image_root_dir = image_root_dir
        self.transform = transform

        self.classes = sorted(self.df["Category"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = {}
        for root, _, files in os.walk(self.image_root_dir):
            for file in files:
                self.image_paths[file] = os.path.join(root, file)
        self.class_to_idx = {
            "Apparel_Boys": 0,
            "Apparel_Girls": 1,
            "Footwear_Men": 2,
            "Footwear_Women": 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image"]

        # Crear label combinando Category + Gender
        combined_label = f"{row['Category']}_{row['Gender']}"
        label = self.class_to_idx[combined_label]

        img_path = self.image_paths.get(img_name)
        if img_path is None:
            raise FileNotFoundError(f"{img_name} not found.")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)