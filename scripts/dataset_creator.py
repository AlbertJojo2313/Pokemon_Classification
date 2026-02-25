from torch.utils.data import Dataset
import os
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

"""
PokemonDataset is a PyTorch Dataset class that loads Pokémon images from a structured directory 
where each subdirectory represents a class (e.g. 001_bulbasaur). It automatically maps class folder names 
to integer indices and keeps image paths and labels in sync. Images are loaded as RGB numpy arrays via PIL 
and passed through an Albumentations augmentation pipeline that includes resizing to 224x224, horizontal flipping, 
shift/scale/rotate, color jitter, and coarse dropout, followed by ImageNet normalization and conversion to a PyTorch tensor. 
The transform pipeline supports a train flag to apply augmentations only during training and skip them during validation. 
Basic error handling is included in __getitem__ to surface failed image loads with a descriptive message.
"""


def get_transform(train=True):

    if train:
        transformations = [
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.8),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(4, 8),
                hole_width_range=(4, 8),
                p=0.5,
            ),
        ]

    else:
        transformations = []

    return A.Compose(
        [
            A.Resize(224, 224),
            *transformations,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class PokemonDataset(Dataset):
    def __init__(self, base_dir="pokemon_images"):
        self.base_dir = os.path.join(Path(__file__).resolve().parent.parent, base_dir)

        self.classes = sorted(
            os.listdir(self.base_dir)
        )  # ["001_bulbasaur", "002_ivysaur"]
        self.classes_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        # makes everything in sync
        for class_name in self.classes:
            class_dir = os.path.join(self.base_dir, class_name)
            for image_file in os.listdir(class_dir):
                if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, image_file))
                    self.labels.append(self.classes_to_idx[class_name])
        self.transform = get_transform(train=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        try:
            images = self.image_paths[idx]
            label = self.labels[idx]

            image = np.array(Image.open(images).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Failed to load image {images}: {e}")
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
