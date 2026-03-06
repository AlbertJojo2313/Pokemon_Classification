### dataset_creator.py
### author: Albert Jojo

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Constants ─────────────────────────────────────────

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

DEX_RANGES = {
    "kanto": (1, 151),
    "johto": (152, 251),
    "hoenn": (252, 386),
    "sinnoh": (387, 493),
    "unova": (494, 649),
    "kalos": (650, 721),
    "alola": (722, 809),
    "galar": (810, 905),
    "paldea": (906, 1025),
}


# ── Transform Pipeline ─────────────────────────────────


def get_transform(train=True):

    if train:
        augmentations = [
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.8),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(20, 40),
                hole_width_range=(4, 8),
                p=0.5,
            ),
            A.MotionBlur(blur_limit=(3, 7), p=0.2),
        ]
    else:
        augmentations = []

    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            *augmentations,
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


# ── Dataset Class ──────────────────────────────────────


class PokemonDataset(Dataset):
    def __init__(self, samples, train=True):

        self.samples = samples
        self.transform = get_transform(train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ── Helper Functions ───────────────────────────────────


def _get_dex_id(folder_name: str) -> int:
    """Extract dex ID from folder name like 001_Bulbasaur"""

    try:
        dex_str = folder_name.split("_")[0]
        return int(dex_str.lstrip("0") or "0")
    except (ValueError, IndexError):
        raise ValueError(f"Invalid folder format: {folder_name}")


def _in_region(folder_name, dex_range):

    try:
        dex_id = _get_dex_id(folder_name)
        return dex_range[0] <= dex_id <= dex_range[1]
    except ValueError:
        return False


def _collect_samples(data_dir, dex_range):

    data_path = Path(data_dir)

    folders = sorted(
        [
            f
            for f in data_path.iterdir()
            if f.is_dir() and _in_region(f.name, dex_range)
        ],
        key=lambda f: _get_dex_id(f.name),
    )

    if not folders:
        raise ValueError("No Pokémon folders found")

    idx_to_name = {i: folder.name for i, folder in enumerate(folders)}

    samples = []

    print("\nDataset Report")
    print("--------------")

    total_images = 0

    for class_idx, folder in enumerate(folders):
        images = [f for f in folder.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]

        num_imgs = len(images)
        total_images += num_imgs

        print(f"{folder.name}: {num_imgs} images")

        for img in images:
            samples.append((str(img), class_idx))

    print("\nSummary")
    print("-------")
    print(f"Classes: {len(folders)}")
    print(f"Total Images: {total_images}")

    return samples, idx_to_name


def _split_by_class(samples, train_ratio=0.8, val_ratio=0.1, seed=42):

    random.seed(seed)

    class_to_samples = {}

    for sample in samples:
        class_to_samples.setdefault(sample[1], []).append(sample)

    train, val, test = [], [], []

    for class_samples in class_to_samples.values():
        random.shuffle(class_samples)

        n = len(class_samples)

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train += class_samples[:n_train]
        val += class_samples[n_train : n_train + n_val]
        test += class_samples[n_train + n_val :]

    print("\nSplit")
    print("-----")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")

    return train, val, test


# ── Public API ─────────────────────────────────────────


def get_dataloaders(
    base_dir="pokemon_images",
    region="kanto",
    batch_size=BATCH_SIZE,
    seed=42,
):

    script_dir = Path(os.path.abspath(__file__)).parent
    project_root = script_dir.parent
    data_dir = project_root / base_dir

    if region not in DEX_RANGES:
        raise ValueError(f"Unknown region: {region}")

    dex_range = DEX_RANGES[region]

    print(f"\nLoading dataset: {data_dir}")
    print(f"Region: {region} ({dex_range[0]}-{dex_range[1]})")

    samples, idx_to_name = _collect_samples(data_dir, dex_range)

    train_samples, val_samples, test_samples = _split_by_class(
        samples,
        seed=seed,
    )

    train_dataset = PokemonDataset(train_samples, train=True)
    val_dataset = PokemonDataset(val_samples, train=False)
    test_dataset = PokemonDataset(test_samples, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader, idx_to_name


def get_num_classes(region="kanto"):

    start, end = DEX_RANGES[region]
    return end - start + 1
