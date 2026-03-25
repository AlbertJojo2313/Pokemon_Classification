import os
import torch
import torchvision.transforms as transforms
from models.efficientnetb2_allreg import PokemonClassifierEfficientNetB2_3reg


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project Root

IMAGES_DIR = os.path.join(BASE_DIR, "pokemon_images")
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "best_model_kanto_johto_hoenn_sinnoh_efficientnetb2.pth",
)

CLASS_NAMES = sorted(os.listdir(IMAGES_DIR))


def get_transform():
    """Match the exact transforms used during training."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model():
    """Load model from checkpoint."""
    model = PokemonClassifierEfficientNetB2_3reg(num_classes=493)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
