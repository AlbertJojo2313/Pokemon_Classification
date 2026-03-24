## author: Albert Jojo

"""
Testing the model performance with unseen data, retrieved the data from kaggle.
"""

import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import PokemonClassifierEfficientNetB2_4reg


def get_transform():
    """Match the exact transforms used during training."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(checkpoint_path, num_classes, device):
    """Load model from checkpoint."""
    model = PokemonClassifierEfficientNetB2_4reg(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"Checkpoint keys: {checkpoint.keys()}")  # remove after confirming key
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    print(f"✅ Model loaded")
    return model


def predict(model, img_path, class_names, device, top_k=3):
    """
    Predict the Pokémon in a single uploaded image.
    Returns top-k predictions with confidence scores.
    """
    transform = get_transform()

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    results = [
        {
            "rank": i + 1,
            "pokemon": class_names[idx.item()].capitalize(),
            "confidence": prob.item(),
        }
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]))
    ]

    return results


def main():
    # --- Config ---
    CHECKPOINT_PATH = (
        "outputs/checkpoints/best_model_kanto_johto_hoenn_sinnoh_efficientnetb2.pth"
    )
    CSV_PATH = "test_images/pokemon_filtered.csv"
    TEST_IMG_PATH = "test_images/images/000_bulbasaur.png"  # swap for any test image
    NUM_CLASSES = 493

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Class names in Pokédex order ---
    df = pd.read_csv(CSV_PATH)
    class_names = df["Name"].str.lower().str.strip().tolist()

    # --- Load model ---
    model = load_model(CHECKPOINT_PATH, NUM_CLASSES, device)

    # --- Predict ---
    results = predict(model, TEST_IMG_PATH, class_names, device, top_k=3)

    print(f"\nImage: {os.path.basename(TEST_IMG_PATH)}")
    print("Top predictions:")
    for r in results:
        print(f"  #{r['rank']} {r['pokemon']:20} — {r['confidence']:.2%} confidence")


if __name__ == "__main__":
    main()

"""
**What this does:**
- Takes a single image path (simulating a user upload)
- Returns **top-3 predictions** with confidence scores, e.g:
```
Image: bulbasaur.png
Top predictions:
  #1 Bulbasaur             — 97.32% confidence
  #2 Ivysaur               — 1.84% confidence
  #3 Venusaur              — 0.51% confidence
"""
