from fastapi import APIRouter, File, UploadFile
import torch
from PIL import Image
from api.model_loader import load_model, get_transform
import io


router = APIRouter()

# Load once the model is imported
model, class_names = load_model()
transform = get_transform()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilites = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilites, dim=1)

    return {
        "predicted": class_names[predicted.item()].split("_", 1)[1].capitalize(),
        "confidence": round(confidence.item() * 100, 2),
    }
