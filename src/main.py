from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()

# carga modelo
import torch.nn as nn
import torchvision.models as models

# 1️⃣ Crear arquitectura
model = models.resnet18(weights=None)

# reemplazar última capa
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 clases

# 2️⃣ Cargar pesos
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.to("cpu")
model.eval()

# transformaciones
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class_names = [
    "Apparel_Boys",
    "Apparel_Girls",
    "Footwear_Men",
    "Footwear_Women"
]

from fastapi import HTTPException

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            preds = model(x)
            pred_class = preds.argmax(1).item()
            probs = torch.softmax(preds, dim=1)
            confidence = probs[0][pred_class].item()

        return {
            "pred_class_index": pred_class,
            "pred_class_name": class_names[pred_class],
            "confidence": round(confidence, 4)
        }

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")