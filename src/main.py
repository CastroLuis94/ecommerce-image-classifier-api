from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
from fastapi import UploadFile, File, HTTPException
import io
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
    "Apparel_Boys/Prenda_chico",
    "Apparel_Girls/Prenda_chica",
    "Footwear_Men/Calzado_hombre",
    "Footwear_Women/Calzado_mujer"
]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer correctamente el archivo
        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Convertir a imagen con PIL
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Transformaciones
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

    except Exception as e:
        print("ERROR:", str(e))  # 🔎 ahora vas a ver el error real en logs
        raise HTTPException(status_code=400, detail="Invalid image file")