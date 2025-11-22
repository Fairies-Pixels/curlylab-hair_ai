from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

from models_loader import load_porosity_model


app = FastAPI(title="Hair Porosity API", version="1.0")

# Загружаем модель один раз при старте сервиса
porosity_model = load_porosity_model()

# Преобразование изображения под ConvNeXt
porosity_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


def predict_porosity(image: Image.Image) -> str:
    """
    Предсказание пористости волос.
    """
    input_tensor = porosity_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = porosity_model(input_tensor)
        pred = output.argmax(dim=1).item()

    classes = ["low", "high", "medium"]  # подгони, если другой порядок!
    return classes[pred]


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(("jpg", "jpeg", "png")):
        raise HTTPException(status_code=400,
                            detail="Файл должен быть изображением JPG или PNG")

    # Читаем изображение
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Предсказание
    porosity = predict_porosity(image)

    return JSONResponse({
        "porosity": porosity
    })
