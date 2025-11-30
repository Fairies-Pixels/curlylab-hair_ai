import torch
import timm
import os

MODEL_PATH = os.path.join("models", "C:/Users/Alina/curlylab-hair_ai/models/swa_convnext.pt")


def load_porosity_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}. "
            f"Убедись, что swa_convnext.pt лежит в папке /models"
        )

    model = timm.create_model(
        "convnext_base",
        pretrained=False,
        num_classes=3
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model
