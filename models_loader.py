import torch
import timm
import os

# Путь к файлу модели
MODEL_PATH = os.path.join("models", "swa_convnext.pt")


def load_porosity_model():
    """
    Загружает модель определения пористости.
    Используется ConvNeXt (или другая твоя архитектура, если нужно).
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}. "
            f"Убедись, что swa_convnext.pt лежит в папке /models"
        )

    # ЗДЕСЬ ВАЖНО: укажи свою архитектуру модели
    model = timm.create_model(
        "convnext_base",  # если нужна другая — скажи, поправлю
        pretrained=False,
        num_classes=3
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)

    model.eval()

    return model
