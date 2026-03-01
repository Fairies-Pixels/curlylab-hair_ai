import pytest
from unittest.mock import patch, MagicMock

import models_loader


def test_model_file_missing():
    """Ошибка если файл модели отсутствует"""

    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            models_loader.load_porosity_model()


def test_model_creation_called():
    """Проверяем что создается модель"""

    mock_model = MagicMock()

    with patch("os.path.exists", return_value=True), \
         patch("timm.create_model", return_value=mock_model) as mock_create, \
         patch("torch.load", return_value={"model_state_dict": {}}):

        models_loader.load_porosity_model()

        mock_create.assert_called_once()


def test_model_creation_parameters():
    """Проверяем параметры архитектуры"""

    mock_model = MagicMock()

    with patch("os.path.exists", return_value=True), \
         patch("timm.create_model", return_value=mock_model) as mock_create, \
         patch("torch.load", return_value={"model_state_dict": {}}):

        models_loader.load_porosity_model()

        mock_create.assert_called_with(
            "convnext_base",
            pretrained=False,
            num_classes=3
        )


def test_state_dict_loaded_into_model():
    """Проверяем загрузку весов"""

    mock_model = MagicMock()
    checkpoint = {"model_state_dict": {"layer": "weights"}}

    with patch("os.path.exists", return_value=True), \
         patch("timm.create_model", return_value=mock_model), \
         patch("torch.load", return_value=checkpoint):

        models_loader.load_porosity_model()

        mock_model.load_state_dict.assert_called_once_with(checkpoint["model_state_dict"])


def test_model_set_to_eval():
    """Модель переводится в eval режим"""

    mock_model = MagicMock()

    with patch("os.path.exists", return_value=True), \
         patch("timm.create_model", return_value=mock_model), \
         patch("torch.load", return_value={"model_state_dict": {}}):

        models_loader.load_porosity_model()

        mock_model.eval.assert_called_once()
import pytest
from unittest.mock import patch, MagicMock
import torch
from models_loader import load_porosity_model

def test_load_model_file_not_found():
    """Проверка ошибки, если файл модели отсутствует"""
    with patch("models_loader.os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_porosity_model()

def test_load_model_returns_module():
    """Проверка, что возвращается torch.nn.Module"""
    mock_model = MagicMock()
    mock_state_dict = {"model_state_dict": {}}

    with patch("models_loader.os.path.exists", return_value=True), \
         patch("models_loader.timm.create_model", return_value=mock_model), \
         patch("models_loader.torch.load", return_value=mock_state_dict):
        model = load_porosity_model()
        assert model == mock_model

def test_load_model_calls_create_and_load_state_dict():
    """Проверка, что create_model и load_state_dict вызываются"""
    mock_model = MagicMock()
    mock_state_dict = {"model_state_dict": {"key": "value"}}

    with patch("models_loader.os.path.exists", return_value=True), \
         patch("models_loader.timm.create_model", return_value=mock_model) as mock_create, \
         patch("models_loader.torch.load", return_value=mock_state_dict) as mock_torch_load:
        load_porosity_model()
        mock_create.assert_called_once()
        mock_model.load_state_dict.assert_called_once_with(mock_state_dict["model_state_dict"])
