import pytest
from unittest.mock import patch, MagicMock
import torch
from models_loader import load_porosity_model

import models_loader


def test_model_file_missing():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            models_loader.load_porosity_model()


def test_model_creation_called():
    mock_model = MagicMock()

    with patch("os.path.exists", return_value=True), \
         patch("timm.create_model", return_value=mock_model) as mock_create, \
         patch("torch.load", return_value={"model_state_dict": {}}):

        models_loader.load_porosity_model()

        mock_create.assert_called_once()


def test_model_creation_parameters():
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

    mock_model = MagicMock()
    checkpoint = {"model_state_dict": {"layer": "weights"}}

    with patch("os.path.exists", return_value=True), \
         patch("timm.create_model", return_value=mock_model), \
         patch("torch.load", return_value=checkpoint):

        models_loader.load_porosity_model()

        mock_model.load_state_dict.assert_called_once_with(checkpoint["model_state_dict"])


def test_model_set_to_eval():
    mock_model = MagicMock()

    with patch("os.path.exists", return_value=True), \
         patch("timm.create_model", return_value=mock_model), \
         patch("torch.load", return_value={"model_state_dict": {}}):

        models_loader.load_porosity_model()

        mock_model.eval.assert_called_once()

def test_load_model_file_not_found():
    with patch("models_loader.os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_porosity_model()

def test_load_model_returns_module():
    mock_model = MagicMock()
    mock_state_dict = {"model_state_dict": {}}

    with patch("models_loader.os.path.exists", return_value=True), \
         patch("models_loader.timm.create_model", return_value=mock_model), \
         patch("models_loader.torch.load", return_value=mock_state_dict):
        model = load_porosity_model()
        assert model == mock_model

def test_load_model_calls_create_and_load_state_dict():
    mock_model = MagicMock()
    mock_state_dict = {"model_state_dict": {"key": "value"}}

    with patch("models_loader.os.path.exists", return_value=True), \
         patch("models_loader.timm.create_model", return_value=mock_model) as mock_create, \
         patch("models_loader.torch.load", return_value=mock_state_dict) as mock_torch_load:
        load_porosity_model()
        mock_create.assert_called_once()
        mock_model.load_state_dict.assert_called_once_with(mock_state_dict["model_state_dict"])
