import main as api
import io
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

def create_test_image():
    img = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_predict_porosity_returns_class():

    mock_model = MagicMock()
    mock_output = MagicMock()

    mock_output.argmax.return_value.item.return_value = 1
    mock_model.return_value = mock_output

    with patch.object(api, "porosity_model", mock_model):
        img = Image.new("RGB", (300, 300))
        result = api.predict_porosity(img)

        assert result in ["low", "high", "medium"]


def test_predict_porosity_calls_model():

    mock_model = MagicMock()
    mock_output = MagicMock()

    mock_output.argmax.return_value.item.return_value = 0
    mock_model.return_value = mock_output

    with patch.object(api, "porosity_model", mock_model):
        img = Image.new("RGB", (300, 300))
        api.predict_porosity(img)

        assert mock_model.called


def test_analyze_endpoint_success():

    # важно: патч на main.predict_porosity
    with patch("main.predict_porosity", return_value="low"):

        image = create_test_image()

        response = client.post(
            "/analyze",
            files={"file": ("test.png", image, "image/png")}
        )

        assert response.status_code == 200
        assert response.json()["porosity"] == "low"


def test_analyze_rejects_wrong_filetype():

    response = client.post(
        "/analyze",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 400
    assert "изображением" in response.json()["detail"]


def test_analyze_calls_predict():

    with patch("main.predict_porosity", return_value="medium") as mock_predict:

        image = create_test_image()

        client.post(
            "/analyze",
            files={"file": ("img.png", image, "image/png")}
        )

        assert mock_predict.called


def create_test_image():
    img = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_predict_porosity_returns_class():
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.argmax.return_value.item.return_value = 1
    mock_model.return_value = mock_output

    with patch.object(main, "porosity_model", mock_model):
        img = Image.new("RGB", (300, 300))
        result = main.predict_porosity(img)
        assert result in ["low", "high", "medium"]

def test_predict_porosity_class_mapping():
    mock_model = MagicMock()
    mock_output = MagicMock()
    for idx, expected in enumerate(["low", "high", "medium"]):
        mock_output.argmax.return_value.item.return_value = idx
        mock_model.return_value = mock_output
        with patch.object(main, "porosity_model", mock_model):
            img = Image.new("RGB", (300, 300))
            result = main.predict_porosity(img)
            assert result == expected

def test_porosity_transform_output():
    img = Image.new("RGB", (300, 300))
    tensor = main.porosity_transform(img)
    import torch
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 3

def test_analyze_endpoint_success():
    with patch("main.predict_porosity", return_value="low"):
        image = create_test_image()
        response = client.post("/analyze", files={"file": ("test.png", image, "image/png")})
        assert response.status_code == 200
        assert response.json()["porosity"] == "low"

def test_analyze_rejects_wrong_filetype():
    response = client.post("/analyze", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "изображением" in response.json()["detail"]

def test_analyze_calls_predict():
    with patch("main.predict_porosity", return_value="medium") as mock_predict:
        image = create_test_image()
        client.post("/analyze", files={"file": ("img.png", image, "image/png")})
        assert mock_predict.called
