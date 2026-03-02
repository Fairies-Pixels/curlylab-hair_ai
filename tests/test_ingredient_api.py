import io
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient

import consists_check_service as api

client = TestClient(api.app)
import consists_check_service as api



def create_test_image():
    img = Image.new("RGB", (100, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_normalize_lowercase():
    assert api.normalize("Dimethicone") == "dimethicone"


def test_normalize_removes_symbols():
    result = api.normalize("Dimethicone!!!")
    assert "!" not in result


def test_split_ingredients_commas():
    text = "water, dimethicone, alcohol"
    parts = api.split_ingredients(text)
    assert "dimethicone" in parts


def test_split_ingredients_newlines():
    text = "water\ndimethicone"
    parts = api.split_ingredients(text)
    assert len(parts) == 2



def test_match_exact_name():
    result = api.match_exact_db("dimethicone", api.INGREDIENT_DB)
    assert result["match_type"] == "exact_name"


def test_match_exact_alias():
    result = api.match_exact_db("sles", api.INGREDIENT_DB)
    assert result["match_type"] == "exact_alias"


def test_regex_detects_sulfate():
    result = api.match_regex("sodium sulfate")
    assert result["category"] == "sulfates"


def test_regex_respects_exception():
    result = api.match_regex("cetyl alcohol")
    assert result is None



def test_fuzzy_match_typo():
    result = api.match_fuzzy("dimethicon", api.INGREDIENT_DB)
    assert result["match_type"] == "fuzzy"



def test_analyze_text_composition_detects_issue():
    text = "water, dimethicone"
    issues = api.analyze_text_composition(text, api.INGREDIENT_DB)

    assert len(issues) > 0
    assert any("dimethicone" in i["ingredient"] for i in issues)

def test_analyze_text_composition_no_issue():
    text = "water, glycerin"
    issues = api.analyze_text_composition(text, api.INGREDIENT_DB)

    assert isinstance(issues, list)



def test_api_text_input():
    response = client.post(
        "/analyze",
        data={"text": "water, dimethicone"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["issues_count"] >= 1


def test_api_requires_input():
    response = client.post("/analyze")
    assert response.status_code == 500 or response.status_code == 400


def test_api_image_ocr():
    img = create_test_image()

    with patch("consists_check_service.ocr_image_to_text", return_value="water, dimethicone"):

        response = client.post(
            "/analyze",
            files={"file": ("test.png", img, "image/png")}
        )

        assert response.status_code == 200
        assert response.json()["issues_count"] > 0

def test_api_text_priority_over_file():
    img = create_test_image()

    with patch("consists_check_service.ocr_image_to_text", return_value="water"):

        response = client.post(
            "/analyze",
            data={"text": "water, dimethicone"},
            files={"file": ("test.png", img, "image/png")}
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True

def test_regex_no_match():
    result = api.match_regex("glycerin")
    assert result is None

def test_split_ingredients_empty():
    parts = api.split_ingredients("")
    assert parts == []


def test_split_ingredients_complex():
    text = "water, (dimethicone; alcohol)/glycerin"
    parts = api.split_ingredients(text)
    assert "dimethicone" in parts
    assert "glycerin" in parts
    assert len(parts) == 4


def test_match_exact_db_not_found():
    result = api.match_exact_db("unknowningredient", api.INGREDIENT_DB)
    assert result is None

def test_match_exact_db_empty_string():
    result = api.match_exact_db("", api.INGREDIENT_DB)
    assert result is None


def test_match_regex_exception_handling():
    # cetyl alcohol is in EXCEPTIONS, should skip
    result = api.match_regex("cetyl alcohol")
    assert result is None

def test_match_fuzzy_low_score_returns_none():
    result = api.match_fuzzy("zzzzzz", api.INGREDIENT_DB)
    assert result is None

def test_match_fuzzy_multiple_aliases():
    # amodimethicone has alias "amino-functional dimethicone"
    result = api.match_fuzzy("amino-functional dimethicone", api.INGREDIENT_DB)
    assert result["match_type"] == "fuzzy"


def test_analyze_text_composition_empty_string():
    issues = api.analyze_text_composition("", api.INGREDIENT_DB)
    assert issues == []

def test_analyze_text_composition_short_tokens():
    # single character tokens are skipped
    issues = api.analyze_text_composition("a,b,c", api.INGREDIENT_DB)
    assert issues == []

def test_analyze_text_composition_regex_warning():
    issues = api.analyze_text_composition("sodium sulfate", api.INGREDIENT_DB)
    assert issues[0]["match_type"] == "pattern"
    assert issues[0]["category"] == "sulfates"


def test_ocr_image_to_text_returns_string_mocked():
    img = Image.new("RGB", (10, 10), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with patch("consists_check_service.pytesseract.image_to_string", return_value="mocked text"):
        text = api.ocr_image_to_text(buf.getvalue())
        assert isinstance(text, str)
        assert text == "mocked text"