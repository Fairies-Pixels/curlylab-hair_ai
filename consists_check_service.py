from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import pytesseract
from PIL import Image
import io
import re
import os
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import spacy

nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Ingredient Analyzer")


INGREDIENT_DB = {
    "dimethicone": {"category": "Silicones", "status": "warning", "aliases": ["dimethicone", "dimethiconol", "dimethylpolysiloxane"], "reason": "Non-water-soluble silicone — может накапливаться на волосах."},
    "amodimethicone": {"category": "Silicones", "status": "warning", "aliases": ["amodimethicone", "amino-functional dimethicone"], "reason": "Aminated silicone — тяжело смывается у некоторых людей."},
    "sodium laureth sulfate": {"category": "Surfactants", "status": "warning", "aliases": ["sodium laureth sulfate", "sles"], "reason": "Сульфаты — могут сушить волосы и кожу головы."},
    "cetyl alcohol": {"category": "Alcohols", "status": "ok", "aliases": ["cetyl alcohol"], "reason": "Фатти-алкоголь, как правило не сушит."},
}

BAD_PATTERNS = {
    "silicones": [
        r"\w*cone\b", r"dimethi", r"\bsil\b", r"siloxane", r"silsesquioxane", r"silylate"
    ],
    "waxes": [r"\bcera", r"\bcire", r"\bwax\b", r"petrolatum", r"\bparaffin\b"],
    "sulfates": [r"\bsulfate\b", r"\bsulphate\b"],
    "alcohols": [r"\balcohol\b", r"ethyl alcohol", r"isopropyl alcohol", r"propyl alcohol", r"sd alcohol", r"isopropanol", r"2-propanol"],
    "soap": [r"saponified", r"\bsoap\b", r"sodium palm", r"sodium carboxylate"]
}

EXCEPTIONS = {
    "silicones": ["peg-12 dimethicone", "peg/ppg-18/18 dimethicone"],
    "waxes": ["peg-8 beeswax", "emulsifying wax"],
    "sulfates": ["behentrimonium methosulfate"],
    "alcohols": ["cetyl alcohol", "stearyl alcohol", "cetearyl alcohol"],
}

FUZZY_THRESHOLD = 85


def normalize(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^a-z0-9\/\-\s]", " ", text)

def split_ingredients(text: str) -> List[str]:
    text = text.replace("\n", ", ")
    parts = re.split(r"[,\;\/\(\)\[\]]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def ocr_image_to_text(image_bytes: bytes, lang: str = "eng") -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def load_ingredient_db_from_csv(path: str):
    import csv
    db = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = normalize(r["name"])
            aliases = [normalize(a).strip() for a in (r.get("aliases") or "").split("|") if a.strip()]
            db[name] = {
                "category": r.get("category", ""),
                "status": r.get("status", "ok"),
                "aliases": aliases + [name],
                "reason": r.get("reason", "")
            }
    return db

def match_exact_db(token: str, db: Dict[str, Dict[str, Any]]):
    tk = normalize(token)
    for name, info in db.items():
        # check exact name
        if tk == name:
            return {"match_type": "exact_name", "name": name, "info": info, "score": 100}
        # check aliases
        for alias in info.get("aliases", []):
            if tk == alias:
                return {"match_type": "exact_alias", "name": name, "info": info, "score": 98}
    return None

def match_regex(token: str):
    tk = normalize(token)
    for category, patterns in BAD_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, tk):
                # check exceptions
                for ex in EXCEPTIONS.get(category, []):
                    if ex in tk:
                        return None
                return {"match_type": "regex", "category": category, "pattern": pat, "score": 90}
    return None

def match_fuzzy(token: str, db: Dict[str, Dict[str, Any]], top_n: int = 3):
    choices = []
    for name, info in db.items():
        choices.append((name, name))
        for alias in info.get("aliases", []):
            if alias != name:
                choices.append((name, alias))
    strings = [s for _, s in choices]
    results = rf_process.extract(token, strings, scorer=rf_fuzz.QRatio, limit=top_n)
    best = None
    for matched_string, score, idx in results:
        if score >= FUZZY_THRESHOLD:
            canonical_name = choices[idx][0]
            info = db[canonical_name]
            return {"match_type": "fuzzy", "name": canonical_name, "matched_string": matched_string, "info": info, "score": int(score)}
    return None

def analyze_text_composition(raw_text: str, db: Dict[str, Dict[str, Any]]):
    norm_text = normalize(raw_text)
    tokens = split_ingredients(norm_text)
    found = []
    seen_names = set()

    for tok in tokens:
        if len(tok) < 2:
            continue

        r = match_exact_db(tok, db)
        if r:
            cname = r["name"]
            if cname not in seen_names:
                entry = {
                    "ingredient": cname,
                    "match_type": r["match_type"],
                    "score": r["score"],
                    "category": r["info"].get("category"),
                    "status": r["info"].get("status"),
                    "reason": r["info"].get("reason")
                }
                found.append(entry)
                seen_names.add(cname)
            continue

        r = match_regex(tok)
        if r:
            entry = {
                "ingredient": tok,
                "match_type": "pattern",
                "pattern": r["pattern"],
                "score": r["score"],
                "category": r["category"],
                "status": "warning",
                "reason": "Возможный нежелательный ингредиент"
            }
            found.append(entry)
            continue

        # 3) fuzzy with DB
        r = match_fuzzy(tok, db)
        if r:
            cname = r["name"]
            if cname not in seen_names:
                entry = {
                    "ingredient": cname,
                    "matched_as": r.get("matched_string"),
                    "match_type": "fuzzy",
                    "score": r["score"],
                    "category": r["info"].get("category"),
                    "status": r["info"].get("status"),
                    "reason": r["info"].get("reason")
                }
                found.append(entry)
                seen_names.add(cname)
            continue

    return found

@app.post("/analyze")
async def analyze(file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), use_ocr_lang: Optional[str] = Form("eng")):
    try:
        if not file and not text:
            raise HTTPException(status_code=400, detail="Нужно прислать файл или текст (form field 'text').")

        raw_text = ""
        if file:
            data = await file.read()
            raw_text = ocr_image_to_text(data, lang=use_ocr_lang)

        if text:
            raw_text = (raw_text + "\n" + text) if raw_text else text

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="После OCR / передачи текста не осталось текста для анализа.")

        issues = analyze_text_composition(raw_text, INGREDIENT_DB)

        pretty_issues = [
            {
                "ingredient": issue["ingredient"],
                "category": issue.get("category"),
                "reason": issue.get("reason")
            }
            for issue in issues
        ]

        if not pretty_issues:
            return JSONResponse(content={
                "ok": True,
                "raw_text_excerpt": raw_text[:200],
                "issues_count": 0,
                "issues": [],
                "result": "Состав отличный! "
            })

        return JSONResponse(content={
            "ok": True,
            "raw_text_excerpt": raw_text[:200],
            "issues_count": len(pretty_issues),
            "issues": pretty_issues,
            "result": "Некоторые ингредиенты могут не подойти"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
