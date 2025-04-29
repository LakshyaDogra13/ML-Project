import io
import re
from typing import List, Dict

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding to improve OCR accuracy
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def extract_text(image: np.ndarray) -> str:
    pil_img = Image.fromarray(image)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(pil_img, config=config)
    return text

def parse_lab_tests(text: str) -> List[Dict]:
    """
    Parses lab test data from OCR text.
    Expected to find lines containing:
    Test Name, Value, Unit, Reference Range
    """
    results = []
    lines = text.split('\n')
    
    # Regex to capture:
    # test name (letters, spaces, parentheses),
    # test value (number, possibly decimal),
    # unit (optional),
    # reference range (e.g. 12.0-15.0 or 12-15)
    pattern = re.compile(
        r'([A-Za-z\s\(\)\-]+?)\s+([\d\.]+)\s*([a-zA-Z/%]*)\s+(\d+\.?\d*)[-–](\d+\.?\d*)'
    )
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if match:
            test_name = match.group(1).strip()
            test_value_str = match.group(2).strip()
            test_unit = match.group(3).strip() if match.group(3) else ""
            ref_low = float(match.group(4))
            ref_high = float(match.group(5))
            
            try:
                test_value = float(test_value_str)
            except:
                # If value can't convert to float, skip this line
                continue
            
            out_of_range = test_value < ref_low or test_value > ref_high
            
            results.append({
                "test_name": test_name,
                "test_value": test_value_str,
                "bio_reference_range": f"{ref_low}-{ref_high}",
                "test_unit": test_unit,
                "lab_test_out_of_range": out_of_range
            })
    return results

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        preprocessed = preprocess_image(image_np)
        text = extract_text(preprocessed)
        lab_tests = parse_lab_tests(text)
        
        return JSONResponse({
            "is_success": True,
            "data": lab_tests
        })
    except Exception as e:
        return JSONResponse({
            "is_success": False,
            "error": str(e)
        }, status_code=500)


