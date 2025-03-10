import os
import re
import cv2
import fitz  # PyMuPDF
import shutil
import base64
import tempfile
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from openai import OpenAI
from transformers import pipeline

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Receipt Analyzer API ")

# Zero-Shot Classifier (CLIP model)
model_name = "openai/clip-vit-large-patch14-336"
classifier = pipeline("zero-shot-image-classification", model=model_name)
labels = ["Transaction receipt", "An image that is not a transaction receipt","exchange trade rate","other"]
CONFIDENCE_THRESHOLD = 0.7

client = OpenAI(api_key='OPENAI_API_KEY')

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to improve local contrast in a grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def threshold_image(gray: np.ndarray) -> np.ndarray:
    """
    Use Otsu's threshold to convert grayscale to binary.
    """
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def denoise_image(binary_image: np.ndarray) -> np.ndarray:
    """
    Use morphological opening to remove small noise from a binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

def convert_pdf_if_needed(upload_file: UploadFile, output_folder="converted_images") -> str:
    """
    If the file is a PDF, convert the first page to PNG using PyMuPDF.
    Otherwise, save the file as is. Return the final image path.
    """
    os.makedirs(output_folder, exist_ok=True)
    suffix = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(upload_file.file.read())
        temp_path = tmp_file.name
    
    if suffix.lower() == ".pdf":
        doc = fitz.open(temp_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        output_path = os.path.join(output_folder, "converted_page_1.png")
        pix.save(output_path)
        doc.close()
        return output_path
    else:
        return temp_path

def preprocess_image(input_path: str) -> str:
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError("Failed to load the image. Possibly invalid file.")
    
    gray = to_grayscale(original)
    enhanced = enhance_contrast(gray)
    thresh = threshold_image(enhanced)
    denoised = denoise_image(thresh)
    
    final_path = input_path + "_processed.png"
    cv2.imwrite(final_path, denoised)
    return final_path

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image to base64 (for passing to GPT).
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# EXAMPLE_IMAGE_1 = "/home/shahriar/Work/AbanTether/cluad/qwen/img3.jpeg"
# EXAMPLE_IMAGE_2 = "/home/shahriar/Work/AbanTether/cluad/qwen/img4.jpg"
# EXAMPLE_IMAGE_3 = "/home/shahriar/Work/AbanTether/cluad/qwen/img2.jpeg"

# example_receipt_1_base64 = encode_image_to_base64(EXAMPLE_IMAGE_1)
# example_receipt_2_base64 = encode_image_to_base64(EXAMPLE_IMAGE_2)
# example_receipt_3_base64 = encode_image_to_base64(EXAMPLE_IMAGE_3)

import os

BASE_DIR = os.path.dirname(__file__)  # Directory where main.py resides
IMAGE_DIR = os.path.join(BASE_DIR, "images")

EXAMPLE_IMAGE_1 = os.path.join(IMAGE_DIR, "img3.jpeg")
EXAMPLE_IMAGE_2 = os.path.join(IMAGE_DIR, "img4.jpg")
EXAMPLE_IMAGE_3 = os.path.join(IMAGE_DIR, "img2.jpeg")


example_receipt_1_base64 = encode_image_to_base64(EXAMPLE_IMAGE_1)
example_receipt_2_base64 = encode_image_to_base64(EXAMPLE_IMAGE_2)
example_receipt_3_base64 = encode_image_to_base64(EXAMPLE_IMAGE_3)


def extract_transaction_data(final_image_path: str) -> str:

    base64_image = encode_image_to_base64(final_image_path)
    
    response = client.chat.completions.create(
        # model="gpt-4o",
        # model='o1',
        model="gpt-4.5-preview",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant who works in an Iranian bank as a professional banker. "
                    "You can read Persian text from images and extract it accurately. "
                    "We will provide example images with the correct extracted text, then you'll get a new image. "
                    "Extract these details:\n"
                    "- Date and Time\n"
                    "- Amount\n"
                    "- Merchant/Recipient\n"
                    "- Transaction Type\n"
                    "- Reference Number\n"
                    "Return a clear textual format (not JSON)."
                ),
            },
            # Example 1
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is **Example 1**. I will provide the image plus the correct text:\n\n"
                            "**Image** (below)\n\n"
                            "Correct Extraction:\n"
                            "نوع انتقال: انتقال ساتنا و پایا\n"
                            "شماره شبا برداشت: IR۶۵۰۶۰۰۵۲۰۶۰۱۰۰۰۹۳۳۰۵۱۰۰۱\n"
                            "مبلغ: ۳,۱۲۰,۰۰۰,۰۰۰ ریال\n"
                            "تاریخ: ۱۴۰۳/۱۱/۱\n"
                            "نام و نام خانوادگی: امین ساعتیان الکام توسعه اماد\n"
                            "بابت: پرداخت قرض و تأدیه دیون\n"
                            "نام بانک: بانک قرض الحسنه مهر ایران\n"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_receipt_1_base64}",
                            "detail": "high",
                        }
                    }
                ]
            },
            # Example 2
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is **Example 2**. I will provide the image plus the correct text:\n\n"
                            "**Image** (below)\n\n"
                            "Correct Extraction:\n"
                            "نوع انتقال: انتقال پول اینترنت بانک\n"
                            "شماره شبا برداشت: IR۸۹۰۵۶۰۹۵۰۱۷۱۰۰۲۹۰۰۶۶۵۰۰\n"
                            "مبلغ: ۵۰۰,۰۰۰,۰۰۰ ریال\n"
                            "تاریخ: ۱۴۰۳/۱۰/۲۷\n"
                            "نام و نام خانوادگی: الکام توسعه اماد\n"
                            "بابت: وثیقه\n"
                            "نام بانک: مشخص نیست\n"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_receipt_2_base64}",
                            "detail": "high",
                        }
                    }
                ]
            },
            # Example 3
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is **Example 3**. I will provide the image plus the correct text:\n\n"
                            "**Image** (below)\n\n"
                            "Correct Extraction:\n"
                            "نوع انتقال: انتقال ساتنا\n"
                            "شماره شبا برداشت: IR۶۵۰۶۰۰۵۲۰۶۰۱۰۰۰۹۳۳۰۵۱۰۰۱\n"
                            "مبلغ : ۱,۱۰۰,۰۰۰,۰۰۰ ریال\n"
                            "تاریخ : مشخض نیست\n"
                            "نام و نام خانوادگی : الکام توسعه اماد\n"
                            "بابت: پرداخت قرض و تأدیه دیون\n"
                            "نام بانک: بانک قرض الحسنه مهر ایران\n"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_receipt_3_base64}",
                            "detail": "high",
                        }
                    }
                ]
            },
            # The new image
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Now process this **NEW receipt** image. "
                            "Please extract the Persian text in the same style. "
                            "But do NOT return JSON—just a structured text with the fields."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        }
                    }
                ]
            },
        ],
        max_tokens=600,
        temperature=0.2,
    )
    return response.choices[0].message.content


def parse_extracted_text(gpt_text: str) -> dict:
    """
    Extract fields from the raw Persian text returned by GPT.
    Adjust patterns as needed for your actual LLM output.
    """
    data = {
        "date_time": None,
        "amount": None,
        "merchant": None,
        "transaction_type": None,
        "reference_number": None
    }
    
    # Regex patterns: update to match your GPT output style
    patterns = {
        "transaction_type": r"نوع انتقال:\s*(.*)",
        "amount": r"مبلغ:\s*(.*)",
        "date_time": r"تاریخ:\s*(.*)",
        "paid by": r"نام و نام خانوادگی:\s*(.*)",
        # Could also match شناسه پرداخت, شماره تراکنش, مرجع, etc.
        "reference_number": r"(?:شناسه پرداخت|شماره تراکنش|مرجع):\s*(.*)"
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, gpt_text)
        if match:
            data[field] = match.group(1).strip()
    
    return data


def generate_base_filename(transaction_data: str) -> str:
    date_match = re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', transaction_data)
    if date_match:
        date_str = date_match.group().replace('/', '-')
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    timestamp = datetime.now().strftime('%H%M%S')
    return f"receipt_{date_str}_{timestamp}"

def save_files(transaction_data: str, image_path: str, output_dir="receipt_data"):
    """
    Saves the raw GPT text in a .txt file and copies the final image to the same folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = generate_base_filename(transaction_data)
    
    text_filepath = os.path.join(output_dir, f"{base_filename}.txt")
    image_extension = os.path.splitext(image_path)[1]
    image_filepath = os.path.join(output_dir, f"{base_filename}{image_extension}")
    
    with open(text_filepath, 'w') as f:
        f.write(transaction_data)
    
    shutil.copy2(image_path, image_filepath)
    return text_filepath, image_filepath

###############################################################################
# 7. FastAPI Endpoint
###############################################################################
@app.post("/analyze-receipt")
async def analyze_receipt(file: UploadFile = File(...)):
    """
    1) Receives an uploaded file (image or PDF).
    2) Converts PDF to PNG if needed.
    3) Preprocesses the image.
    4) Classifies to check if it's a valid transaction receipt.
    5) If valid, runs GPT-4O few-shot extraction (freeform text), parses the text,
       and returns only the parsed fields in extracted_data.
    """
    try:
        # Step A: Convert PDF if needed
        converted_path = convert_pdf_if_needed(file)
        
        # Step B: Preprocess
        final_image_path = preprocess_image(converted_path)
        
        # Step C: Zero-shot classification
        scores = classifier(final_image_path, candidate_labels=labels)
        top_label = scores[0]['label']
        top_score = scores[0]['score']
        
        if top_label != "Transaction receipt" or top_score < CONFIDENCE_THRESHOLD:
            raise HTTPException(status_code=400, detail="Not a valid transaction receipt.")
        
        # Step D: GPT extraction => raw text
        gpt_raw_text = extract_transaction_data(final_image_path)
        
        # Step E: Parse fields into a dictionary
        extracted_details = parse_extracted_text(gpt_raw_text)
        
        # Step F: Save raw text + image
        text_filepath, image_filepath = save_files(gpt_raw_text, final_image_path)
        
        # Step G: Return only the parsed fields
        return {
            "status": "success",
            "classification_score": top_score,
            "extracted_data": extracted_details,
            # "saved_text_path": text_filepath,
            # "saved_image_path": image_filepath
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optionally run with:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
