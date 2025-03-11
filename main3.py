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


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image to base64 (for passing to GPT).
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



import os

BASE_DIR = os.path.dirname(__file__)  # Directory where main.py resides
IMAGE_DIR = os.path.join(BASE_DIR, "images")

EXAMPLE_IMAGE_1 = os.path.join(IMAGE_DIR, "img3.jpeg")
EXAMPLE_IMAGE_2 = os.path.join(IMAGE_DIR, "img4.jpg")
EXAMPLE_IMAGE_3 = os.path.join(IMAGE_DIR, "img2.jpeg")


example_receipt_1_base64 = encode_image_to_base64(EXAMPLE_IMAGE_1)
example_receipt_2_base64 = encode_image_to_base64(EXAMPLE_IMAGE_2)
example_receipt_3_base64 = encode_image_to_base64(EXAMPLE_IMAGE_3)


def generate_base_filename(transaction_data):
    # Try to extract date from transaction data
    try:
        date_match = re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', transaction_data)
        if date_match:
            date_str = date_match.group().replace('/', '-')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
    except (TypeError, AttributeError):

        date_str = datetime.now().strftime('%Y-%m-%d')

    timestamp = datetime.now().strftime('%H%M%S')
    
    return f"receipt_{date_str}_{timestamp}"


def extract_transaction_data(image_path):
    base64_image = encode_image_to_base64(image_path)
    
    response = anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=[
            {
                "role": "assistant",
                "content": (
                    '''You are a helpful assistant who work in Iran's bank as professional banker that can read Persian text from images and extract and OCR it accurately. "
                    "We will provide example images with the correct extracted text. Then you'll get a new image "
                    and should provide the extracted text in Persian, Extract the following transaction details:
                    - Date and Time
                    - Amount
                    - Merchant/Recipient
                    - Transaction Type
                    - Reference Number (if any)
                    Return in a clear, structured format.'''            
                ),
            },
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
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example_receipt_1_base64,
                        },
                    },
                ]
            },
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
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example_receipt_2_base64,
                        },
                    },
                ]
            },
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
                            "تاریخ : مشخض نیست \n"
                            "نام و نام خانوادگی : الکام توسعه اماد\n"
                            "بابت: پرداخت قرض و تأدیه دیون\n"
                            "نام بانک: بانک قرض الحسنه مهر ایران\n"
                        ),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example_receipt_3_base64,
                        },
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Now process this **NEW receipt** image. "
                            "Please extract the Persian text in the same style."
                        )
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ]
            },
        ]
    )

    try:
        # Anthropic response structure has content that contains the actual text
        return response.content[0].text
    except (AttributeError, IndexError, TypeError) as e:
        print(f"Error extracting content from response: {e}")
        print(f"Response structure: {response}")
        # Return a fallback value
        return "Error extracting transaction data"


# def parse_extracted_text(gpt_text: str) -> dict:
#     """
#     Extract fields from the raw Persian text returned by GPT.
#     Adjust patterns as needed for your actual LLM output.
#     """
#     data = {
#         "date_time": None,
#         "amount": None,
#         "merchant": None,
#         "transaction_type": None,
#         "reference_number": None
#     }
    
#     # Regex patterns: update to match your GPT output style
#     patterns = {
#         "transaction_type": r"نوع انتقال:\s*(.*)",
#         "amount": r"مبلغ:\s*(.*)",
#         "date_time": r"تاریخ:\s*(.*)",
#         "paid by": r"نام و نام خانوادگی:\s*(.*)",
#         # Could also match شناسه پرداخت, شماره تراکنش, مرجع, etc.
#         "reference_number": r"(?:شناسه پرداخت|شماره تراکنش|مرجع):\s*(.*)"
#     }
    
#     for field, pattern in patterns.items():
#         match = re.search(pattern, gpt_text)
#         if match:
#             data[field] = match.group(1).strip()
    
#     return data


def parse_extracted_text(gpt_text: str) -> dict:

    data = {
        "date_time": None,
        "amount": None,
        "merchant": None,
        # "transaction_type": None,
        # "reference_number": None,
        # "purpose": None,
        # "fee": None,
        # "total_amount": None
    }
    
    # Try to find the markdown block with transaction details.
    markdown_start = gpt_text.find("**Transaction Details:**")
    if markdown_start != -1:
        markdown_block = gpt_text[markdown_start:]
        # This regex looks for lines like:
        # - **Field Name:** value
        matches = re.findall(r'^\s*-\s*\*\*(.+?)\*\*:\s*(.+)$', markdown_block, re.MULTILINE)
        # Map the field names from GPT to our dictionary keys
        mapping = {
            "Date and Time": "date_time",
            "Amount": "amount",
            "Merchant/Recipient": "merchant",
            # "Transaction Type": "transaction_type",
            # "Reference Number": "reference_number",
            # "Purpose": "purpose",
            # "Fee": "fee",
            # "Total Amount": "total_amount"
        }
        for field, value in matches:
            field = field.strip()
            value = value.strip()
            if field in mapping:
                data[mapping[field]] = value
    else:
        # Fallback: Use regex patterns on the raw text if markdown block is missing
        patterns = {
            "transaction_type": r"نوع انتقال\s*:\s*(.+)",
            "amount": r"مبلغ(?: اصل حواله)?\s*:\s*(.+)",
            "date_time": r"(?:تاریخ(?: صدور| اجرا)?|تاریخ)\s*:\s*(.+)",
            "merchant": r"(?:نام صاحب حساب|نام و نام خانوادگی)\s*:\s*(.+)",
            # "reference_number": r"(?:کد رهگیری|شناسه گیرنده|شماره حساب گیرنده)\s*:\s*(.+)",
            # "purpose": r"بابت\s*:\s*(.+)",
            # "fee": r"مبلغ کارمزد حواله\s*:\s*(.+)",
            # "total_amount": r"کل مبلغ سود/اضافی از حساب\s*:\s*(.+)"
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
        
        # Step C: Zero-shot classification
        scores = classifier(converted_path, candidate_labels=labels)
        top_label = scores[0]['label']
        top_score = scores[0]['score']
        
        if top_label != "Transaction receipt" or top_score < CONFIDENCE_THRESHOLD:
            raise HTTPException(status_code=400, detail="Not a valid transaction receipt.")
        
        # Step D: GPT extraction => raw text
        gpt_raw_text = extract_transaction_data(converted_path)
        # gpt_raw_text = extract_transaction_data(converted_path)
        print("Raw GPT Output:\n", gpt_raw_text)
        
        # Step E: Parse fields into a dictionary
        extracted_details = parse_extracted_text(gpt_raw_text)
        
        
        # Step F: Save raw text + image
        text_filepath, image_filepath = save_files(gpt_raw_text, converted_path)
        
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