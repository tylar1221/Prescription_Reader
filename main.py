import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import pandas as pd

def preprocess_image(image: np.array) -> np.array:
    """Preprocess the image for better OCR results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text_easyocr(image: np.array) -> str:
    """Extract text from image using EasyOCR."""
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image, detail=0)
    return " ".join(result)

def clean_text(text: str) -> str:
    """Clean the extracted text to fix OCR mistakes."""
    # Replace common OCR errors (e.g., `I` with `1`, `O` with `0`, and similar)
    text = text.replace('I', '1').replace('O', '0').replace('~', '0').replace('|', 'l')
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()
    return text

def analyze_prescription_with_gemini(text: str) -> str:
    """Use Google Gemini API to extract structured details from the prescription."""
    load_dotenv()  # Load environment variables from .env file
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    prompt = (
    "You are a specialized AI for processing medical prescriptions, including those with Indian names. "
    "Analyze the provided text and extract the following details with high accuracy:\n\n"
    "Text: " + text + "\n\n"
    "Provide the output in the following structured format:\n"
    "- Patient Name: \n"
    "- Age: \n"
    "- Gender: \n"
    "- Medicines (list each medicine with detailed information):\n"
    "  1. Medicine Name: \n"
    "     - Dosage: (e.g., '500mg', '250mg'; if unclear, leave this blank)\n"
    "     - Frequency: (interpret patterns like '1 1 1' = '3 times a day', '1 0 1' = 'twice a day: morning and night'; if unclear, leave this blank)\n"
    "     - Timing: (e.g., 'before meals', 'after meals'; if not clearly stated, leave this blank)\n"
    "     - Description: (if dosage/timing is unclear, provide a brief general description of the medicine if possible)\n"
    "\n- Additional Notes: \n\n"
    "Important Guidelines:\n"
    "- Do not guess or infer any information that is not clearly legible.\n"
    "- Do not make assumptions about missing information.\n"
    "- Pay close attention to details like medication names, dosages, and frequencies.\n"
    "- If portions of the text are not clear, leave the corresponding fields empty.\n"
)

    response = model.generate_content(prompt)
    return response.text if response else "Failed to process prescription."


def main():
    st.title("📄 Prescription Reader using EasyOCR & Gemini AI")
    st.write("Upload a prescription image, and this app will extract and structure the details.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Prescription", use_container_width=True)

        with st.spinner("Extracting text..."):
            extracted_text = extract_text_easyocr(image)

        st.subheader("📄 Extracted Text:")
        st.text_area("", extracted_text, height=200)

        # Clean extracted text to fix common OCR mistakes
        cleaned_text = clean_text(extracted_text)

        with st.spinner("Analyzing prescription using Gemini..."):
            structured_info = analyze_prescription_with_gemini(cleaned_text)

        st.subheader("📑 Structured Prescription Data:")
        st.text_area("", structured_info, height=200)

        # Extract structured data (adjust based on Gemini's output format)
        # For the example, using mock data, you should replace it with actual parsed data from Gemini's response
        structured_data = {
            "Patient Name": "Not mentioned in the text",  # Replace with actual extracted data
            "Age": "28",  # Replace with actual extracted data
            "Gender": "Male",  # Replace with actual extracted data
            "Medicines": "Augmentin 625mg, Hexigel Mouthwash",  # Replace with actual extracted data
           
            "Additional Notes": "None",  # Replace with actual extracted data
        }

        # Save the structured data to an Excel file

if __name__ == "__main__":
    main()
