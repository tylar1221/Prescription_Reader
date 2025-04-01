from __future__ import annotations
import os
import shutil
import json
import dotenv
import easyocr
import streamlit as st
import pandas as pd
import google.generativeai as genai
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError

# Load environment variables from .env file
dotenv.load_dotenv()

# Retrieve API key securely with error handling
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {str(e)}")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(layout="wide")

# Apply custom CSS for better UI
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styling.")

# Caching EasyOCR model
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'])

# Define Medication Schema with optional fields and default values
class MedicationItem(BaseModel):
    name: str = Field(description="Medication name")
    dosage: str = Field(description="Medication dosage", default="Not specified")
    frequency: str = Field(description="How often to take the medication", default="Not specified")
    duration: str = Field(description="How long to take the medication", default="Not specified")
    
    class Config:
        extra = "ignore"  # Ignore additional fields

# Define Prescription Schema
class PrescriptionInformation(BaseModel):
    patient_name: str = Field(description="Patient's name", default="Unknown")
    patient_age: int = Field(description="Patient's age", default=0)
    patient_gender: str = Field(description="Patient's gender", default="Unknown")
    doctor_name: str = Field(description="Doctor's name", default="Unknown")
    doctor_license: str = Field(description="Doctor's license number", default="Unknown")
    prescription_date: str = Field(description="Date of the prescription (YYYY-MM-DD)", default="")
    medications: List[MedicationItem] = Field(default_factory=list)
    additional_notes: str = Field(description="Additional notes or instructions", default="")
    
    class Config:
        extra = "ignore"  # Ignore additional fields

# Extract text using EasyOCR
def extract_text_easyocr(image_path: str) -> str:
    try:
        reader = load_easyocr()  # Cached reader
        result = reader.readtext(image_path, detail=0)
        return " ".join(result) if result else "Text not readable"
    except Exception as e:
        st.error(f"Error extracting text with EasyOCR: {str(e)}")
        return "Error during text extraction"

# Gemini AI - Extract structured prescription details
def parse_prescription_with_gemini(extracted_text: str) -> Dict[str, Any]:
    if not extracted_text or extracted_text == "Text not readable" or extracted_text == "Error during text extraction":
        return {"error": "No readable text was extracted from the image"}
    
    # Use a more capable model for complex parsing
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        return {"error": f"Failed to load Gemini model: {str(e)}"}
    
    prompt = f"""
    You are an expert medical transcriptionist. Analyze the given prescription text and extract structured details in JSON format.

    **Required Fields:**
    1. Patient's Full Name (use best guess if unclear)
    2. Patient's Age (in years as a number, e.g., 42, use 0 if unclear)
    3. Patient's Gender (Male/Female, use Unknown if unclear)
    4. Doctor's Full Name (use best guess if unclear)
    5. Doctor's License Number (use best guess if unclear)
    6. Prescription Date (Format: YYYY-MM-DD, use today's date if unclear)
    7. Medications List (Each item must include: Name, Dosage, Frequency, Duration)
       - For any medication where you can't determine dosage, frequency, or duration, use "Not specified"
       - Never use null or empty values for these fields
    8. Additional Notes (if present, otherwise empty string)

    **Prescription Text:**
    {extracted_text}

    **Output Format (JSON Example):**
    {{
        "patient_name": "John Doe",
        "patient_age": 45,
        "patient_gender": "Male",
        "doctor_name": "Dr. Jane Smith",
        "doctor_license": "ABC123456",
        "prescription_date": "2023-04-01",
        "medications": [
            {{
                "name": "Amoxicillin",
                "dosage": "500 mg",
                "frequency": "Twice a day",
                "duration": "7 days"
            }},
            {{
                "name": "Ibuprofen",
                "dosage": "200 mg",
                "frequency": "Every 4 hours as needed",
                "duration": "5 days"
            }}
        ],
        "additional_notes": "Take medications with food. Drink plenty of water."
    }}

    Return ONLY valid JSON and nothing else. Make reasonable assumptions for missing information.
    Never return null values - use default text like "Not specified" or 0 for numeric fields instead.
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Clean up response to ensure it's valid JSON
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        structured_data = json.loads(response_text)
        return structured_data
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response from Gemini: {str(e)}\nResponse was: {response_text[:200]}..."}
    except Exception as e:
        return {"error": f"Gemini API Error: {str(e)}"}

# Fix missing or null fields in medication items
def sanitize_medications(medications: List[Dict]) -> List[Dict]:
    sanitized = []
    for med in medications:
        # Create a copy with all required fields having default values
        sanitized_med = {
            "name": med.get("name", "Unknown Medication"),
            "dosage": med.get("dosage") or "Not specified",
            "frequency": med.get("frequency") or "Not specified",
            "duration": med.get("duration") or "Not specified"
        }
        sanitized.append(sanitized_med)
    return sanitized

# Process Prescription Image and validate with Pydantic
def get_prescription_information(image_path: str) -> Dict[str, Any]:
    extracted_text = extract_text_easyocr(image_path)
    structured_data = parse_prescription_with_gemini(extracted_text)
    
    if "error" in structured_data:
        return structured_data
    
    # Sanitize data before validation
    if "medications" in structured_data and isinstance(structured_data["medications"], list):
        structured_data["medications"] = sanitize_medications(structured_data["medications"])
    
    # Ensure prescription_date is set
    if not structured_data.get("prescription_date"):
        structured_data["prescription_date"] = datetime.now().strftime("%Y-%m-%d")
    
    # Validate data against Pydantic model
    try:
        # Validate entire prescription
        validated_data = PrescriptionInformation(**structured_data)
        return validated_data.dict()
    except ValidationError as e:
        return {"error": f"Data validation error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error during validation: {str(e)}"}

# Create a temporary directory to store uploaded files
def create_temp_folder() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(".", f"temp_prescription_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

# Clean up temporary folder
def remove_temp_folder(path: str):
    try:
        if os.path.exists(path):
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    except Exception as e:
        st.warning(f"Failed to remove temporary files: {str(e)}")

# Format date for display
def format_date(date_str: str) -> str:
    try:
        if not date_str:
            return "Not specified"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%B %d, %Y")
    except:
        return date_str or "Not specified"

# Streamlit UI
def main():
    st.title('🩺 AI-Powered Prescription Parser (EasyOCR & Gemini)')
    
    # Load CSS if exists
    try:
        local_css("styles.css")
    except:
        pass
    
    uploaded_file = st.file_uploader("📂 Upload a Prescription Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Create temp folder
        temp_folder = create_temp_folder()
        
        # Save uploaded file
        image_path = os.path.join(temp_folder, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Show uploaded image
        with st.expander("📷 View Uploaded Prescription", expanded=False):
            st.image(uploaded_file, caption='Uploaded Prescription Image', use_column_width=True)

        # Process prescription
        with st.spinner('🔍 Extracting & Analyzing Prescription...'):
            final_result = get_prescription_information(image_path)

            if "error" in final_result:
                st.error(final_result["error"])
                st.session_state['temp_folder'] = temp_folder  # Store for delayed cleanup
            else:
                # Display patient and doctor details
                st.success("✅ Prescription analyzed successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("👤 Patient Information")
                    st.write(f"**Name:** {final_result['patient_name']}")
                    st.write(f"**Age:** {final_result['patient_age'] if final_result['patient_age'] > 0 else 'Not specified'} {'' if final_result['patient_age'] == 0 else 'years'}")
                    st.write(f"**Gender:** {final_result['patient_gender']}")
                
                with col2:
                    st.subheader("👨‍⚕️ Doctor Information")
                    st.write(f"**Name:** {final_result['doctor_name']}")
                    st.write(f"**License:** {final_result['doctor_license']}")
                    st.write(f"**Date:** {format_date(final_result['prescription_date'])}")

                # Display medications
                if "medications" in final_result and final_result["medications"]:
                    st.subheader("💊 Medications")
                    medications_df = pd.DataFrame(final_result["medications"])
                    st.table(medications_df)
                else:
                    st.warning("No medications were identified in the prescription.")

                # Display additional notes
                if "additional_notes" in final_result and final_result["additional_notes"]:
                    st.subheader("📝 Additional Notes")
                    st.info(final_result["additional_notes"])
                
                # Add download options
                st.subheader("📥 Download Options")
                
                # Convert to various formats
                json_data = json.dumps(final_result, indent=4)
                csv_data = pd.DataFrame(final_result["medications"]).to_csv(index=False) if "medications" in final_result and final_result["medications"] else "No medications found"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name="prescription_data.json",
                        mime="application/json"
                    )
                with col2:
                    st.download_button(
                        label="Download Medications CSV",
                        data=csv_data,
                        file_name="medications.csv",
                        mime="text/csv"
                    )
                
                st.session_state['temp_folder'] = temp_folder  # Store for delayed cleanup

    # Cleanup temp files when session ends or new file is uploaded
    if 'temp_folder' in st.session_state and st.session_state['temp_folder']:
        if st.button("Clear Data & Upload New Prescription"):
            remove_temp_folder(st.session_state['temp_folder'])
            st.session_state['temp_folder'] = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()
