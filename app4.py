
import streamlit as st
import re
import cv2
import numpy as np
import easyocr
import pandas as pd
from rapidfuzz import fuzz
from dateutil import parser as dateparser
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import tempfile
import os
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Dhruv OCR",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: white;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .success-badge {
        background-color: #10b981;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .failed-badge {
        background-color: #ef4444;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    h1 {
        color: #1e293b;
        font-weight: 700;
    }
    h2 {
        color: #334155;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    h3 {
        color: #475569;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    h4 {
        color: #64748b;
        font-weight: 600;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    /* Make markdown headers more visible */
    [data-testid="stMarkdownContainer"] h3 {
        color: #1e40af !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    [data-testid="stMarkdownContainer"] h4 {
        color: #3730a3 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    </style>
    """, unsafe_allow_html=True)

# Configuration
class Config:
    OCR_LANGS = ['en']
    FUZZY_MATCH_THRESHOLD_MATCH = 85
    FUZZY_MATCH_THRESHOLD_PARTIAL = 60
    RESIZE_MAX = 2000
    GPU_ENABLED = False
    DEBUG_MODE = False
    
    AADHAAR_PATTERN = re.compile(r'(\d{4}\s*\d{4}\s*\d{4})')
    PAN_PATTERN = re.compile(r'\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b')
    DOB_PATTERN = re.compile(
        r'(\b\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}\b)|'
        r'(\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}\b)',
        flags=re.IGNORECASE
    )
    MOBILE_PATTERN = re.compile(r'\b(?:mobile\s*(?:no\.?)?[:\s]*)?(\d{10})\b', flags=re.IGNORECASE)
    VID_PATTERN = re.compile(r'\b(?:VID\s*[:\s])?(\d{4}\s\d{4}\s*\d{4}\s*\d{4})\b', flags=re.IGNORECASE)
    
    GENDER_KEYWORDS = {
        "Male": ["male", "m", "पुरुष", "पुरुष / MALE", "पुरुष/ MALE"],
        "Female": ["female", "f", "महिला", "महिला / FEMALE", "महिला/female"]
    }
    
    FILTER_KEYWORDS = [
        'government', 'india', 'aadhaar', 'aadhar', 'unique', 'identification',
        'authority', 'male', 'female', 'dob', 'birth', 'mobile', 'address',
        'download', 'date', 'vid', 'help', 'email', 'phone', 'issued', 'issue',
        'enrollment', 'permanent', 'account', 'income', 'tax', 'department',
        'bharath', 'bharat', 'sarkar', 'sarkaar', 'card', 'number', 'year',
        'www', 'http', 'uidai', 'gov', 'box', 'bengaluru', 'bangalore', 'delhi',
        'mumbai', 'proof', 'identity', 'citizen', 'online', 'update', 'biometric',
        'authentication', 'resident', 'pradhan', 'mantri', 'yojana', 'scheme',
        'registration', 'verify', 'verification', 'service', 'center', 'name',
        'gender', 'document', 'type', 'issuer', 'vendor', 'report'
    ]
    
    NAME_CONTEXT_KEYWORDS = [
        'name', 'नाम', 'naam', 'holder', 'cardholder', 's/o', 'd/o', 'w/o',
        'son of', 'daughter of', 'wife of', 'father', 'mother', 'spouse'
    ]

# Initialize OCR Reader with caching
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(Config.OCR_LANGS, gpu=Config.GPU_ENABLED)

reader = get_ocr_reader()

# All the core classes from your original code
class ImageProcessor:
    @staticmethod
    def preprocess(img_array):
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        h, w = gray.shape[:2]
        target_width = 2000
        scale = target_width / w
        if scale > 1:
            scale = min(scale, 2.0)
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        elif scale < 1:
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrasted = clahe.apply(denoised)
        kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrasted, -1, kernel_sharpen)
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        return dilated

class OCRExtractor:
    @staticmethod
    def extract_text(img_array) -> str:
        try:
            results_combined = []
            img = ImageProcessor.preprocess(img_array)
            results1 = reader.readtext(img, detail=1, paragraph=False)
            
            if img_array is not None:
                results2 = reader.readtext(img_array, detail=1, paragraph=False)
                for r in results2:
                    bbox, text, conf = r
                    if conf > 0.3:
                        results_combined.append(r)
            
            for r in results1:
                bbox, text, conf = r
                if conf > 0.2:
                    results_combined.append(r)
            
            seen_texts = set()
            unique_results = []
            for r in results_combined:
                text = r[1].strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    unique_results.append(r)
            
            results_sorted = sorted(unique_results, key=lambda x: x[0][0][1])
            text_lines = [res[1].strip() for res in results_sorted if res[1].strip()]
            full_text = "\n".join(text_lines)
            return full_text
        except Exception as e:
            st.error(f"OCR extraction failed: {e}")
            return ""

class FieldExtractor:
    @staticmethod
    def extract_aadhaar(text: str) -> str:
        text_cleaned = re.sub(r'VID\s*[:\s]*\d+', '', text, flags=re.IGNORECASE)
        text_cleaned = re.sub(r'mobile\s*(?:no\.?)?[:\s]*\d+', '', text_cleaned, flags=re.IGNORECASE)
        matches = Config.AADHAAR_PATTERN.findall(text_cleaned.replace('-', ' '))
        for match in matches:
            digits = re.sub(r'\D', '', match)
            if len(digits) == 12 and digits[0] not in ['0', '1']:
                return " ".join([digits[:4], digits[4:8], digits[8:12]])
        return ""
    
    @staticmethod
    def extract_pan(text: str) -> str:
        match = Config.PAN_PATTERN.search(text)
        return match.group(1) if match else ""
    
    @staticmethod
    def extract_dob(text: str) -> str:
        match = Config.DOB_PATTERN.search(text)
        if match:
            candidate = match.group(0)
            try:
                dt = dateparser.parse(candidate, dayfirst=True, fuzzy=True)
                return dt.strftime("%d/%m/%Y")
            except Exception:
                return candidate.strip()
        return ""
    
    @staticmethod
    def extract_gender(text: str) -> str:
        text_normalized = re.sub(r'\s+', ' ', text.lower())
        for gender, keywords in Config.GENDER_KEYWORDS.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_normalized):
                    return gender
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if Config.DOB_PATTERN.search(line):
                search_text = line
                if i + 1 < len(lines):
                    search_text += ' ' + lines[i + 1]
                if re.search(r'\b[mM]\b(?!\w)', search_text):
                    return "Male"
                elif re.search(r'\b[fF]\b(?!\w)', search_text):
                    return "Female"
        return ""
    
    @staticmethod
    def extract_mobile(text: str) -> str:
        match = Config.MOBILE_PATTERN.search(text)
        if match:
            mobile = re.sub(r'\D', '', match.group(0))
            if len(mobile) == 10 and mobile[0] in ['6', '7', '8', '9']:
                return mobile
        return ""
    
    @staticmethod
    def extract_vid(text: str) -> str:
        match = Config.VID_PATTERN.search(text)
        if match:
            digits = re.sub(r'\D', '', match.group(0))
            if len(digits) == 16:
                return " ".join([digits[:4], digits[4:8], digits[8:12], digits[12:16]])
        return ""
    
    @staticmethod
    def clean_ocr_text(text_input: str) -> str:
        if not text_input:
            return ""
        cleaned = re.sub(r'[|_\{\}\[\]<>\\\/\*\+\=\~\`\^\$\#\@\!]', '', text_input)
        cleaned = re.sub(r'\b[^a-zA-Z\s]\b', '', cleaned)
        cleaned = ' '.join(cleaned.split())
        cleaned = re.sub(r'\.{2,}', '', cleaned)
        return cleaned.strip()
    
    @staticmethod
    def is_valid_name_word(word: str) -> bool:
        if not word or len(word) < 2 or len(word) > 20:
            return False
        alpha_count = sum(c.isalpha() for c in word)
        if alpha_count < len(word) * 0.75:
            return False
        if word.lower() in Config.FILTER_KEYWORDS:
            return False
        if any(c.isdigit() for c in word):
            return False
        return True
    
    @staticmethod
    def is_valid_name(name_text: str) -> bool:
        if not name_text or len(name_text) < 3:
            return False
        cleaned = FieldExtractor.clean_ocr_text(name_text)
        if not cleaned:
            return False
        if cleaned.lower().strip() in ['name', 'नाम', 'naam', 'gender', 'date', 'birth', 'dob', 'aadhaar', 'document']:
            return False
        words = cleaned.split()
        if not (1 <= len(words) <= 5):
            return False
        valid_words = [w for w in words if FieldExtractor.is_valid_name_word(w)]
        if len(valid_words) < len(words) * 0.8:
            return False
        if not (3 <= len(cleaned) <= 50):
            return False
        for keyword in Config.FILTER_KEYWORDS:
            if keyword in cleaned.lower():
                return False
        if sum(c.isdigit() for c in cleaned) > 0:
            return False
        alpha_count = sum(c.isalpha() for c in cleaned)
        if alpha_count < len(cleaned.replace(' ', '')) * 0.85:
            return False
        return True
    
    @staticmethod
    def extract_name_with_context(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if line_lower in ['name', 'नाम', 'naam', 'name:', 'name :']:
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    cleaned = FieldExtractor.clean_ocr_text(next_line)
                    if FieldExtractor.is_valid_name(cleaned):
                        return cleaned
            
            pattern1 = r'(?:name|नाम|naam)\s*[:\-]\s*([A-Za-z\s]+?)(?:\s*$|\s*\n|DOB|Date|\d|Gender|जन्म)'
            name_label_match = re.search(pattern1, line, flags=re.IGNORECASE)
            if name_label_match:
                candidate = name_label_match.group(1).strip()
                cleaned = FieldExtractor.clean_ocr_text(candidate)
                if FieldExtractor.is_valid_name(cleaned):
                    return cleaned
            
            for keyword in Config.NAME_CONTEXT_KEYWORDS:
                if keyword in line_lower:
                    pattern2 = r'(?:name|नाम|naam|s/o|d/o|w/o)[:\s]*'
                    parts = re.split(pattern2, line, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        candidate = parts[1].strip()
                        pattern3 = r'\s+(?:dob|date|birth|gender|male|female|\d)'
                        candidate = re.split(pattern3, candidate, flags=re.IGNORECASE)[0]
                        cleaned = FieldExtractor.clean_ocr_text(candidate)
                        if FieldExtractor.is_valid_name(cleaned):
                            return cleaned
        return ""
    
    @staticmethod
    def extract_name_near_hindi(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i in range(len(lines) - 1):
            current_line = lines[i]
            next_line = lines[i + 1]
            has_hindi = bool(re.search(r'[\u0900-\u097F]', current_line))
            if has_hindi:
                has_mostly_english = bool(re.search(r'^[A-Za-z\s]+$', next_line))
                if has_mostly_english:
                    cleaned = FieldExtractor.clean_ocr_text(next_line)
                    if FieldExtractor.is_valid_name(cleaned):
                        return cleaned
        return ""
    
    @staticmethod
    def extract_name_near_dob(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pattern1 = r'(\b\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}\b)|(\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}\b)|(?:dob|birth|जन्म|तिथि)'
        dob_pattern = re.compile(pattern1, flags=re.IGNORECASE)
        dob_indices = [i for i, line in enumerate(lines) if dob_pattern.search(line)]
        
        for dob_idx in dob_indices:
            for offset in range(1, 5):
                check_idx = dob_idx - offset
                if 0 <= check_idx < len(lines):
                    candidate_line = lines[check_idx]
                    if candidate_line.lower().strip() in ['name', 'dob', 'date of birth', 'gender']:
                        continue
                    if re.search(r'\d{4,}', candidate_line):
                        continue
                    cleaned = FieldExtractor.clean_ocr_text(candidate_line)
                    if FieldExtractor.is_valid_name(cleaned):
                        return cleaned
        return ""
    
    @staticmethod
    def extract_name_title_case(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, line in enumerate(lines[:20]):
            if len(re.findall(r'\d', line)) > 4:
                continue
            title_case_matches = re.findall(r'\b([A-Z][a-z]{2,19}(?:\s+[A-Z][a-z]{2,19}){0,4})\b', line)
            for match in title_case_matches:
                if FieldExtractor.is_valid_name(match):
                    return match
        return ""
    
    @staticmethod
    def extract_name_from_address_section(text: str) -> str:
        pattern = r'(?:S/O|D/O|W/O|s/o|d/o|w/o)\s+([A-Za-z\s]+?)(?:\s*,|\s*\n|\s+house|\s+\d)'
        relation_pattern = re.compile(pattern, flags=re.IGNORECASE)
        match = relation_pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            cleaned = FieldExtractor.clean_ocr_text(candidate)
            if FieldExtractor.is_valid_name(cleaned):
                return cleaned
        return ""
    
    @staticmethod
    def extract_name(text: str) -> str:
        candidates: List[Tuple[str, int, str]] = []
        
        name = FieldExtractor.extract_name_with_context(text)
        if name:
            candidates.append((name, 100, "context_label"))
        
        name = FieldExtractor.extract_name_near_hindi(text)
        if name:
            candidates.append((name, 95, "hindi_pattern"))
        
        name = FieldExtractor.extract_name_from_address_section(text)
        if name:
            candidates.append((name, 90, "address_relation"))
        
        name_dob = FieldExtractor.extract_name_near_dob(text)
        name_title = FieldExtractor.extract_name_title_case(text)
        
        if name_dob and name_title:
            similarity = fuzz.ratio(name_dob.lower(), name_title.lower())
            if similarity > 70:
                candidates.append((name_title, 88, "title_case"))
                candidates.append((name_dob, 85, "near_dob"))
            else:
                candidates.append((name_dob, 85, "near_dob"))
                candidates.append((name_title, 80, "title_case"))
        elif name_dob:
            candidates.append((name_dob, 85, "near_dob"))
        elif name_title:
            candidates.append((name_title, 80, "title_case"))
        
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, line in enumerate(lines[3:18]):
            if len(re.findall(r'\d', line)) > 4:
                continue
            cleaned = FieldExtractor.clean_ocr_text(line)
            if cleaned and re.match(r'^[A-Za-z\s]+$', cleaned):
                if FieldExtractor.is_valid_name(cleaned):
                    score = 70 - i
                    candidates.append((cleaned, score, f"alphabetic_line_{i+3}"))
                    break
        
        if candidates:
            unique_candidates: Dict[str, Tuple[str, int, str]] = {}
            for name, score, strategy in candidates:
                name_key = name.lower().replace(' ', '')
                if name_key not in unique_candidates or score > unique_candidates[name_key][1]:
                    unique_candidates[name_key] = (name, score, strategy)
            
            sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1], reverse=True)
            return sorted_candidates[0][0]
        return ""
    
    @staticmethod
    def detect_document_type(text: str) -> str:
        text_lower = text.lower()
        text_normalized = re.sub(r'\s+', ' ', text_lower)
        
        if any(keyword in text_normalized for keyword in ['aadhaar', 'aadhar', 'आधार', 'uidai', 'unique identification']):
            return "Aadhaar Card"
        if any(keyword in text_normalized for keyword in ['income tax', 'permanent account', 'pan card']):
            return "PAN Card"
        if 'passport' in text_normalized:
            return "Passport"
        if any(keyword in text_normalized for keyword in ['driving', 'license', 'licence', 'dl no']):
            return "Driving License"
        if any(keyword in text_normalized for keyword in ['voter', 'election', 'epic']):
            return "Voter ID"
        if Config.AADHAAR_PATTERN.search(text):
            return "Aadhaar Card"
        if Config.PAN_PATTERN.search(text):
            return "PAN Card"
        return "Unknown"

class DocumentParser:
    @staticmethod
    def parse(text: str) -> Dict[str, str]:
        doc_type = FieldExtractor.detect_document_type(text)
        fields = {
            "Document Type": doc_type,
            "Name": FieldExtractor.extract_name(text),
            "Date of Birth": FieldExtractor.extract_dob(text),
            "Gender": FieldExtractor.extract_gender(text),
        }
        
        if doc_type == "Aadhaar Card":
            aadhaar = FieldExtractor.extract_aadhaar(text)
            if aadhaar:
                fields["Aadhaar Number"] = aadhaar
            mobile = FieldExtractor.extract_mobile(text)
            if mobile:
                fields["Mobile Number"] = mobile
            vid = FieldExtractor.extract_vid(text)
            if vid:
                fields["VID"] = vid
        elif doc_type == "PAN Card":
            fields["PAN Number"] = FieldExtractor.extract_pan(text)
        
        fields = {k: v for k, v in fields.items() if v}
        return fields

class DocumentProcessor:
    @staticmethod
    def process_document(img_array) -> Dict[str, str]:
        ocr_text = OCRExtractor.extract_text(img_array)
        fields = DocumentParser.parse(ocr_text)
        fields['_raw_ocr'] = ocr_text
        fields['_processed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return fields

class ComparisonEngine:
    @staticmethod
    def compare(user_report: Dict[str, str], vendor_report: Dict[str, str], fields: Optional[List[str]] = None) -> pd.DataFrame:
        if fields is None:
            critical_fields = ["Document Type", "Name", "Date of Birth", "Gender", "Aadhaar Number", "PAN Number"]
            all_fields = set(user_report.keys()) | set(vendor_report.keys())
            fields = [f for f in critical_fields if f in all_fields]
            for field in all_fields:
                if not field.startswith('_') and field not in fields:
                    fields.append(field)
        
        rows = []
        for field in fields:
            user_val = (user_report.get(field) or "").strip()
            vendor_val = (vendor_report.get(field) or "").strip()
            
            if not user_val and not vendor_val:
                continue
            elif not user_val or not vendor_val:
                status = "Missing in " + ("User" if not user_val else "Vendor")
                score = 0
            else:
                score = fuzz.token_set_ratio(user_val, vendor_val)
                if field == "Document Type":
                    status = "Match" if user_val == vendor_val else "Mismatch"
                elif score >= Config.FUZZY_MATCH_THRESHOLD_MATCH:
                    status = "Match"
                elif score >= Config.FUZZY_MATCH_THRESHOLD_PARTIAL:
                    status = "Partial Match"
                else:
                    status = "Mismatch"
            
            rows.append({
                "Field": field,
                "User Document": user_val,
                "Vendor Document": vendor_val,
                "Status": status,
                "Similarity Score": int(score)
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def get_verification_summary(comparison_df: pd.DataFrame) -> Dict[str, any]:
        total = len(comparison_df)
        matches = len(comparison_df[comparison_df['Status'] == 'Match'])
        partial = len(comparison_df[comparison_df['Status'] == 'Partial Match'])
        mismatches = len(comparison_df[comparison_df['Status'] == 'Mismatch'])
        missing = len(comparison_df[comparison_df['Status'].str.contains('Missing', na=False)])
        avg_score = comparison_df['Similarity Score'].mean()
        
        critical_fields = ["Name", "Date of Birth", "Aadhaar Number", "PAN Number"]
        critical_matches = 0
        critical_total = 0
        
        for field in critical_fields:
            field_data = comparison_df[comparison_df['Field'] == field]
            if not field_data.empty:
                critical_total += 1
                if field_data.iloc[0]['Status'] == 'Match':
                    critical_matches += 1
        
        pass_threshold = critical_matches >= critical_total * 0.8 if critical_total > 0 else False
        overall_match_rate = matches / total if total > 0 else 0
        verification_passed = pass_threshold or (overall_match_rate >= 0.7 and mismatches == 0)
        
        return {
            "total_fields": total,
            "matches": matches,
            "partial_matches": partial,
            "mismatches": mismatches,
            "missing": missing,
            "critical_fields_matched": f"{critical_matches}/{critical_total}",
            "average_similarity": round(avg_score, 2),
            "match_rate": round(overall_match_rate * 100, 1),
            "verification_status": "PASSED" if verification_passed else "FAILED"
        }

class ReportGenerator:
    @staticmethod
    def generate_comparison_report(user_report: Dict[str, str], vendor_report: Dict[str, str], 
                                  comparison_df: pd.DataFrame) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
                                     textColor=colors.HexColor('#1a1a1a'), spaceAfter=30, alignment=1)
        story.append(Paragraph("DHRUV OCR - VERIFICATION REPORT", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        summary = ComparisonEngine.get_verification_summary(comparison_df)
        summary_data = [
            ["Verification Status:", summary['verification_status']],
            ["Total Fields Compared:", str(summary['total_fields'])],
            ["Matches:", str(summary['matches'])],
            ["Partial Matches:", str(summary['partial_matches'])],
            ["Mismatches:", str(summary['mismatches'])],
            ["Match Rate:", f"{summary['match_rate']}%"],
            ["Critical Fields Matched:", summary['critical_fields_matched']],
            ["Average Similarity:", f"{summary['average_similarity']}%"],
            ["Generated At:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a90e2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.5*inch))
        
        story.append(Paragraph("DETAILED COMPARISON", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        comparison_data = [["Field", "User", "Vendor", "Status", "Score"]]
        for _, row in comparison_df.iterrows():
            comparison_data.append([
                row['Field'],
                row['User Document'][:30],
                row['Vendor Document'][:30],
                row['Status'],
                f"{row['Similarity Score']}%"
            ])
        
        comparison_table = Table(comparison_data, colWidths=[1.5*inch, 1.8*inch, 1.8*inch, 1.2*inch, 0.8*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(comparison_table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

# Main Streamlit Application
def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #1e293b; font-size: 3rem; margin-bottom: 0;'>
            🔍 Dhruv OCR
        </h1>
        <p style='text-align: center; color: #64748b; font-size: 1.2rem; margin-top: 0;'>
            Professional Document Verification System
        </p>
        <hr style='margin: 20px 0; border: 1px solid #e2e8f0;'>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        st.info("*Dhruv OCR* uses advanced OCR and AI to verify identity documents with high accuracy.")
        
        st.markdown("### 📋 Supported Documents")
        st.markdown("""
        - ✅ Aadhaar Card
        - ✅ PAN Card
        - ✅ Passport
        - ✅ Driving License
        - ✅ Voter ID
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Features")
        st.markdown("""
        - Multi-strategy name extraction
        - Fuzzy field matching
        - Automatic document type detection
        - PDF report generation
        - Real-time processing
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Match Thresholds")
        match_threshold = st.slider("Match Threshold (%)", 70, 100, Config.FUZZY_MATCH_THRESHOLD_MATCH)
        partial_threshold = st.slider("Partial Match Threshold (%)", 50, 80, Config.FUZZY_MATCH_THRESHOLD_PARTIAL)
        
        Config.FUZZY_MATCH_THRESHOLD_MATCH = match_threshold
        Config.FUZZY_MATCH_THRESHOLD_PARTIAL = partial_threshold
        
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>© 2025 Dhruv OCR | v1.0</p>", unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📄 Document Verification", "📊 Batch Processing", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Upload Documents for Verification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 User Document")
            user_file = st.file_uploader("Upload User Document", type=['png', 'jpg', 'jpeg', 'webp'], key="user")
            if user_file:
                user_image = Image.open(user_file)
                st.image(user_image, caption="User Document", use_container_width=True)
        
        with col2:
            st.markdown("#### 🏢 Vendor Document")
            vendor_file = st.file_uploader("Upload Vendor Document", type=['png', 'jpg', 'jpeg', 'webp'], key="vendor")
            if vendor_file:
                vendor_image = Image.open(vendor_file)
                st.image(vendor_image, caption="Vendor Document", use_container_width=True)
        
        st.markdown("---")
        
        if st.button("🚀 Start Verification", type="primary", use_container_width=True):
            if user_file and vendor_file:
                with st.spinner("🔄 Processing documents... This may take a moment."):
                    # Process user document
                    user_img_array = np.array(user_image)
                    user_report = DocumentProcessor.process_document(user_img_array)
                    
                    # Process vendor document
                    vendor_img_array = np.array(vendor_image)
                    vendor_report = DocumentProcessor.process_document(vendor_img_array)
                    
                    # Compare documents
                    comparison_df = ComparisonEngine.compare(user_report, vendor_report)
                    summary = ComparisonEngine.get_verification_summary(comparison_df)
                    
                    # Store in session state
                    st.session_state['comparison_df'] = comparison_df
                    st.session_state['summary'] = summary
                    st.session_state['user_report'] = user_report
                    st.session_state['vendor_report'] = vendor_report
                
                st.success("✅ Verification completed successfully!")
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Verification Results")
                
                # Status badge
                status = summary['verification_status']
                if status == "PASSED":
                    st.markdown(f"<div style='text-align: center;'><span class='success-badge'>✓ VERIFICATION {status}</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: center;'><span class='failed-badge'>✗ VERIFICATION {status}</span></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Fields", summary['total_fields'])
                with col2:
                    st.metric("Matches", summary['matches'], delta=f"+{summary['matches']}")
                with col3:
                    st.metric("Match Rate", f"{summary['match_rate']}%")
                with col4:
                    st.metric("Avg Similarity", f"{summary['average_similarity']}%")
                
                # Additional metrics
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Partial Matches", summary['partial_matches'])
                with col6:
                    st.metric("Mismatches", summary['mismatches'], delta=f"-{summary['mismatches']}" if summary['mismatches'] > 0 else "0")
                with col7:
                    st.metric("Critical Fields", summary['critical_fields_matched'])
                
                st.markdown("---")
                
                # Detailed comparison table
                st.markdown("### 📋 Detailed Field Comparison")
                
                # Style the dataframe
                def highlight_status(row):
                    if row['Status'] == 'Match':
                        return ['background-color: #d1fae5'] * len(row)
                    elif row['Status'] == 'Partial Match':
                        return ['background-color: #fef3c7'] * len(row)
                    elif 'Missing' in row['Status']:
                        return ['background-color: #fee2e2'] * len(row)
                    else:
                        return ['background-color: #fecaca'] * len(row)
                
                styled_df = comparison_df.style.apply(highlight_status, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Extracted fields comparison
                st.markdown("---")
                st.markdown("### 🔍 Extracted Fields Detail")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 👤 User Document Fields")
                    user_display = {k: v for k, v in user_report.items() if not k.startswith('_')}
                    for key, value in user_display.items():
                        st.text(f"{key}: {value}")
                
                with col2:
                    st.markdown("#### 🏢 Vendor Document Fields")
                    vendor_display = {k: v for k, v in vendor_report.items() if not k.startswith('_')}
                    for key, value in vendor_display.items():
                        st.text(f"{key}: {value}")
                
                # Download report
                st.markdown("---")
                st.markdown("### 📥 Download Report")
                
                pdf_bytes = ReportGenerator.generate_comparison_report(user_report, vendor_report, comparison_df)
                
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"dhruv_ocr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
            else:
                st.error("⚠️ Please upload both User and Vendor documents to proceed.")
    
    with tab2:
        st.markdown("### 📊 Batch Processing")
        st.info("🚧 Batch processing feature coming soon! Process multiple document pairs at once.")
        
        st.markdown("""
        *Upcoming Features:*
        - Process multiple document pairs simultaneously
        - Export results to CSV/Excel
        - Automated workflow integration
        - API access for enterprise clients
        """)
    
    with tab3:
        st.markdown("### ℹ️ About Dhruv OCR")
        
        st.markdown("""
        *Dhruv OCR* is a professional document verification system designed to accurately extract 
        and compare information from identity documents using advanced Optical Character Recognition (OCR) 
        and intelligent field matching algorithms.
        
        #### 🎯 Key Capabilities:
        
        1. *Multi-Strategy Name Extraction*
           - Context-based extraction
           - Hindi-English pattern recognition
           - Address relation parsing
           - Title case detection
           - DOB proximity analysis
        
        2. *Field Extraction*
           - Aadhaar Number (12 digits)
           - PAN Number (alphanumeric)
           - Date of Birth (multiple formats)
           - Gender identification
           - Mobile Number (10 digits)
           - VID (Virtual ID - 16 digits)
        
        3. *Intelligent Matching*
           - Fuzzy string matching
           - Configurable thresholds
           - Critical field prioritization
           - Comprehensive scoring system
        
        4. *Document Types Supported*
           - Aadhaar Card
           - PAN Card
           - Passport
           - Driving License
           - Voter ID
        
        #### 🔒 Privacy & Security:
        
        - All processing happens locally
        - No data is stored or transmitted
        - Documents are processed in memory only
        - Automatic cleanup after processing
        
        #### 📚 Technology Stack:
        
        - *OCR Engine:* EasyOCR
        - *Image Processing:* OpenCV
        - *Fuzzy Matching:* RapidFuzz
        - *PDF Generation:* ReportLab
        - *Web Framework:* Streamlit
        
        #### 📞 Contact & Support:
        
        For enterprise solutions, API access, or technical support, please contact your system administrator.
        
        ---
        
        *Version:* 1.0.0  
        *Last Updated:* October 2025  
        *Developed with ❤️ for accurate document verification*
        """)

if __name__ == "__main__":
    main()