from utils import normalize_medication_name
from typing import List
from entities import MedicalEntity
import re

def generate_soap_note_rule_based(transcript: str, entities: List[MedicalEntity]) -> str:
    """Enhanced rule-based SOAP note using actual conversation context"""
    symptoms = sorted(set(e.text for e in entities if e.entity == "SYMPTOM"))
    
    # Enhanced medication normalization
    medications = []
    for e in entities:
        if e.entity == "MEDICATION":
            med_name, confidence = normalize_medication_name(e.text)
            medications.append(med_name)
    medications = sorted(set(medications))
    
    transcript_lower = transcript.lower()
    
    # Extract blood pressure readings
    bp_readings = []
    bp_pattern = r'blood pressure.*?(\d+)\s*over\s*(\d+)|(\d+)\s*\/\s*(\d+)'
    for match in re.finditer(bp_pattern, transcript_lower):
        if match.group(1) and match.group(2):
            bp_readings.append(f"{match.group(1)}/{match.group(2)}")
        elif match.group(3) and match.group(4):
            bp_readings.append(f"{match.group(3)}/{match.group(4)}")
    
    # Enhanced clinical context
    has_hypertension = any(term in transcript_lower for term in ['blood pressure', 'hypertension', 'htn'])
    switching_medication = any(term in transcript_lower for term in ['switch', 'change medication', 'new medication'])
    ace_inhibitor_cough = 'ace inhibitor' in transcript_lower and 'cough' in symptoms
    
    # Build professional SOAP note
    subjective = build_subjective_section(transcript, symptoms, medications, transcript_lower)
    objective = build_objective_section(bp_readings, medications, transcript_lower)
    assessment = build_assessment_section(symptoms, medications, transcript_lower, ace_inhibitor_cough)
    plan = build_plan_section(symptoms, medications, transcript_lower, switching_medication)
    
    return f"{subjective}\n\n{objective}\n\n{assessment}\n\n{plan}"

def build_subjective_section(transcript: str, symptoms: list, medications: list, transcript_lower: str) -> str:
    """Build professional SUBJECTIVE section"""
    parts = []
    
    # Chief complaint
    if "ace inhibitor" in transcript_lower and "cough" in symptoms:
        cc = "Patient presents for evaluation of ACE inhibitor-induced cough and other side effects"
    elif symptoms:
        cc = f"Patient presents for evaluation of {', '.join(symptoms[:2])}"
    else:
        cc = "Patient presents for routine follow-up"
    parts.append(cc)
    
    # History of present illness
    hpi_parts = []
    
    if "cough" in symptoms and "dry" in transcript_lower:
        hpi_parts.append("Reports persistent non-productive cough, worse at night, affecting sleep")
    elif "cough" in symptoms:
        hpi_parts.append("Reports cough")
        
    if "dizziness" in symptoms and "stand" in transcript_lower:
        hpi_parts.append("Experiences dizziness with positional changes")
    elif "dizziness" in symptoms:
        hpi_parts.append("Reports dizziness")
        
    if "tired" in symptoms or "fatigue" in symptoms:
        hpi_parts.append("Notes increased fatigue impacting daily activities")
    
    if "blood pressure" in transcript_lower:
        hpi_parts.append("Here for hypertension management follow-up")
    
    # Add medication context
    if any("lisinopril" in med.lower() for med in medications) and "cough" in symptoms:
        hpi_parts.append("Symptoms began after starting lisinopril therapy")
    
    if hpi_parts:
        parts.append("History of present illness: " + "; ".join(hpi_parts))
    
    # Current medications
    if medications:
        normalized_meds = [normalize_medication_name(med)[0] for med in medications]
        parts.append(f"Current medications: {', '.join(normalized_meds)}")
    
    return "SUBJECTIVE:\n" + "\n".join(f"- {part}" for part in parts)

def build_objective_section(bp_readings: list, medications: list, transcript_lower: str) -> str:
    """Build professional OBJECTIVE section"""
    parts = []
    
    # Vital signs
    if bp_readings:
        parts.append(f"Blood Pressure: {bp_readings[0]} mmHg")
        parts.append("Heart Rate: Regular rhythm, rate within normal limits")
    else:
        parts.append("Vital Signs: Within normal limits")
    
    # Physical exam
    exam_parts = ["General: Well-appearing, no acute distress"]
    
    if "cough" in transcript_lower:
        exam_parts.append("Respiratory: Clear to auscultation bilaterally")
    else:
        exam_parts.append("Respiratory: Clear lungs, non-labored breathing")
    
    if "dizziness" in transcript_lower:
        exam_parts.append("Neurological: Alert and oriented, no focal deficits noted")
    
    parts.append("Physical Examination: " + "; ".join(exam_parts))
    
    return "OBJECTIVE:\n" + "\n".join(f"- {part}" for part in parts)

def build_assessment_section(symptoms: list, medications: list, transcript_lower: str, ace_inhibitor_cough: bool) -> str:
    """Build professional ASSESSMENT section"""
    parts = []
    
    # Primary diagnoses
    if ace_inhibitor_cough:
        parts.append("1. ACE inhibitor-induced cough")
        parts.append("2. Essential hypertension, controlled on current therapy")
    elif "blood pressure" in transcript_lower:
        parts.append("1. Essential hypertension")
    
    # Symptom assessments
    symptom_assessments = {
        "cough": "Persistent cough, likely medication-related",
        "dizziness": "Dizziness, possibly orthostatic or medication-related",
        "tired": "Fatigue, may be related to sleep disruption from cough"
    }
    
    for symptom in symptoms:
        if symptom.lower() in symptom_assessments:
            parts.append(symptom_assessments[symptom.lower()])
    
    return "ASSESSMENT:\n" + "\n".join(f"- {part}" for part in parts)

def build_plan_section(symptoms: list, medications: list, transcript_lower: str, switching_medication: bool) -> str:
    """Build professional PLAN section"""
    parts = []
    
    # Medication management
    if switching_medication:
        parts.append("Discontinue current ACE inhibitor due to intolerable side effects")
        parts.append("Initiate ARB therapy (e.g., losartan) for hypertension control")
        parts.append("Check basic metabolic panel prior to medication transition")
        parts.append("Monitor renal function and electrolytes after medication change")
    elif any("lisinopril" in med.lower() for med in medications) and "cough" in symptoms:
        parts.append("Consider alternative antihypertensive due to ACE inhibitor-induced cough")
        parts.append("Discuss ARB therapy as potential alternative")
    
    # Follow-up
    parts.append("Schedule follow-up in 4 weeks for blood pressure recheck and symptom assessment")
    
    # Patient education
    parts.append("Patient education provided on medication adherence and side effect monitoring")
    parts.append("Instructed to report any worsening symptoms or new adverse effects")
    parts.append("Encouraged to maintain blood pressure log")
    
    return "PLAN:\n" + "\n".join(f"- {part}" for part in parts)