# Keep the MEDICAL_KEYWORDS and other constants from your original code
MEDICAL_KEYWORDS = {
    "SYMPTOM": ["headache", "fever", "cough", "pain", "nausea", "dizziness", 
                "fatigue", "tired", "tiredness", "shortness of breath", 
                "dry cough", "exhaustion", "weakness", "nausea", "chest pain",
                "sore throat", "body aches", "chills", "sweating", "vomiting"],
    "MEDICATION": ["ibuprofen", "aspirin", "amoxicillin", "lisinopril", 
                  "laciniprol", "metformin", "tylenol", "advil", "atenolol",
                  "amlodipine", "simvastatin", "atorvastatin", "omeprazole",
                  "acetaminophen", "warfarin", "insulin", "prednisone"],
    "DIAGNOSIS": ["hypertension", "high blood pressure", "diabetes", 
                 "migraine", "infection", "arthritis", "asthma", "pneumonia",
                 "bronchitis", "influenza", "covid", "coronary artery disease",
                 "heart failure", "copd", "depression", "anxiety"],
    "BODY_PART": ["head", "chest", "arm", "leg", "back", "stomach", "throat",
                 "neck", "abdomen", "heart", "lungs", "kidney", "liver"],
    "PROCEDURE": ["surgery", "operation", "biopsy", "scan", "x-ray", "mri"],
    "LAB_TEST": ["blood test", "urine test", "ct scan", "ekg", "ecg"]
}

MEDICATION_SYNONYMS = {
    "laciniprol": "lisinopril",
    "tylenol": "acetaminophen", 
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "lusinoprol": "lisinopril",
    "lucinipral": "lisinopril",
    "lizzanoprol": "lisinopril", 
    "lissinoprol": "lisinopril",
    "lysinoprol": "lisinopril",
    "low-sorten": "losartan",
    "losartin": "losartan",
    "losertan": "losartan",
    "cozaar": "losartan",
    "lipitor": "atorvastatin",
    "zocor": "simvastatin",
    "glucophage": "metformin",
    "vasotec": "enalapril",
    "prinivil": "lisinopril",
    "zestril": "lisinopril"
}