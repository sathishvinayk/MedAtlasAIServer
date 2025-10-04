# Keep the MEDICAL_KEYWORDS and other constants from your original code
MEDICAL_KEYWORDS = {
    "SYMPTOM": [
        # Headache patterns
        "headache", "headaches", "migraine", "migraines", "head pain", 
        "pressure in my head", "band around my head", "head pounding",
        "head throbbing", "head hurts", "head ache",
        
        # Nausea patterns
        "nausea", "nauseous", "queasy", "upset stomach", "feel sick", 
        "feel like throwing up", "stomach upset",
        
        # Dizziness patterns
        "dizziness", "dizzy", "lightheaded", "light-headed", "vertigo",
        "room spinning", "feel faint", "unsteady",
        
        # Existing symptoms
        "fever", "cough", "pain", "fatigue", "tired", "tiredness", 
        "shortness of breath", "dry cough", "exhaustion", "weakness",
        "chest pain", "sore throat", "body aches", "chills", "sweating", "vomiting",
        
        # Vision patterns
        "blurred vision", "blurry vision", "sensitivity to light", "light bothers me",
        "eyes sensitive", "vision problems"
    ],
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
    # ACE Inhibitors
    "laciniprol": "lisinopril",
    "lucinipral": "lisinopril", 
    "luciniprol": "lisinopril",
    "lusinoprol": "lisinopril",
    "lucinipral": "lisinopril",
    "lizzanoprol": "lisinopril",
    "lissinoprol": "lisinopril",
    "lysinoprol": "lisinopril",
    "lizanoprol": "lisinopril",
    
    # ARBs
    "low-sorten": "losartan",
    "losartin": "losartan",
    "losertan": "losartan",
    "lossarton": "losartan",  # ADDED
    "los arden": "losartan",  # ADDED
    
    # Pain medications
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "ibuprophen": "ibuprofen",  # ADDED
    
    # Existing mappings
    "cozaar": "losartan",
    "lipitor": "atorvastatin",
    "zocor": "simvastatin",
    "glucophage": "metformin",
    "vasotec": "enalapril",
    "prinivil": "lisinopril",
    "zestril": "lisinopril"
}