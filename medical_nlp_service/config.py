import os

# Configuration
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
WHISPER_MODEL_SIZE = "base"
MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', 'microsoft/BioGPT-Large')

# Thread pool sizes
MAX_WORKERS_BIOBERT = int(os.getenv('MAX_WORKERS_BIOBERT', '2'))
MAX_WORKERS_SPACY = int(os.getenv('MAX_WORKERS_SPACY', '2'))
MAX_WORKERS_WHISPER = int(os.getenv('MAX_WORKERS_WHISPER', '1'))
MAX_WORKERS_SENTENCE = int(os.getenv('MAX_WORKERS_SENTENCE', '2'))
MAX_WORKERS_GENERAL = int(os.getenv('MAX_WORKERS_GENERAL', '4'))
MAX_WORKERS_LLM = int(os.getenv('MAX_WORKERS_LLM', '1'))

# Medical keywords and patterns
MEDICAL_KEYWORDS = {
    "SYMPTOM": ["headache", "fever", "cough", "pain", "nausea", "dizziness", 
                "fatigue", "tired", "tiredness", "shortness of breath", 
                "dry cough", "exhaustion", "weakness", "nausea"],
    "MEDICATION": ["ibuprofen", "aspirin", "amoxicillin", "lisinopril", 
                  "laciniprol", "metformin", "tylenol", "advil", "atenolol",
                  "amlodipine", "simvastatin", "atorvastatin", "omeprazole"],
    "DIAGNOSIS": ["hypertension", "high blood pressure", "diabetes", 
                 "migraine", "infection", "arthritis", "asthma", "pneumonia",
                 "bronchitis", "influenza", "covid"],
    "BODY_PART": ["head", "chest", "arm", "leg", "back", "stomach", "throat",
                 "neck", "abdomen", "heart", "lungs"]
}

MEDICATION_SYNONYMS = {
    "laciniprol": "lisinopril",
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen"
}