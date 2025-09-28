import os

# Configuration
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
WHISPER_MODEL_SIZE = "base"
# MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', 'emilyalsentzer/Bio_ClinicalBERT')
# MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', 'microsoft/BioGPT-Large')
MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', '')
# Alternatives: 'mistralai/Mistral-7B-v0.1', 'microsoft/BioGPT-Large', 'stanford-crfm/BioMedLM'
PYANNOTE_AUTH_TOKEN = os.getenv('PYANNOTE_AUTH_TOKEN', '') 
# Medical keywords and patterns (unchanged)

