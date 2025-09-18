import os

# Configuration
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
WHISPER_MODEL_SIZE = "base"
MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', 'microsoft/BioGPT-Large')
# Alternatives: 'mistralai/Mistral-7B-v0.1', 'microsoft/BioGPT-Large', 'stanford-crfm/BioMedLM'
# Thread pools for each model type
MAX_WORKERS_BIOBERT = int(os.getenv('MAX_WORKERS_BIOBERT', '2'))
MAX_WORKERS_SPACY = int(os.getenv('MAX_WORKERS_SPACY', '2'))
MAX_WORKERS_WHISPER = int(os.getenv('MAX_WORKERS_WHISPER', '1'))
MAX_WORKERS_SENTENCE = int(os.getenv('MAX_WORKERS_SENTENCE', '2'))
MAX_WORKERS_GENERAL = int(os.getenv('MAX_WORKERS_GENERAL', '4'))
MAX_WORKERS_LLM = int(os.getenv('MAX_WORKERS_LLM', '1'))  # LLM is memory-intensive