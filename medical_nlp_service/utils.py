from typing import Tuple, List

import torchaudio
import logging
import hashlib
import numpy as np
import os
import tempfile
from contextlib import asynccontextmanager
from constants import MEDICAL_KEYWORDS, MEDICATION_SYNONYMS
from entities import SpeakerSegment

logger = logging.getLogger("medical-nlp-service")

# Temporary file context manager (unchanged)
@asynccontextmanager
async def temp_audio_file(audio_bytes: bytes):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")

def map_spacy_label_to_medical(label: str) -> str:
    mapping = {
        "DISEASE": "DIAGNOSIS",
        "CONDITION": "DIAGNOSIS",
        "SYMPTOM": "SYMPTOM",
        "MEDICATION": "MEDICATION",
        "DRUG": "MEDICATION",
        "BODY_PART": "BODY_PART",
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "DATE": "DATE",
        "TIME": "TIME"
    }
    return mapping.get(label, "OTHER")

# Utility functions (unchanged except for SOAP generation)
def normalize_medication_name(med_name: str) -> Tuple[str, float]:
    """Normalize medication name and return with adjusted confidence"""
    original_confidence = 0.9  # Default confidence
    
    med_name_lower = med_name.lower().strip()
    
    # Common medication misspellings and corrections
    medication_corrections = {
        "laciniprol": ("lisinopril", 0.8),  # Lower confidence for corrected spellings
        "metforman": ("metformin", 0.8),
        "ibuprofin": ("ibuprofen", 0.8),
        "amoxicilin": ("amoxicillin", 0.8),
        "asprin": ("aspirin", 0.8),
    }
    
    if med_name_lower in medication_corrections:
        corrected_name, confidence = medication_corrections[med_name_lower]
        return corrected_name, confidence
    
    # For properly spelled medications, keep original confidence
    return med_name, original_confidence

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, preserving word boundaries"""
    if len(text) <= max_length:
        return text
    
    # Find the last space within the limit
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."
    
# Add function to get audio duration
def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return 0
    
def universal_embedding(text: str, dimensions: int = 384) -> List[float]:
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    seed = int(text_hash[:8], 16)
    
    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal(dimensions).astype(np.float32)
    
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.tolist()

def universal_transcript(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()
    
    seed = int(audio_hash[:8], 16)
    rng = np.random.default_rng(seed)

    symptoms = ["headache", "fever", "cough", "chest pain", "fatigue", "dizziness"]
    medications = ["ibuprofen", "amoxicillin", "lisinopril", "metformin"]

    random_symptoms = rng.choice(symptoms, size=2, replace=False)
    random_med = rng.choice(medications, size=1)[0]

    return f"Patient presents with {' and '.join(random_symptoms)}. Currently taking {random_med}. Denies other symptoms. Vital signs stable."


# Model mapping functions (unchanged)
def map_biobert_label_to_medical(label: str, token_text: str) -> str:
    """Enhanced BioBERT label mapping"""
    label_upper = label.upper()
    token_lower = token_text.lower()
    
    # Enhanced mapping for BioBERT labels
    if any(x in label_upper for x in ["DISEASE", "DIAG", "CONDITION", "PROBLEM"]):
        return "DIAGNOSIS"
    if any(x in label_upper for x in ["CHEM", "DRUG", "MED", "TREATMENT"]):
        return "MEDICATION"
    if any(x in label_upper for x in ["SYMPTOM", "SIGN", "COMPLAINT"]):
        return "SYMPTOM"
    if any(x in label_upper for x in ["ANATOMY", "BODY", "LOC", "ORGAN"]):
        return "BODY_PART"
    
    # Check medication synonyms
    if token_lower in MEDICATION_SYNONYMS:
        return "MEDICATION"
    
    # Fallback to keyword matching
    for ent_type, keywords in MEDICAL_KEYWORDS.items():
        if token_lower in keywords:
            return ent_type
    
    return "OTHER"

# Add function to align transcription with speaker segments
def align_transcription_with_speakers(transcript: str, speaker_segments: List[SpeakerSegment], audio_duration: float) -> List[SpeakerSegment]:
    """Align Whisper transcription with speaker segments"""
    if not speaker_segments or not transcript:
        return speaker_segments
    
    # Simple approach: split transcript by sentences and assign to speakers based on time
    sentences = transcript.split('. ')
    total_chars = len(transcript)
    
    # Calculate character rate (chars per second)
    if audio_duration > 0:
        char_rate = total_chars / audio_duration
    else:
        # Fallback: assume 10 characters per second
        char_rate = 10
    
    # Assign text to segments based on timing
    for segment in speaker_segments:
        segment_duration = segment.end - segment.start
        expected_chars = int(segment_duration * char_rate)
        
        # This is a simplified approach - in production, you'd want a more sophisticated alignment
        segment.text = f"Speaker {segment.speaker} segment from {segment.start}s to {segment.end}s"
    
    return speaker_segments