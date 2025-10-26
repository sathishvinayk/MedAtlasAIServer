# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import validator, BaseModel, Field
from dataclasses import dataclass, field
import logging
from typing import Any, List, Tuple, AsyncGenerator, Dict, Optional
import base64
from pyannote.audio import Pipeline
import os
import re
import asyncio
import time
from threading import Lock
from contextlib import asynccontextmanager
import uuid
from concurrent.futures import ThreadPoolExecutor
import torch
import base64 
import time    
import struct
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
)
import tempfile  # Make sure this is imported
import os
# from utils import normalize_medication_name, truncate_text, get_audio_duration, map_spacy_label_to_medical, universal_transcript, temp_audio_file, map_biobert_label_to_medical, align_transcription_with_speakers
# from soap_generator import generate_soap_note_rule_based
from audio_models import ProcessAudioRequest, ProcessAudioResponse
# from shared_models import (
#     MedicalEntity, 
#     SpeakerSegment
# )
from datetime import datetime, timedelta
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
# from entity_extractor import extract_entities_keywords, deduplicate_entities, filter_negated_entities, extract_medication_changes, extract_medical_patterns
# from constants import MEDICAL_KEYWORDS
from config import WHISPER_MODEL_SIZE, MEDICAL_LLM_NAME, PYANNOTE_AUTH_TOKEN
from audioStreamProcessor import AudioStreamProcessor
import numpy as np

# from shared_models import RealtimeResult, PatientContext
# from medical_conversation_analyzer import MedicalConversationAnalyzer
# from real_time_medical_validator import RealTimeMedicalValidator
# from progressive_soap_builder import ProgressiveSOAPBuilder

# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="emilyalsentzer/Bio_ClinicalBERT", filename="pytorch_model.bin", force_download=True)
# hf_hub_download(repo_id="microsoft/BioGPT-Large", filename="pytorch_model.bin", force_download=True)

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    await load_models_async()
    yield
    # Shutdown: cleanup
    await cleanup_models()

app = FastAPI(
    title="Medical NLP Service",
    description="API for medical audio processing, transcription, and SOAP note generation with fine-tuned LLMs",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-Request-ID"]
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("medical-nlp-service")

# Global models (loaded asynchronously)
WHISPER_MODEL = None
BIOBERT_MODEL = None
CLINICALBERT_MODEL = None
SPACY_MODEL = None
MEDICAL_LLM = None
MEDICAL_TOKENIZER = None
PYANNOTE_PIPELINE = None
REALTIME_PROCESSOR = None

# Thread pools for each model type
MAX_WORKERS_BIOBERT = int(os.getenv('MAX_WORKERS_BIOBERT', '2'))
MAX_WORKERS_SPACY = int(os.getenv('MAX_WORKERS_SPACY', '2'))
MAX_WORKERS_WHISPER = int(os.getenv('MAX_WORKERS_WHISPER', '1'))
MAX_WORKERS_SENTENCE = int(os.getenv('MAX_WORKERS_SENTENCE', '2'))
MAX_WORKERS_GENERAL = int(os.getenv('MAX_WORKERS_GENERAL', '4'))
MAX_WORKERS_LLM = int(os.getenv('MAX_WORKERS_LLM', '1'))  # LLM is memory-intensive
MAX_WORKERS_PYANNOTE = int(os.getenv('MAX_WORKERS_PYANNOTE', '1'))

WHISPER_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_WHISPER, thread_name_prefix="whisper_")
BIOBERT_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_BIOBERT, thread_name_prefix="biobert_")
SPACY_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_SPACY, thread_name_prefix="spacy_")
GENERAL_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_GENERAL, thread_name_prefix="general_")
LLM_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_LLM, thread_name_prefix="llm_")
PYANNOTE_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_PYANNOTE, thread_name_prefix="pyannote_")  # Add pyannote pool

# Thread safety
_biobert_lock = Lock()
_spacy_lock = Lock()
_llm_lock = Lock()
_pyannote_lock = Lock()  # Add pyannote lock
_model_load_lock = Lock()
_models_loaded = False

class MedicalEntity(BaseModel):
    entity: str = Field(..., description="Type of medical entity (SYMPTOM, MEDICATION, etc.)")
    text: str = Field(..., description="The actual text of the entity")
    start: int = Field(..., description="Start character position in text")
    end: int = Field(..., description="End character position in text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

class EnhancedMedicalEntity(MedicalEntity):
    """Enhanced medical entity with negation and temporal attributes"""
    negated: bool = Field(default=False, description="Whether the entity is negated")
    negation_confidence: float = Field(default=0.0, description="Confidence of negation detection")
    negation_phrase: Optional[str] = Field(default=None, description="The negation phrase found")
    duration: Optional[str] = Field(default=None, description="Duration mentioned with entity")
    normalized_duration: Optional[str] = Field(default=None, description="Normalized duration")
    onset: Optional[str] = Field(default=None, description="Onset time mentioned")
    normalized_onset: Optional[str] = Field(default=None, description="Normalized onset time")
    temporal_context: Optional[str] = Field(default=None, description="Temporal context")

class SpeakerSegment(BaseModel):
    speaker: str = Field(..., description="Speaker identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")

@dataclass
class PatientContext:
    """Patient context maintained throughout the conversation"""
    session_id: str
    conversation_history: List[str] = field(default_factory=list)
    current_symptoms: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    medical_history: List[str] = field(default_factory=list)
    soap_note_sections: Dict[str, str] = field(default_factory=lambda: {
        "subjective": "", "objective": "", "assessment": "", "plan": ""
    })
    extracted_entities: List[EnhancedMedicalEntity] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    audio_buffer: bytearray = field(default_factory=bytearray)
    last_activity: float = field(default_factory=time.time)

@dataclass
class RealtimeResult:
    """Internal real-time processing result"""
    type: str  # "transcript", "entities", "soap_update", "medical_alert"
    data: Dict[str, Any]
    session_id: str
    is_partial: bool = True
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.9

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

class NegationTemporalProcessor:
    """Handles negation detection and temporal normalization for medical entities"""
    
    def __init__(self):
        # Negation patterns
        self.negation_patterns = [
            r'\b(no|not|denies|denied|without|negative|absence of|free of)\b',
            r'\b(doesn\'t have|does not have|hasn\'t|has not)\b', 
            r'\b(ruled out|excluded|dismissed)\b',
            r'\b(unremarkable|normal|clear)\b.*\b(for|of)\b',
            r'\b(no history of|no sign of|no evidence of)\b',  # ADDED
            r'\b(negative|neg)\b.*\b(for|of)\b',  # ADDED
            r'\b(declines|refutes)\b',  # ADDED
        ]
        
        # Temporal patterns for duration
        self.duration_patterns = [
            r'\bfor\s+(\d+)\s*(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b',
            r'\b(\d+)\s*(day|days|week|weeks|month|months|year|years|hour|hours)\s+ago\b',
            r'\blast\s*(night|week|month|year)\b',
            r'\bpast\s+(\d+)\s*(day|days|week|weeks|month|months)\b',
            r'\bfor the past\s+(\d+)\s*(day|days|week|weeks|month|months)\b',  # ADDED
            r'\bfor the last\s+(\d+)\s*(day|days|week|weeks|month|months)\b',  # ADDED
            r'\bover the past\s+(\d+)\s*(day|days|week|weeks)\b',  # ADDED
            r'\bsince last\s+(week|month|night)\b',  # ADDED
        ]
        
        # Temporal patterns for onset
        self.onset_patterns = [
            r'\bsince\s+(.*?)(?=\.|,|$)',
            r'\bstarted\s+(.*?)(?=\.|,|$)',
            r'\bbegan\s+(.*?)(?=\.|,|$)',
            r'\bonset\s+(.*?)(?=\.|,|$)',
            r'\b(?:since|from)\s+(yesterday|last night|last week|last month)\b',
            r'\b(?:in|during)\s+(the\s+)?(morning|afternoon|evening|night)\b',  # ADDED
            r'\bwhen\s+I\s+(.*?)(?=\.|,|$)',  # ADDED - "when I stand up"
            r'\bafter\s+I\s+(.*?)(?=\.|,|$)',  # ADDED - "after I've been at the computer"
        ]
        
        # Relative time mappings
        self.relative_time_map = {
            'yesterday': lambda: datetime.now() - timedelta(days=1),
            'last night': lambda: datetime.now().replace(hour=20, minute=0, second=0) - timedelta(days=1),
            'last week': lambda: datetime.now() - timedelta(weeks=1),
            'last month': lambda: datetime.now() - relativedelta(months=1),
            'today': lambda: datetime.now(),
            'this morning': lambda: datetime.now().replace(hour=8, minute=0, second=0)
        }

    def detect_negation(self, text: str, entity_text: str, entity_start: int) -> Dict[str, any]:
        """Detect if an entity is negated in the text"""
        text_lower = text.lower()
        entity_lower = entity_text.lower()
        
        # Look for negation patterns in proximity to the entity
        entity_context = self._get_entity_context(text, entity_start, len(entity_text))
        
        for pattern in self.negation_patterns:
            negation_matches = list(re.finditer(pattern, entity_context.lower()))
            for match in negation_matches:
                # Check if negation is close to the entity (within 10 words)
                if self._is_negation_proximal(match, entity_start, entity_context):  # ← FIXED CALL
                    return {
                        "negated": True,
                        "negation_phrase": match.group(),
                        "confidence": 0.9,
                        "context": entity_context
                    }
        
        return {"negated": False, "confidence": 0.9}

    def _get_entity_context(self, text: str, entity_start: int, entity_length: int, window_size: int = 100) -> str:
        """Extract context around the entity"""
        start = max(0, entity_start - window_size)
        end = min(len(text), entity_start + entity_length + window_size)
        return text[start:end]

    def _is_negation_proximal(self, negation_match, entity_start: int, context: str) -> bool:
        """Check if negation is close enough to the entity to be relevant"""
        try:
            # Get positions within the context
            negation_pos = negation_match.start()
            
            # Calculate where the entity appears in this context
            # The context starts at (entity_start - window_size) in original text
            window_size = 100  # Should match _get_entity_context
            context_start_in_original = max(0, entity_start - window_size)
            entity_pos_in_context = entity_start - context_start_in_original
            
            # If entity doesn't appear in this context, return False
            if entity_pos_in_context < 0 or entity_pos_in_context >= len(context):
                return False
            
            # Calculate distance between negation and entity in the context
            distance = abs(negation_pos - entity_pos_in_context)
            
            # Use word-based distance for more accurate medical context
            words_between = self._count_words_between(context, negation_pos, entity_pos_in_context)
            
            # Return True if within reasonable character AND word distance
            return distance < 150 and words_between < 10
            
        except Exception as e:
            logger.warning(f"Proximity check error: {e}")
            return False

    def _count_words_between(self, text: str, pos1: int, pos2: int) -> int:
        """Count words between two positions in text"""
        try:
            start = min(pos1, pos2)
            end = max(pos1, pos2)
            segment = text[start:end]
            words = re.findall(r'\b\w+\b', segment)
            return len(words)
        except:
            return 999  # Return high number on error to fail safe

    def extract_temporal_info(self, text: str, entity_text: str, entity_start: int) -> Dict[str, any]:
        """Extract temporal information related to an entity"""
        temporal_info = {
            "duration": None,
            "onset": None,
            "normalized_duration": None,
            "normalized_onset": None,
            "temporal_context": None
        }
        
        entity_context = self._get_entity_context(text, entity_start, len(entity_text))
        
        # Extract duration
        duration_info = self._extract_duration(entity_context)
        if duration_info:
            temporal_info.update(duration_info)
        
        # Extract onset
        onset_info = self._extract_onset(entity_context)
        if onset_info:
            temporal_info.update(onset_info)
        
        return temporal_info

    def _extract_duration(self, text: str) -> Optional[Dict[str, any]]:
        """Extract duration information from text"""
        for pattern in self.duration_patterns:
            matches = list(re.finditer(pattern, text.lower()))
            for match in matches:
                if match.group(1) and match.group(2):  # for X days/weeks/etc
                    quantity = int(match.group(1))
                    unit = match.group(2)
                    normalized = self._normalize_duration(quantity, unit)
                    return {
                        "duration": match.group(),
                        "normalized_duration": normalized,
                        "duration_quantity": quantity,
                        "duration_unit": unit
                    }
                elif 'last' in match.group() or 'past' in match.group():
                    return self._parse_relative_duration(match.group())
        
        return None

    def _extract_onset(self, text: str) -> Optional[Dict[str, any]]:
        """Extract onset information from text"""
        for pattern in self.onset_patterns:
            matches = list(re.finditer(pattern, text.lower()))
            for match in matches:
                if match.group(1):
                    onset_phrase = match.group(1).strip()
                    normalized_onset = self._normalize_onset(onset_phrase)
                    return {
                        "onset": onset_phrase,
                        "normalized_onset": normalized_onset
                    }
        
        return None

    def _normalize_duration(self, quantity: int, unit: str) -> str:
        """Normalize duration to standard format"""
        unit_map = {
            'hour': 'hours', 'hours': 'hours',
            'day': 'days', 'days': 'days',
            'week': 'weeks', 'weeks': 'weeks', 
            'month': 'months', 'months': 'months',
            'year': 'years', 'years': 'years'
        }
        
        normalized_unit = unit_map.get(unit, 'days')
        return f"P{quantity}{normalized_unit[0].upper()}"  # ISO 8601 format

    def _normalize_onset(self, onset_phrase: str) -> str:
        """Normalize onset time to standard format"""
        # Check for relative times
        if onset_phrase in self.relative_time_map:
            onset_date = self.relative_time_map[onset_phrase]()
            return onset_date.strftime("%Y-%m-%d")
        
        # Try to parse as absolute date
        try:
            parsed_date = parse(onset_phrase, fuzzy=True)
            return parsed_date.strftime("%Y-%m-%d")
        except:
            # Return the original phrase if parsing fails
            return onset_phrase

    def _parse_relative_duration(self, duration_phrase: str) -> Dict[str, any]:
        """Parse relative duration phrases like 'last week', 'past 3 days'"""
        if 'last week' in duration_phrase:
            return {
                "duration": duration_phrase,
                "normalized_duration": "P7D",
                "duration_quantity": 7,
                "duration_unit": "days"
            }
        elif 'last month' in duration_phrase:
            return {
                "duration": duration_phrase,
                "normalized_duration": "P30D", 
                "duration_quantity": 30,
                "duration_unit": "days"
            }
        
        return None

class MedicalConversationAnalyzer:
    """Analyzes clinical conversation patterns in real-time"""
    
    def __init__(self):
        self.doctor_phrases = [
            r"how are you feeling", r"any pain", r"describe the", r"when did",
            r"where does it hurt", r"rate your pain", r"any medications",
            r"medical history", r"any allergies", r"what brings you"
        ]
        self.patient_phrases = [
            r"i have", r"i feel", r"my pain", r"it hurts", r"i take",
            r"i was diagnosed", r"my doctor said", r"i've been"
        ]
        self.history_phrases = [
            r"history of", r"diagnosed with", r"previous", r"past medical",
            r"for the past", r"since last"
        ]
    
    def analyze_conversation_phase(self, transcript: str) -> str:
        """Detect the phase of clinical conversation"""
        transcript_lower = transcript.lower()
        
        doctor_matches = sum(1 for phrase in self.doctor_phrases 
                           if re.search(phrase, transcript_lower))
        patient_matches = sum(1 for phrase in self.patient_phrases 
                            if re.search(phrase, transcript_lower))
        history_matches = sum(1 for phrase in self.history_phrases 
                            if re.search(phrase, transcript_lower))
        
        if history_matches > 2:
            return "history_taking"
        elif doctor_matches > patient_matches:
            return "questioning"
        elif patient_matches > doctor_matches:
            return "symptom_reporting"
        else:
            return "general"

class ProgressiveSOAPBuilder:
    """Builds SOAP notes progressively as conversation unfolds"""
    
    def __init__(self):
        self.section_templates = {
            "subjective": {
                "symptom_reporting": "Patient reports {symptoms}.",
                "history_sharing": "Relevant history: {history}.",
                "medication_discussion": "Current medications: {medications}."
            }
        }
    
    def update_soap_sections(self, current_sections: Dict[str, str], 
                           transcript: str, entities: List[MedicalEntity],
                           conversation_phase: str) -> Dict[str, str]:
        """Update SOAP sections based on new content and conversation context"""
        updated_sections = current_sections.copy()
        
        # Analyze content and update appropriate sections
        if conversation_phase == "symptom_reporting":
            updated_sections["subjective"] = self._add_to_section(
                updated_sections["subjective"], 
                self._extract_symptom_content(transcript, entities)
            )
        
        elif conversation_phase == "history_taking":
            updated_sections["subjective"] = self._add_to_section(
                updated_sections["subjective"],
                self._extract_history_content(transcript, entities)
            )
        
        elif conversation_phase == "questioning":
            # Doctor questions might go to objective or help structure assessment
            updated_sections["objective"] = self._add_to_section(
                updated_sections["objective"],
                self._extract_clinical_findings(transcript, entities)
            )
        
        # Always update assessment and plan based on new information
        updated_sections["assessment"] = self._update_assessment(updated_sections, entities)
        updated_sections["plan"] = self._update_plan(updated_sections, entities)
        
        return updated_sections
    
    def _add_to_section(self, current_content: str, new_content: str) -> str:
        """Add new content to a section without duplication"""
        if not new_content.strip():
            return current_content
        
        if new_content in current_content:
            return current_content
        
        if current_content:
            return current_content + ". " + new_content
        else:
            return new_content
    
    def _extract_symptom_content(self, transcript: str, entities: List[MedicalEntity]) -> str:
        """Extract symptom-related content for subjective section"""
        symptoms = [e.text for e in entities if e.entity == "SYMPTOM"]
        if symptoms:
            return f"Reports {', '.join(symptoms)}"
        return ""
    
    def _extract_history_content(self, transcript: str, entities: List[MedicalEntity]) -> str:
        """Extract history-related content"""
        medications = [e.text for e in entities if e.entity == "MEDICATION"]
        conditions = [e.text for e in entities if e.entity in ["DIAGNOSIS", "CONDITION"]]
        
        parts = []
        if medications:
            parts.append(f"Current medications: {', '.join(medications)}")
        if conditions:
            parts.append(f"History of {', '.join(conditions)}")
        
        return ". ".join(parts)
    
    def _extract_clinical_findings(self, transcript: str, entities: List[MedicalEntity]) -> str:
        """Extract objective findings from clinical discussion"""
        # This would be enhanced with vital signs, exam findings, etc.
        return "Clinical discussion regarding patient condition."
    
    def _update_assessment(self, sections: Dict[str, str], entities: List[MedicalEntity]) -> str:
        """Update assessment based on current information"""
        symptoms = [e.text for e in entities if e.entity == "SYMPTOM"]
        if symptoms:
            return f"Assessment of {', '.join(symptoms[:2])}"
        return "Ongoing assessment"
    
    def _update_plan(self, sections: Dict[str, str], entities: List[MedicalEntity]) -> str:
        """Update plan based on current information"""
        medications = [e.text for e in entities if e.entity == "MEDICATION"]
        if medications:
            return f"Consider {medications[0]} management"
        return "Continue evaluation and management"

class EnhancedSOAPBuilder(ProgressiveSOAPBuilder):
    """SOAP builder that handles negation and temporal information"""
    
    def __init__(self):
        super().__init__()
        self.negation_processor = NegationTemporalProcessor()
    
    def _extract_symptom_content(self, transcript: str, entities: List[EnhancedMedicalEntity]) -> str:
        """Extract symptom-related content with negation and temporal context"""
        symptom_parts = []
        
        for entity in entities:
            if entity.entity == "SYMPTOM":
                if entity.negated:
                    symptom_parts.append(f"Denies {entity.text}")
                else:
                    symptom_desc = entity.text
                    # Add temporal context if available
                    if entity.duration:
                        symptom_desc += f" for {entity.duration}"
                    if entity.onset:
                        symptom_desc += f" since {entity.onset}"
                    symptom_parts.append(symptom_desc)
        
        if symptom_parts:
            return "Reports " + "; ".join(symptom_parts)
        return ""
    
    def _update_assessment(self, sections: Dict[str, str], entities: List[EnhancedMedicalEntity]) -> str:
        """Update assessment considering negation and temporal factors"""
        active_symptoms = []
        negated_symptoms = []
        
        for entity in entities:
            if entity.entity == "SYMPTOM":
                if entity.negated:
                    negated_symptoms.append(entity.text)
                else:
                    active_symptoms.append(entity.text)
        
        assessment_parts = []
        if active_symptoms:
            # Add temporal context to assessment
            temporal_symptoms = []
            for entity in entities:
                if entity.entity == "SYMPTOM" and not entity.negated:
                    if entity.duration or entity.onset:
                        temp_info = []
                        if entity.duration:
                            temp_info.append(f"duration: {entity.duration}")
                        if entity.normalized_onset:
                            temp_info.append(f"onset: {entity.normalized_onset}")
                        temporal_symptoms.append(f"{entity.text} ({', '.join(temp_info)})")
                    else:
                        temporal_symptoms.append(entity.text)
            
            if temporal_symptoms:
                assessment_parts.append(f"Active: {', '.join(temporal_symptoms)}")
            else:
                assessment_parts.append(f"Active: {', '.join(active_symptoms)}")
        
        if negated_symptoms:
            assessment_parts.append(f"Negated: {', '.join(negated_symptoms)}")
        
        return ". ".join(assessment_parts) if assessment_parts else "Ongoing assessment"
    
class RealTimeMedicalValidator:
    """Validates medical content in real-time"""
    
    def __init__(self):
        self.dangerous_combinations = [
            ("warfarin", "aspirin"),("lisinopril", "ibuprofen"), ("lisinopril", "naproxen"),("ace_inhibitor", "nsaid"),
            ("metformin", "alcohol"),("simvastatin", "grapefruit"), ("digoxin", "furosemide"),("levothyroxine", "calcium"),
            ("phenytoin", "warfarin"),("ace_inhibitor", "potassium_sparing_diuretics"),("ace_inhibitor", "lithium")
        ]
        
        self.red_flag_symptoms = [
            "chest pain", "shortness of breath", "severe headache",
            "uncontrolled bleeding", "loss of consciousness",
            "sudden weakness", "difficulty speaking", "severe abdominal pain"
        ]

        self.contraindications = {
            "beta_blockers": ["asthma", "copd"],
            "ace_inhibitors": ["pregnancy", "angioedema"],
            "nsaids": ["peptic ulcer", "kidney disease"],
            "statins": ["liver disease", "pregnancy"]
        }
    
    def _map_to_medication_class(self, medication: str) -> List[str]:
        """Map specific medications to their drug classes"""
        medication = medication.lower()
        classes = []
        
        if any(ace in medication for ace in ['lisinopril', 'enalapril', 'ramipril', 'ace inhibitor']):
            classes.append('ace_inhibitor')
        if any(arb in medication for arb in ['losartan', 'valsartan', 'arb']):
            classes.append('arb')
        if any(nsaid in medication for nsaid in ['ibuprofen', 'naproxen', 'nsaid']):
            classes.append('nsaid')
        if 'statin' in medication:
            classes.append('statin')
        
        return classes

    def check_ace_inhibitor_safety(self, medications: List[str], symptoms: List[str]) -> List[Dict[str, str]]:
        """Specific safety checks for ACE inhibitors"""
        alerts = []
        
        # Convert to lowercase for case-insensitive matching
        meds_lower = [med.lower() for med in medications]
        symptoms_lower = [symptom.lower() for symptom in symptoms]
        
        # Check if any ACE inhibitors are present
        ace_medications = []
        for med in medications:
            med_lower = med.lower()
             # Expanded ACE inhibitor detection
            ace_indicators = [
                'lisinopril', 'enalapril', 'ramipril', 'benazepril', 'quinapril',
                'ace inhibitor', 'ace', 'pril', 'lisiniprol', 'lucinipral'
            ]
            if any(ace in med_lower for ace in ace_indicators):
                ace_medications.append(med)
        
        if ace_medications:
            logger.info(f"🔍 ACE INHIBITOR CHECK: Found {ace_medications} with symptoms {symptoms_lower}")

            # Check for ACE inhibitor cough
            cough_terms = ['cough', 'coughing', 'dry cough', 'persistent cough', 'chronic cough']
            cough_detected = any(any(term in symptom for term in cough_terms) for symptom in symptoms_lower)
            if cough_detected:
                alerts.append({
                "type": "ace_inhibitor_cough",
                "message": f"Classic ACE inhibitor side effect: {', '.join(ace_medications)} is likely causing persistent dry cough",
                "severity": "high",
                "entities": ace_medications,
                "recommendation": "Strongly consider switching to ARB (losartan, valsartan) - cough typically resolves within 1-4 weeks after discontinuation"
            })
            # Enhanced dizziness detection
            dizziness_terms = ['dizziness', 'dizzy', 'lightheaded', 'orthostatic']
            dizziness_detected = any(any(term in symptom for term in dizziness_terms) for symptom in symptoms_lower)
            
            if dizziness_detected:
                alerts.append({
                    "type": "ace_inhibitor_hypotension", 
                    "message": f"ACE inhibitor may be causing dizziness/orthostatic hypotension",
                    "severity": "moderate",
                    "entities": ace_medications,
                    "recommendation": "Check blood pressure in sitting and standing positions, consider dose adjustment"
                })

            # Check for hyperkalemia risk symptoms
            hyperkalemia_symptoms = ['weakness', 'fatigue', 'palpitations', 'tired', 'dizziness']
            if any(symptom in ' '.join(symptoms_lower) for symptom in hyperkalemia_symptoms):
                alerts.append({
                    "type": "safety_alert", 
                    "message": f"ACE inhibitor use with these symptoms may indicate hyperkalemia",
                    "severity": "high",
                    "entities": ace_medications,
                    "recommendation": "Check potassium levels and renal function"
                })
            
            # Check for angioedema risk
            angioedema_terms = ['swelling', 'swollen', 'angioedema', 'face swelling', 'lip swelling']
            if any(term in ' '.join(symptoms_lower) for term in angioedema_terms):
                alerts.append({
                    "type": "safety_alert",
                    "message": "Possible angioedema with ACE inhibitor use",
                    "severity": "urgent",
                    "entities": ace_medications,
                    "recommendation": "Discontinue ACE inhibitor immediately and seek emergency care"
                })
        
        return alerts
    
    def validate_medication_safety(self, medications: List[str]) -> List[Dict[str, str]]:
        """Check for dangerous medication combinations with class mapping"""
        alerts = []
        meds_lower = [med.lower() for med in medications]
        
        # Check exact medication matches
        for med1, med2 in self.dangerous_combinations:
            if med1 in meds_lower and med2 in meds_lower:
                alerts.append({
                    "type": "drug_interaction",
                    "message": f"Potential dangerous interaction between {med1} and {med2}",
                    "severity": "high",
                    "entities": [med1, med2],
                    "recommendation": "Monitor closely or consider alternative medications"
                })
        
        # Check medication class interactions
        all_classes = []
        for med in medications:
            all_classes.extend(self._map_to_medication_class(med))
        
        for class1, class2 in self.dangerous_combinations:
            if class1 in all_classes and class2 in all_classes:
                involved_meds = [med for med in medications 
                            if class1 in self._map_to_medication_class(med) or 
                                class2 in self._map_to_medication_class(med)]
                alerts.append({
                    "type": "drug_interaction",
                    "message": f"Potential class interaction between {class1} and {class2}",
                    "severity": "moderate", 
                    "entities": involved_meds,
                    "recommendation": f"Monitor for {class1}-{class2} interactions"
                })
        
        return alerts
    
    def check_red_flags(self, symptoms: List[str], transcript: str) -> List[Dict[str, str]]:
        """Check for red flag symptoms requiring urgent attention"""
        alerts = []
        transcript_lower = transcript.lower()
        
        for red_flag in self.red_flag_symptoms:
            if red_flag in transcript_lower:
                alerts.append({
                    "type": "red_flag_symptom", 
                    "message": f"Red flag symptom detected: {red_flag}",
                    "severity": "urgent",
                    "entities": [red_flag],
                    "recommendation": "Requires immediate medical attention"
                })
        
        return alerts

    def check_contraindications(self, medication: str, conditions: List[str]) -> List[Dict[str, str]]:
        """Check medication contraindications with patient conditions"""
        alerts = []
        med_lower = medication.lower()
        
        for med_class, contra_conditions in self.contraindications.items():
            if med_class in med_lower:
                for condition in conditions:
                    if condition.lower() in contra_conditions:
                        alerts.append({
                            "type": "contraindication",
                            "message": f"Medication {medication} may be contraindicated with {condition}",
                            "severity": "high",
                            "entities": [medication, condition],
                            "recommendation": "Consult prescribing guidelines"
                        })
        
        return alerts

def generate_soap_note_rule_based(transcript: str, entities: List[MedicalEntity]) -> Dict[str, str]:
    """Enhanced rule-based SOAP note returning structured data"""
    symptoms = sorted(set(e.text for e in entities if e.entity == "SYMPTOM"))
    
    # Enhanced medication normalization
    medications = []
    for e in entities:
        if e.entity == "MEDICATION":
            med_name, confidence = normalize_medication_name(e.text)
            medications.append(med_name)
    medications = sorted(set(medications))
    
    transcript_lower = transcript.lower()
    
    # Extract blood pressure readings (your existing logic)
    bp_readings = []
    bp_pattern = r'\bblood pressure\s*(?:is|of|:)?\s*(\d{2,3})\s*\/\s*(\d{2,3})|\b(\d{2,3})\s*\/\s*(\d{2,3})\s*(?:mmhg|mmHg)'
    for match in re.finditer(bp_pattern, transcript_lower):
        if match.group(1) and match.group(2):
            bp_readings.append(f"{match.group(1)}/{match.group(2)}")
        elif match.group(3) and match.group(4):
            bp_readings.append(f"{match.group(3)}/{match.group(4)}")
    
    # Enhanced clinical context (your existing logic)
    has_hypertension = any(term in transcript_lower for term in ['blood pressure', 'hypertension', 'htn'])
    switching_medication = any(term in transcript_lower for term in ['switch', 'change medication', 'new medication'])
    ace_inhibitor_cough = 'ace inhibitor' in transcript_lower and 'cough' in symptoms
    
    # Build professional SOAP sections
    subjective = build_subjective_section(transcript, symptoms, medications, transcript_lower)
    objective = build_objective_section(bp_readings, medications, transcript_lower)
    assessment = build_assessment_section(symptoms, medications, transcript_lower, ace_inhibitor_cough)
    plan = build_plan_section(symptoms, medications, transcript_lower, switching_medication)
    
    # Return structured data instead of formatted text
    return {
        "subjective": subjective,  # Just the content: "- Patient presents for..."
        "objective": objective,    # Just the content: "- Blood Pressure: 8/82 mmHg"
        "assessment": assessment,  # Just the content: "- 1. Essential hypertension"
        "plan": plan,              # Just the content: "- Discontinue current ACE inhibitor..."
        "full_note": f"SUBJECTIVE:\n{subjective}\n\nOBJECTIVE:\n{objective}\n\nASSESSMENT:\n{assessment}\n\nPLAN:\n{plan}"
    }

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
    
    return "\n".join(f"- {part}" for part in parts)  # ← NO HEADER

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
    
    return "\n".join(f"- {part}" for part in parts)

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
    
    # FIX: Return empty string if no content, not a newline
    if not parts:
        return ""
    
    return "\n".join(f"- {part}" for part in parts)

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

    # FIX: Return empty string if no content, not a newline
    if not parts:
        return ""
    
    return "\n".join(f"- {part}" for part in parts)

# Main entity extraction function (unchanged)
def extract_medical_entities_sync(text: str) -> Tuple[List[MedicalEntity], str]:
    entities = []
    model_used = "keyword-fallback"
    logger.info(f"🔍 EXTRACTION DEBUG: Processing text: '{text[:100]}...'")
    
    global BIOBERT_MODEL, SPACY_MODEL

    pattern_entities = extract_medical_patterns(text)
    change_entities = extract_medication_changes(text)
    entities.extend(pattern_entities)
    entities.extend(change_entities)
    
    
    # PRIMARY: spaCy with enhanced patterns (SWITCHED TO PRIMARY)
    if SPACY_MODEL is not None:
        try:
            with _spacy_lock:
                doc = SPACY_MODEL(text)
            
            for ent in doc.ents:
                entity_type = map_spacy_label_to_medical(ent.label_)
                if entity_type != "OTHER":
                    # Normalize medication names for display
                    display_text = ent.text
                    if entity_type == "MEDICATION":
                        display_text = normalize_medication_name(ent.text)[0]
                    
                    entities.append(MedicalEntity(
                        entity=entity_type,
                        text=display_text,  # Use normalized name
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.9
                    ))
            
            if entities:
                entities = deduplicate_entities(entities)
                model_used = "spacy-medical"
                logger.info(f"spaCy extracted {len(entities)} entities")
                # DON'T return yet - continue to BioBERT for additional entities
                
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")
    
    # SECONDARY: BioBERT with lower confidence threshold (DEMOTED TO SECONDARY)
    if BIOBERT_MODEL is not None:
        try:
            with _biobert_lock:
                results = BIOBERT_MODEL(text)
            
            logger.info(f"BioBERT raw results: {len(results)} entities found")
            
            biobert_entities = []
            for entity in results:
                # LOWER confidence threshold from 0.6 to 0.3 for better recall
                if entity.get('score', 0) > 0.3:
                    entity_type = map_biobert_label_to_medical(
                        entity.get('entity_group', ''),
                        entity.get('word', '')
                    )
                    if entity_type != "OTHER":
                        biobert_entities.append(MedicalEntity(
                            entity=entity_type,
                            text=entity.get('word', ''),
                            start=entity.get('start', 0),
                            end=entity.get('end', 0),
                            confidence=float(entity.get('score', 0.7))
                        ))
                        logger.info(f"BioBERT found: {entity_type} - {entity.get('word', '')}")
            
            # Add BioBERT entities to the main list (don't replace)
            entities.extend(biobert_entities)
            
            if biobert_entities:
                logger.info(f"BioBERT added {len(biobert_entities)} additional entities")
                model_used = "spacy-medical+biobert"  # Updated model used
                
        except Exception as e:
            logger.warning(f"BioBERT extraction failed: {e}")
    
    # If we have entities from either model, return them
    if entities:
        entities = deduplicate_entities(entities)
        logger.info(f"Final ensemble extracted {len(entities)} entities")
        return entities, model_used
    
    # FALLBACK: Enhanced keyword extraction (only if both models failed)
    entities = extract_entities_keywords(text)
    # entities = filter_negated_entities(text, entities)
    model_used = "keyword-fallback"
    logger.info(f"Keyword fallback extracted {len(entities)} entities")
    
    return entities, model_used

def extract_medical_entities_with_negation_temporal(text: str) -> Tuple[List[EnhancedMedicalEntity], str]:
    """Extract medical entities with negation and temporal information"""
    base_entities, model_used = extract_medical_entities_sync(text)
    processor = NegationTemporalProcessor()
    
    enhanced_entities = []
    for entity in base_entities:
        # Detect negation
        negation_info = processor.detect_negation(text, entity.text, entity.start)
        
        # Extract temporal information
        temporal_info = processor.extract_temporal_info(text, entity.text, entity.start)
        
        # Create enhanced entity
        enhanced_entity = EnhancedMedicalEntity(
            **entity.dict(),
            negated=negation_info["negated"],
            negation_confidence=negation_info["confidence"],
            negation_phrase=negation_info.get("negation_phrase"),
            duration=temporal_info.get("duration"),
            normalized_duration=temporal_info.get("normalized_duration"),
            onset=temporal_info.get("onset"),
            normalized_onset=temporal_info.get("normalized_onset"),
            temporal_context=temporal_info.get("temporal_context")
        )
        enhanced_entities.append(enhanced_entity)
    
    return enhanced_entities, f"{model_used}+negation_temporal"

class HybridEntityExtractor:
    def __init__(self, clinicalbert_model):
        self.clinicalbert_model = clinicalbert_model
        self.keyword_extractor = None
        self.confidence_thresold = 0.7
        self.processing_pool = ThreadPoolExecutor(max_workers=2)
        self.clinicalbert_available = self.clinicalbert_model is not None
        if not self.clinicalbert_available:
            logger.warning("ClinicalBERT model is None - will use keyword fallback only")

    def _map_bert_label_to_medical(self, bert_label: str) -> str:
        label_mapping = {
            # Common ClinicalBERT labels - adjust based on your fine-tuned model
            "SYMPTOM": "SYMPTOM",
            "MEDICATION": "MEDICATION", 
            "DISEASE": "DIAGNOSIS",
            "CONDITION": "DIAGNOSIS",
            "PROCEDURE": "PROCEDURE",
            "LAB_TEST": "LAB_TEST",
            "BODY_PART": "BODY_PART",
            "DOSAGE": "MEDICATION",
            "FREQUENCY": "OTHER",
            "DURATION": "OTHER"
        }
        # return label_mapping.get(bert_label, "OTHER")
        return "OTHER"
    
    async def run_clinicalbert_extractor(self, text:str) -> List[EnhancedMedicalEntity]:
        # ✅ ADD: Check availability first
        if not self.clinicalbert_available:
            logger.debug("ClinicalBERT not available, skipping")
            return []
        try:
            loop = asyncio.get_event_loop()
            raw_entities = await loop.run_in_executor(
                self.processing_pool,
                lambda: self.clinicalbert_model(text)
            )
            print("RAW ENTRIES", raw_entities)
            entities = []
            for entity in raw_entities:
                entity_type = self._map_bert_label_to_medical(entity['entity_group'])

                if entity_type != "OTHER":
                    enhanced_entity = EnhancedMedicalEntity(
                        entity=entity_type,
                        text=entity['word'],
                        start=entity['start'],
                        end=entity['end'],
                        confidence=float(entity['score']),
                        negated=False,  # Will be handled by negation detector
                        negation_confidence=0.0,
                        duration=None,
                        onset=None,
                        normalized_duration=None,
                        normalized_onset=None,
                        temporal_context=None
                    )
                    entities.append(enhanced_entity)
            logger.debug(f"ClinicalBERT extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"ClinicalBERT entity extraction error: {e}")
            return []
        
    def _has_sufficient_entities(self, entities: List[EnhancedMedicalEntity]) -> bool:
        """Check if ClinicalBERT found enough high-confidence entities"""
        if not entities:
            return False
        
        high_confidence_entities = [
            e for e in entities 
            if e.confidence >= self.confidence_thresold
        ]
        return len(high_confidence_entities) >= 1 or len(entities) >= 3
    
    async def _extract_with_keyword_fallback(self, text: str) -> Tuple[List[EnhancedMedicalEntity], str]:
        """Use existing keyword-based extraction as fallback"""
        try:
            entities, model_used = extract_medical_entities_sync(text)

            enhanced_entities = []
            for entity in entities:
                if hasattr(entity, "negated"):
                    enhanced_entities.append(entity)
                else:
                    enhanced_entity = EnhancedMedicalEntity(
                        **entity.dict(),
                        negated=False,
                        negation_confidence=0.0,
                        negation_phrase=None,
                        duration=None,
                        normalized_duration=None,
                        onset=None,
                        normalized_onset=None,
                        temporal_context=None
                    )
                    enhanced_entities.append(enhanced_entity)
            logger.info(f"Keyword fallback extracted {len(enhanced_entities)} entities")
            return enhanced_entities, f"keyword_{model_used}"
        
        except Exception as e:
            logger.error(f"Keyword fallback extraction error: {e}")
            return [], "keyword_error"
        
    async def extract_entities(self, text: str) -> Tuple[List[EnhancedMedicalEntity], str]:
        if not text.strip():
            return [], "no_text"
        
        clinical_entities = await self.run_clinicalbert_extractor(text)
        if self._has_sufficient_entities(clinical_entities):
            logger.info(f"ClinicalBERT found {len(clinical_entities)} entities")
            return clinical_entities, "clinical_bert"

        return await self._extract_with_keyword_fallback(text)
            
class RealTimeMedicalProcessor:
    """Optimized real-time medical audio processor"""
    
    def __init__(self, whisper_model, medical_llm=None, pyannote_pipeline=None,
                biobert_model=None, clinicalbert_model=None, spacy_model=None):
        self.whisper_model = whisper_model
        self.medical_llm = medical_llm
        self.pyannote_pipeline = pyannote_pipeline
        self.biobert_model = biobert_model
        self.clinicalbert_model = clinicalbert_model
        self.spacy_model = spacy_model
        self.soap_builder = EnhancedSOAPBuilder()
        
        self.hybrid_extractor = HybridEntityExtractor(self.clinicalbert_model)

        self.conversation_analyzer = MedicalConversationAnalyzer()
        self.negation_processor = NegationTemporalProcessor()  # ← ADDED
        
        # Real-time processing with optimizations
        self.audio_processor = AudioStreamProcessor()
        self.processing_pool = ThreadPoolExecutor(max_workers=2)  # Reduced workers
        
        # Session management
        self.active_sessions: Dict[str, PatientContext] = {}
        self.session_lock = Lock()
        
        # Performance tracking
        self.last_processing_time = 0
        self.processing_interval = 5.0  # Process every 5 seconds
        
        logger.info("Enhanced RealTimeMedicalProcessor initialized with negation & temporal support")    
    
    async def _extract_medical_entities_enhanced(self, transcript: str) -> List[EnhancedMedicalEntity]:
        """Extract medical entities with negation and temporal information"""
        if not transcript.strip():
            return []
        
        try:
            entities, model_used = await self.hybrid_extractor.extract_entities(transcript)

            # Log the extraction method
            logger.info(f"🔍 HYBRID EXTRACTION: {len(entities)} entities using {model_used}")

            if model_used == "clinical_bert" and entities:
                clinical_entities = [e for e in entities if e.confidence >= 0.7]
                logger.info(f"🔍 CLINICALBERT: {len(clinical_entities)} high-confidence entities")
                for entity in clinical_entities[:3]:  # Log first 3
                    logger.info(f"🔍   - {entity.entity}: '{entity.text}' (conf: {entity.confidence:.2f})")
            
            return entities
        except Exception as e:
            logger.error(f"Hybrid entity extraction error: {e}")
            return await self._fallback_entity_extraction(transcript)

    async def _fallback_entity_extraction(self, transcript: str) -> List[EnhancedMedicalEntity]:
            try:
                loop = asyncio.get_event_loop()
                entities, model_used = await loop.run_in_executor(
                    self.processing_pool,
                    extract_medical_entities_with_negation_temporal, transcript
                )

                logger.info(f"🔍 FALLBACK EXTRACTION: {len(entities)} entities using {model_used}")
                # 🔍 ADD THIS DEBUG
                print(f"🔍 ENTITY EXTRACTION DEBUG: {len(entities)} entities found using {model_used}")
                if entities:
                    print(f"🔍 FIRST ENTITY TYPE: {type(entities[0]).__name__}")
                    print(f"🔍 HAS NEGATED ATTR: {hasattr(entities[0], 'negated')}")
                
                return entities
            except Exception as e:
                logger.error(f"Enhanced entity extraction error: {e}")
                return []

    def _format_current_soap(self, sections: Dict[str, str]) -> str:
        """Format the current SOAP state and remove empty lines"""
        soap_note = ""
        for section, content in sections.items():
            if content.strip():
                # Remove leading/trailing whitespace and empty lines
                clean_content = content.strip()
                # Remove any leading empty lines
                clean_content = '\n'.join(line for line in clean_content.split('\n') if line.strip())
                
                soap_note += f"{section.upper()}:\n{clean_content}\n\n"
        return soap_note.strip()

    def generate_enhanced_soap_note(self, transcript: str, entities: List[EnhancedMedicalEntity]) -> Dict[str, str]:
        """SOAP generation that actually uses negation/temporal info"""
        
         # 🔍 ADD THIS DEBUG BLOCK
        print(f"🔍 NEGATION DEBUG: Processing {len(entities)} entities for transcript: '{transcript[:100]}...'")
        negated_count = 0
        for i, entity in enumerate(entities):
            if hasattr(entity, 'negated'):
                if entity.negated:
                    print(f"   🚫 NEGATED: '{entity.text}' (confidence: {entity.negation_confidence}, phrase: {entity.negation_phrase})")
                    negated_count += 1
                else:
                    print(f"   ✅ ACTIVE: '{entity.text}'")
            else:
                print(f"   ⚠️ BASIC: '{entity.text}' (no negated attribute)")

        print(f"🔍 NEGATION SUMMARY: {negated_count}/{len(entities)} entities are negated")
        
        # Separate negated vs active symptoms
        active_symptoms = []
        negated_symptoms = []
        
        for entity in entities:
            if entity.entity == "SYMPTOM":
                if hasattr(entity, 'negated') and entity.negated:
                    negated_symptoms.append(entity.text)
                else:
                    symptom_desc = entity.text
                    if hasattr(entity, 'duration') and entity.duration:
                        symptom_desc += f" for {entity.duration}"
                    if hasattr(entity, 'onset') and entity.onset:
                        symptom_desc += f" since {entity.onset}"
                    active_symptoms.append(symptom_desc)
        
        # Convert to basic entities for existing rule-based generator
        basic_entities = [MedicalEntity(**entity.dict()) for entity in entities]
        soap_data = generate_soap_note_rule_based(transcript, basic_entities)
        
        # Enhance the subjective section with negation info
        if negated_symptoms:
            enhanced_subjective = soap_data["subjective"] + f"\n- Denies: {', '.join(negated_symptoms)}"
            soap_data["subjective"] = enhanced_subjective
            # Update the full note
            soap_data["full_note"] = f"SUBJECTIVE:\n{enhanced_subjective}\n\nOBJECTIVE:\n{soap_data['objective']}\n\nASSESSMENT:\n{soap_data['assessment']}\n\nPLAN:\n{soap_data['plan']}"
        
        return soap_data

    def _ensure_enhanced_entities(self, entities: List[Any]) -> List[EnhancedMedicalEntity]:
        """Ensure all entities are EnhancedMedicalEntity type"""
        enhanced_entities = []
        for entity in entities:
            if hasattr(entity, 'negated'):
                # Already enhanced
                enhanced_entities.append(entity)
            else:
                # Convert basic MedicalEntity to EnhancedMedicalEntity
                enhanced_entity = EnhancedMedicalEntity(
                    **entity.dict(),
                    negated=False,
                    negation_confidence=0.0,
                    negation_phrase=None,
                    duration=None,
                    normalized_duration=None,
                    onset=None,
                    normalized_onset=None,
                    temporal_context=None
                )
                enhanced_entities.append(enhanced_entity)
        return enhanced_entities

    async def _update_progressive_soap(self, patient_context: PatientContext, transcript: str, entities: List[EnhancedMedicalEntity]):
        """Update SOAP note progressively using structured data"""
        
        # Combine all transcripts so far for context
        all_transcripts = " ".join(patient_context.conversation_history + [transcript])
        all_entities = self._ensure_enhanced_entities(patient_context.extracted_entities)
        
        # basic_entities = [MedicalEntity(**entity.dict()) for entity in entities]        
        print(f"🔍 DEBUG: {len(patient_context.conversation_history)} transcripts, {len(entities)} new entities")
        print(f"🔍 DEBUG: Total stored entities: {len(all_entities)}")
        
        # Use your sophisticated rule-based generator (now returns dict)
        
        soap_data = self.generate_enhanced_soap_note(all_transcripts, all_entities)
        
        print(f"🔍 DEBUG: Generated SOAP sections: {list(soap_data.keys())}")
        print(f"🔍 DEBUG: SOAP preview: {soap_data['full_note'][:200]}...")
        
        # Update patient context with structured sections
        patient_context.soap_note_sections = {
            "subjective": soap_data["subjective"],
            "objective": soap_data["objective"],
            "assessment": soap_data["assessment"], 
            "plan": soap_data["plan"]
        }
        
        return soap_data["full_note"]  # Return formatted version for display

    def _parse_soap_to_sections(self, soap_note: str) -> Dict[str, str]:
        """Parse the rule-based SOAP note back into sections for progressive updates"""
        sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
        
        current_section = None
        for line in soap_note.split('\n'):
            line = line.strip()
            if line.startswith('SUBJECTIVE:'):
                current_section = "subjective"
            elif line.startswith('OBJECTIVE:'):
                current_section = "objective"
            elif line.startswith('ASSESSMENT:'):
                current_section = "assessment"
            elif line.startswith('PLAN:'):
                current_section = "plan"
            elif current_section and line and not line.startswith('-'):
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line
        
        return sections
    
    async def _extract_medical_entities(self, transcript: str) -> List[MedicalEntity]:
        """Extract medical entities from transcript"""
        if not transcript.strip():
            return []
        
        try:
            loop = asyncio.get_event_loop()
            entities, _ = await loop.run_in_executor(
                self.processing_pool,
                extract_medical_entities_sync, transcript
            )
            return entities
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []
    
    async def process_realtime_stream(self, session_id: str, 
                               audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[RealtimeResult, None]:
        """
        Debug version to track final processing
        """
        logger.info(f"🚀 STARTING processing for session: {session_id}")
        
        patient_context = PatientContext(session_id=session_id)
        
        with self.session_lock:
            self.active_sessions[session_id] = patient_context
        
        try:
            buffer_count = 0
            
            async for audio_chunk in audio_stream:
                buffer_count += 1
                self.audio_processor.add_audio_to_buffer(patient_context.audio_buffer, audio_chunk)
                
                # Process every 50 chunks to avoid backlog
                if buffer_count % 50 == 0:
                    logger.info(f"🔄 Processing chunk {buffer_count}, buffer size: {len(patient_context.audio_buffer)}")
                    async for result in self._process_audio_buffer(patient_context, is_final=False):
                        yield result
                    # Clear buffer after processing to prevent duplicates
                    patient_context.audio_buffer.clear()
            
            # CRITICAL FIX: The stream has ended - process final buffer immediately
            logger.info(f"🎬 STREAM ENDED for {session_id}. Buffer count: {buffer_count}, Final buffer size: {len(patient_context.audio_buffer)}")
            
            if patient_context.audio_buffer:
                logger.info(f"📦 Processing final buffer ({len(patient_context.audio_buffer)} bytes)")
                async for result in self._process_audio_buffer(patient_context, is_final=True):
                    yield result
                patient_context.audio_buffer.clear()
            else:
                logger.info("📦 No final buffer to process")
            
            # Force final results regardless of buffer
            if patient_context.conversation_history:
                transcript_count = len(patient_context.conversation_history)
                total_chars = sum(len(t) for t in patient_context.conversation_history)
                logger.info(f"📄 Generating final results from {transcript_count} transcripts ({total_chars} chars)")
                
                full_transcript = " ".join(patient_context.conversation_history)
                print("full_transcript", full_transcript)
            else:
                logger.warning("❌ No conversation history for final processing")
                yield RealtimeResult(
                    type="error",
                    data={"message": "No audio was transcribed successfully"},
                    session_id=session_id,
                    is_partial=False
                )
                    
        except Exception as e:
            logger.error(f"💥 Stream processing error: {e}", exc_info=True)
            yield RealtimeResult(
                type="error",
                data={"message": f"Processing error: {str(e)}"},
                session_id=session_id,
                is_partial=False
            )
        finally:
            with self.session_lock:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
            logger.info(f"🧹 Cleaned up session: {session_id}")

    def _update_patient_context_enhanced(self, patient_context: PatientContext, entities: List[EnhancedMedicalEntity]):
        """Update patient context with enhanced findings including negation and temporal info"""
        for entity in entities:
            if entity.entity == "SYMPTOM":
                if not entity.negated:  # Only add non-negated symptoms to current symptoms
                    if entity.text not in patient_context.current_symptoms:
                        patient_context.current_symptoms.append(entity.text)
            elif entity.entity == "MEDICATION":
                med_name = normalize_medication_name(entity.text)[0]
                if med_name not in patient_context.medications:
                    patient_context.medications.append(med_name)
            elif entity.entity == "DIAGNOSIS":
                if entity.text not in patient_context.medical_history:
                    patient_context.medical_history.append(entity.text)

    async def _transcribe_audio(self, wav_data: bytes) -> str:
        """Transcribe WAV audio data"""
        if not wav_data:
            return ""
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(wav_data)
                audio_path = tmp_file.name
            
            try:
                loop = asyncio.get_event_loop()
                transcript_result = await loop.run_in_executor(
                    self.processing_pool,
                    lambda: self.whisper_model.transcribe(audio_path)
                )
                return transcript_result.get("text", "").strip()
            finally:
                os.unlink(audio_path)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
        
    def _check_medication_contraindications(self, medication: str, conditions: List[str], symptoms: List[str]) -> List[Dict]:
        """Check individual medication against patient conditions and symptoms"""
        alerts = []
        validator = RealTimeMedicalValidator()
        
        # Check against medical conditions
        condition_alerts = validator.check_contraindications(medication, conditions)
        alerts.extend(condition_alerts)
        
        # Check against current symptoms
        symptom_alerts = validator.check_contraindications(medication, symptoms)
        alerts.extend(symptom_alerts)
        
        return alerts

    def _check_condition_contraindications(self, medications: List[str], conditions: List[str]) -> List[Dict]:
        """Check all medications against all conditions"""
        alerts = []
        validator = RealTimeMedicalValidator()
        
        for medication in medications:
            condition_alerts = validator.check_contraindications(medication, conditions)
            alerts.extend(condition_alerts)
        
        return alerts
    
    async def _perform_medical_validation(self, patient_context: PatientContext, 
                                    new_entities: List[MedicalEntity], 
                                    transcript: str) -> List[RealtimeResult]:
        """Perform real-time medical safety validation"""
        alerts = []
        validator = RealTimeMedicalValidator()
        
        try:
            # Extract medications from new entities
            new_medications = [e.text for e in new_entities if e.entity == "MEDICATION"]
            new_symptoms = [e.text for e in new_entities if e.entity == "SYMPTOM"]
            
            # 1. Check for dangerous medication combinations
            if new_medications and patient_context.medications:
                all_medications = patient_context.medications + new_medications
                medication_alerts = validator.validate_medication_safety(all_medications)
                alerts.extend(self._convert_to_realtime_results(medication_alerts, patient_context.session_id))
            
            # 2. Check for red flag symptoms
            if new_symptoms:
                symptom_alerts = validator.check_red_flags(new_symptoms, transcript)
                alerts.extend(self._convert_to_realtime_results(symptom_alerts, patient_context.session_id))
            
            # 3. Check individual new medications against existing conditions
            for medication in new_medications:
                medication_alerts = self._check_medication_contraindications(
                    medication, patient_context.medical_history, patient_context.current_symptoms
                )
                alerts.extend(self._convert_to_realtime_results(medication_alerts, patient_context.session_id))
            
            # 🆕 ADD ACE INHIBITOR SPECIFIC CHECKS HERE
            ace_alerts = validator.check_ace_inhibitor_safety(
                patient_context.medications + new_medications,
                patient_context.current_symptoms + new_symptoms
            )
            alerts.extend(self._convert_to_realtime_results(ace_alerts, patient_context.session_id))
        
        except Exception as e:
            logger.error(f"Medical validation error: {e}")
        
        return alerts

    async def _perform_final_validation(self, patient_context: PatientContext) -> List[RealtimeResult]:
        """Perform comprehensive final validation"""
        alerts = []
        validator = RealTimeMedicalValidator()
        
        try:
            # Final check for all accumulated medications
            if patient_context.medications:
                medication_alerts = validator.validate_medication_safety(patient_context.medications)
                alerts.extend(self._convert_to_realtime_results(medication_alerts, patient_context.session_id))
            
            # Final check for red flags in entire conversation
            full_transcript = " ".join(patient_context.conversation_history)
            symptom_alerts = validator.check_red_flags(patient_context.current_symptoms, full_transcript)
            alerts.extend(self._convert_to_realtime_results(symptom_alerts, patient_context.session_id))
            
            # Check for critical medication-condition interactions
            condition_alerts = self._check_condition_contraindications(
                patient_context.medications, patient_context.medical_history
            )
            alerts.extend(self._convert_to_realtime_results(condition_alerts, patient_context.session_id))
            
            # 🆕 FINAL ACE INHIBITOR CHECK
            ace_alerts = validator.check_ace_inhibitor_safety(
                patient_context.medications,
                patient_context.current_symptoms
            )
            alerts.extend(self._convert_to_realtime_results(ace_alerts, patient_context.session_id))
            
        except Exception as e:
            logger.error(f"Final validation error: {e}")
        
        return alerts

    def _convert_to_realtime_results(self, alerts: List[Dict], session_id: str) -> List[RealtimeResult]:
        """Convert validator alerts to RealtimeResult objects"""
        results = []
        for alert in alerts:
            results.append(RealtimeResult(
                type="medical_alert",
                data={
                    "severity": alert.get("severity", "moderate"),
                    "alert_type": alert.get("type", "safety_alert"),
                    "message": alert.get("message", "Medical safety alert"),
                    "entities_involved": alert.get("entities", []),
                    "recommendation": alert.get("recommendation", "Please review"),
                    "timestamp": time.time()
                },
                session_id=session_id,
                is_partial=False,
                confidence=0.9
            ))
        return results
    
    async def _process_audio_buffer(self, patient_context: PatientContext, 
                          is_final: bool) -> AsyncGenerator[RealtimeResult, None]:
        """Process audio buffer with progressive SOAP building - DeepScribe style"""
        try:
            # Convert buffer to WAV format
            wav_data = self.audio_processor.create_wav_from_buffer(patient_context.audio_buffer)
            if not wav_data or len(wav_data) < 1000:
                if is_final:
                    # Even with small audio, send current SOAP state immediately
                    final_soap = self._format_current_soap(patient_context.soap_note_sections)
                    yield RealtimeResult(
                        type="soap_note_complete",
                        data={
                            "content": final_soap,
                            "is_complete": True,
                            "is_progressive": True,
                            "session_duration": time.time() - patient_context.start_time
                        },
                        session_id=patient_context.session_id,
                        is_partial=False
                    )
                return
            
            # Transcribe audio
            transcript = await self._transcribe_audio(wav_data)
            if not transcript or not transcript.strip():
                if is_final:
                    # Send current SOAP state even if no new transcript
                    final_soap = self._format_current_soap(patient_context.soap_note_sections)
                    yield RealtimeResult(
                        type="soap_note_complete",
                        data={
                            "content": final_soap,
                            "is_complete": True,
                            "is_progressive": True,
                            "session_duration": time.time() - patient_context.start_time
                        },
                        session_id=patient_context.session_id,
                        is_partial=False
                    )
                return
            
            # Check for duplicate transcript
            if self._is_duplicate_transcript(patient_context, transcript):
                logger.debug("Duplicate transcript, skipping processing")
                if is_final:
                    # Still send final SOAP on duplicate if this is final
                    final_soap = self._format_current_soap(patient_context.soap_note_sections)
                    yield RealtimeResult(
                        type="soap_note_complete",
                        data={
                            "content": final_soap,
                            "is_complete": True,
                            "is_progressive": True,
                            "session_duration": time.time() - patient_context.start_time
                        },
                        session_id=patient_context.session_id,
                        is_partial=False
                    )
                return
            
            logger.info(f"Processing: '{transcript[:50]}...'")
            
            # Update conversation history
            patient_context.conversation_history.append(transcript)
            
            # Yield transcript immediately
            yield RealtimeResult(
                type="transcript",
                data={
                    "text": transcript,
                    "full_transcript": " ".join(patient_context.conversation_history),
                    "is_partial": not is_final
                },
                session_id=patient_context.session_id,
                is_partial=not is_final
            )
            
            # Extract entities (fast operation)
            entities = await self._extract_medical_entities_enhanced(transcript)  # ← CHANGED
            # 🔍 ADD THIS DEBUG RIGHT AFTER ENTITY EXTRACTION
            print(f"🔍 PROCESSING DEBUG: Extracted {len(entities)} entities for: '{transcript[:50]}...'")
            if entities:
                print(f"🔍 ENTITY TYPES IN BUFFER: {[type(e).__name__ for e in entities]}")
                negated_entities = [e for e in entities if hasattr(e, 'negated') and e.negated]
                print(f"🔍 NEGATED IN THIS CHUNK: {len(negated_entities)}")

            if entities:
                # STORE ENTITIES WITH DEDUPLICATION
                patient_context.extracted_entities.extend(entities)
                patient_context.extracted_entities = deduplicate_entities(patient_context.extracted_entities)

                 # 🔍 ADD DEBUG FOR STORED ENTITIES
                print(f"🔍 STORED ENTITIES COUNT: {len(patient_context.extracted_entities)}")
                stored_negated = [e for e in patient_context.extracted_entities if hasattr(e, 'negated') and e.negated]
                print(f"🔍 TOTAL NEGATED STORED: {len(stored_negated)}")
                
                self._update_patient_context_enhanced(patient_context, entities)  # ← CHANGED
                
                yield RealtimeResult(
                    type="entities",
                    data={
                        "entities": [entity.dict() for entity in entities],
                        "new_entities": len(entities),
                        "negated_entities": len([e for e in entities if getattr(e, 'negated', False)]),
                        "temporal_entities": len([e for e in entities if getattr(e, 'duration', None) or getattr(e, 'onset', None)])
                    },
                    session_id=patient_context.session_id,
                    is_partial=not is_final
                )
            
            # 🚨 CRITICAL: PERFORM REAL-TIME MEDICAL VALIDATION
            alerts = await self._perform_medical_validation(patient_context, entities, transcript)
            for alert in alerts:
                yield alert
            
            # DEEPSCRIBE STRATEGY: Progressive SOAP building
            current_soap = await self._update_progressive_soap(patient_context, transcript, entities)
            
            if current_soap:
                yield RealtimeResult(
                    type="soap_update", 
                    data={
                        "current_sections": patient_context.soap_note_sections,  # Already structured!
                        "soap_note": current_soap,
                        "is_progressive": True,
                        "new_content": transcript[:100] + "..." if len(transcript) > 100 else transcript
                    },
                    session_id=patient_context.session_id,
                    is_partial=not is_final
                )
            
            # DEEPSCRIBE STRATEGY: If this is final processing, send SOAP immediately
            if is_final:
                final_alerts = await self._perform_final_validation(patient_context)
                for alert in final_alerts:
                    yield alert
                # Use the CURRENT_SOAP that was just generated by rule-based generator, not the basic sections
                final_soap = current_soap  # ← USE THIS instead of _format_current_soap
                
                # PRINT TO SERVER CONSOLE FOR DEBUGGING
                print(f"\n" + "="*80)
                print("🎉 FINAL SOAP NOTE - PROGRESSIVE BUILD (DeepScribe Strategy):")
                print("="*80)
                print(final_soap)  # This will now show the actual SOAP content
                print("="*80)
                print(f"📊 Stats: {len(final_soap)} chars, {len(patient_context.conversation_history)} transcripts")
                print(f"⏱️  Session duration: {time.time() - patient_context.start_time:.2f}s")
                print("="*80 + "\n")
                
                yield RealtimeResult(
                    type="soap_note_complete",
                    data={
                        "content": final_soap,  # Send the actual SOAP content
                        "is_complete": True,
                        "is_progressive": True,
                        "session_duration": time.time() - patient_context.start_time,
                        "transcript_count": len(patient_context.conversation_history),
                        "entities_count": len(patient_context.extracted_entities)
                    },
                    session_id=patient_context.session_id,
                    is_partial=False
                )
                logger.info(f"✅ Final SOAP note sent immediately - WebSocket ready to close")
                            
        except Exception as e:
            logger.error(f"Audio buffer processing error: {e}")
            
            # Even on error, try to send current SOAP state if this is final
            if is_final:
                try:
                    final_soap = self._format_current_soap(patient_context.soap_note_sections)
                    yield RealtimeResult(
                        type="soap_note_complete",
                        data={
                            "content": final_soap,
                            "is_complete": True,
                            "is_progressive": True,
                            "session_duration": time.time() - patient_context.start_time,
                            "error_note": f"Completed with processing error: {str(e)}"
                        },
                        session_id=patient_context.session_id,
                        is_partial=False
                    )
                except Exception as final_error:
                    logger.error(f"Even final SOAP sending failed: {final_error}")
            
            yield RealtimeResult(
                type="error",
                data={"message": f"Processing error: {str(e)}"},
                session_id=patient_context.session_id,
                is_partial=not is_final
            )
    
    async def _is_valid_audio_data(self, wav_data: bytes) -> bool:
        """Quick validation of audio data"""
        return len(wav_data) > 1000 and wav_data.startswith(b'RIFF')
    
    def _is_duplicate_transcript(self, patient_context: PatientContext, transcript: str) -> bool:
        """Check if transcript is duplicate of recent content"""
        if not patient_context.conversation_history:
            return False
        
        # Compare with last transcript
        last_transcript = patient_context.conversation_history[-1] if patient_context.conversation_history else ""
        similarity_threshold = 0.8
        
        # Simple similarity check
        words_current = set(transcript.lower().split())
        words_previous = set(last_transcript.lower().split())
        
        if words_current and words_previous:
            common_words = words_current.intersection(words_previous)
            similarity = len(common_words) / max(len(words_current), len(words_previous))
            return similarity > similarity_threshold
        
        return False
    
    async def _extract_medical_entities_fast(self, transcript: str) -> List[MedicalEntity]:
        """Fast entity extraction - skip slow models if possible"""
        if not transcript.strip():
            return []
        
        try:
            # Try fast keyword extraction first
            keyword_entities = extract_entities_keywords(transcript)
            if keyword_entities:
                return deduplicate_entities(keyword_entities)
            
            # Fall back to full extraction with timeout
            loop = asyncio.get_event_loop()
            entities, _ = await asyncio.wait_for(
                loop.run_in_executor(self.processing_pool, extract_medical_entities_sync, transcript),
                timeout=5.0
            )
            return entities
            
        except asyncio.TimeoutError:
            logger.warning("Entity extraction timeout, returning empty")
            return []
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []
    
    async def _generate_final_results_fast(self, patient_context: PatientContext, 
                                full_transcript: str) -> AsyncGenerator[RealtimeResult, None]:
        """Fast final results generation with minimal processing"""
        logger.info(f"🚀 FAST FINAL: Generating quick results for {len(full_transcript)} chars")
        print("🔔 DEBUG: _generate_final_results METHOD ENTERED!")
        print(f"🔔 DEBUG: Session {patient_context.session_id}")
        print(f"🔔 DEBUG: Transcript length: {len(full_transcript)}")
        try:
            # Send immediate progress
            yield RealtimeResult(
                type="progress",
                data={"message": "Finalizing results..."},
                session_id=patient_context.session_id,
                is_partial=False
            )
            
            final_entities = await self._extract_medical_entities_fast(full_transcript)

            # Quick entity extraction (skip slow models)
            logger.info("🔄 Quick entity extraction...")
            simple_entities = []
            try:
                # Use only fast keyword extraction for final processing
                simple_entities = await asyncio.wait_for(
                    self._extract_medical_entities_fast(full_transcript),
                    timeout=5.0
                )
                logger.info(f"✅ Quick entities found: {len(simple_entities)}")
            except asyncio.TimeoutError:
                logger.warning("Quick entity extraction timeout")
            
            # Generate simple SOAP note quickly
            logger.info("🔄 Generating quick SOAP note...")
            soap_note = ""
            try:
                # Use rule-based generation only (faster than LLM)
                soap_note = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.processing_pool,
                        self._generate_quick_soap_note, full_transcript, simple_entities
                    ),
                    timeout=10.0
                )
                logger.info("✅ Quick SOAP note generated")
            except asyncio.TimeoutError:
                soap_note = "Summary: Medical conversation processed. Full SOAP note generation timed out."
                logger.warning("SOAP note generation timeout, using fallback")
            
            # Calculate session duration
            session_duration = time.time() - patient_context.start_time
            
            logger.info(f"🎉 FAST FINAL COMPLETE: {len(soap_note)} chars, {session_duration:.2f}s")

            print(f"\n" + "="*80)
            print("🎉 FINAL SOAP NOTE GENERATED - COMPLETE OUTPUT:")
            print("="*80)
            print(soap_note)  # This prints the ENTIRE SOAP note
            print("="*80)
            print(f"📊 Stats: {len(soap_note)} chars, {len(final_entities)} entities, {session_duration:.2f}s duration")
            print("="*80 + "\n")
            # Yield final results
            yield RealtimeResult(
                type="soap_note_complete",
                data={
                    "content": soap_note,
                    "is_complete": True,
                    "entities_found": len(simple_entities),
                    "model_used": "quick_generation",
                    "session_duration": session_duration,
                    "transcript_length": len(full_transcript),
                    "note": "Quick generation completed successfully"
                },
                session_id=patient_context.session_id,
                is_partial=False
            )
            
        except Exception as e:
            logger.error(f"💥 Fast final results error: {e}")
            yield RealtimeResult(
                type="error", 
                data={"message": f"Final processing error: {str(e)}"},
                session_id=patient_context.session_id,
                is_partial=False
            )

    def _generate_quick_soap_note(self, transcript: str, entities: List[MedicalEntity]) -> str:
        """Generate a quick SOAP note without LLM"""
        try:
            # Extract basic information using simple rules
            symptoms = [e.text for e in entities if e.entity == "SYMPTOM"]
            medications = [e.text for e in entities if e.entity == "MEDICATION"]
            diagnoses = [e.text for e in entities if e.entity == "DIAGNOSIS"]
            
            soap_note = f"""SOAP NOTE - QUICK GENERATION

    SUBJECTIVE:
    Patient reported: {transcript[:500]}...

    Key symptoms: {', '.join(symptoms) if symptoms else 'None identified'}

    OBJECTIVE:
    Medications discussed: {', '.join(medications) if medications else 'None identified'}

    ASSESSMENT:
    Potential conditions: {', '.join(diagnoses) if diagnoses else 'Discussion of symptoms and treatment'}

    PLAN:
    Follow up as discussed. Consider formal SOAP note generation for detailed analysis.

    NOTE: This is an automatically generated quick summary. For detailed analysis, process with full medical LLM.
    """
            return soap_note
        except Exception as e:
            return f"Quick SOAP note generation failed: {str(e)}"

class EnhancedMedicalProcessor(RealTimeMedicalProcessor):
    """Enhanced medical processor with negation and temporal capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soap_builder = EnhancedSOAPBuilder()
        self.negation_processor = NegationTemporalProcessor()
    
    async def _extract_medical_entities(self, transcript: str) -> List[EnhancedMedicalEntity]:
        """Extract medical entities with negation and temporal information"""
        if not transcript.strip():
            return []
        
        try:
            loop = asyncio.get_event_loop()
            entities, _ = await loop.run_in_executor(
                self.processing_pool,
                extract_medical_entities_with_negation_temporal, transcript
            )
            return entities
        except Exception as e:
            logger.error(f"Enhanced entity extraction error: {e}")
            return []

# Keyword-based entity extraction (unchanged)
def extract_entities_keywords(text: str) -> List[MedicalEntity]:
    """Enhanced keyword extraction with partial matching"""
    entities = []
    text_lower = text.lower()
    matched_positions = set()

    symptom_patterns = {
        'cough': r'\b(cough|coughing|dry cough|persistent cough|chronic cough)\b',
        'dizziness': r'\b(dizziness|dizzy|lightheaded|vertigo)\b', 
        'tired': r'\b(tired|fatigue|exhausted|weakness)\b',
        'headache': r'\b(headache|head pain|migraine)\b',
        'shortness of breath': r'\b(shortness of breath|sob|difficulty breathing|breathless)\b',
        'nausea': r'\b(nausea|nauseous|sick to stomach)\b',
        'chest pain': r'\b(chest pain|chest discomfort)\b'
    }
    
    # Check for medication synonyms first
    for misspelling, canonical in MEDICATION_SYNONYMS.items():
        pattern = r"\b" + re.escape(misspelling) + r"\b"
        for match in re.finditer(pattern, text_lower):
            start, end = match.start(), match.end()
            position_key = (start, end)
            if position_key not in matched_positions:
                entities.append(MedicalEntity(
                    entity="MEDICATION",
                    text=canonical,  # Use canonical name
                    start=start,
                    end=end,
                    confidence=0.85  # High confidence for known synonyms
                ))
                matched_positions.add(position_key)

    # Enhanced symptom detection
    for symptom, pattern in symptom_patterns.items():
        for match in re.finditer(pattern, text_lower):
            start, end = match.start(), match.end()
            position_key = (start, end)
            if position_key not in matched_positions:
                entities.append(MedicalEntity(
                    entity="SYMPTOM",
                    text=symptom,
                    start=start,
                    end=end,
                    confidence=0.8
                ))
                matched_positions.add(position_key)
    
    # Then check regular medical keywords
    for entity_type, keywords in MEDICAL_KEYWORDS.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            for match in re.finditer(pattern, text_lower):
                start, end = match.start(), match.end()
                
                position_key = (start, end)
                if position_key not in matched_positions:
                    entities.append(MedicalEntity(
                        entity=entity_type,
                        text=text[start:end],
                        start=start,
                        end=end,
                        confidence=0.8
                    ))
                    matched_positions.add(position_key)
                    
    logger.info(f"🔍 KEYWORD EXTRACTION: Found {len(entities)} entities")
    return entities

def deduplicate_entities(entities: List[MedicalEntity]) -> List[MedicalEntity]:
    if not entities:
        return []
    
    # Sort by start position and length (longer first)
    entities.sort(key=lambda x: (x.start, -(x.end - x.start)))
    
    unique_entities = []
    seen_texts = set()
    
    for entity in entities:
        # Normalize text for comparison
        normalized_text = entity.text.lower().strip()
        
        # Check for exact duplicates
        if normalized_text in seen_texts:
            continue
            
        # Check for overlapping entities (keep the longer one)
        overlapping = False
        for selected in unique_entities:
            if (entity.start < selected.end and entity.end > selected.start):
                # If current entity is longer, replace the existing one
                if (entity.end - entity.start) > (selected.end - selected.start):
                    unique_entities.remove(selected)
                    seen_texts.discard(selected.text.lower().strip())
                else:
                    overlapping = True
                break
        
        if not overlapping:
            unique_entities.append(entity)
            seen_texts.add(normalized_text)
    
    return unique_entities

def filter_negated_entities(transcript: str, entities: List[MedicalEntity]) -> List[MedicalEntity]:
    """Filter out entities that are mentioned in negative context"""
    filtered_entities = []
    transcript_lower = transcript.lower()
    
    negation_phrases = [
        "no ", "not ", "denies ", "denied ", "without ", "negative for ",
        "never ", "none ", "doesn't have ", "haven't had "
    ]
    
    for entity in entities:
        entity_text = entity.text.lower()
        start, end = entity.start, entity.end
        
        # Check if the entity appears in a negative context
        context_start = max(0, start - 50)
        context_end = min(len(transcript), end + 20)
        context = transcript_lower[context_start:context_end]
        
        is_negated = any(neg in context for neg in negation_phrases)
        
        if not is_negated:
            filtered_entities.append(entity)
        else:
            logger.info(f"Filtered out negated entity: {entity.text}")
    
    return filtered_entities

def extract_medical_patterns(text: str) -> List[MedicalEntity]:
    """Enhanced pattern matching for structured clinical data"""
    entities = []
    
    # Blood pressure patterns
    bp_patterns = [
        r'blood pressure.*?(\d+)\s*\/\s*over\s*(\d+)',
        r'(\d+)\s*over\s*(\d+).*?blood pressure',
        r'bp.*?(\d+)\s*\/\s*(\d+)'
    ]
    
    for pattern in bp_patterns:
        for match in re.finditer(pattern, text.lower()):
            entities.append(MedicalEntity(
                entity="VITAL_SIGN",
                text=f"BP {match.group(1)}/{match.group(2)}",
                start=match.start(),
                end=match.end(),
                confidence=0.95
            ))
    
    # Heart rate patterns
    hr_patterns = [
        r'heart rate.*?(\d+)',
        r'pulse.*?(\d+)',
        r'hr.*?(\d+)'
    ]
    
    for pattern in hr_patterns:
        for match in re.finditer(pattern, text.lower()):
            entities.append(MedicalEntity(
                entity="VITAL_SIGN", 
                text=f"HR {match.group(1)}",
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
    
    return entities

def extract_medication_changes(text: str) -> List[MedicalEntity]:
    """Detect medication start/stop/switch actions"""
    entities = []
    
    # Medication action patterns
    change_patterns = [
        (r'start.*?(losartan|lisinopril|metformin|amlodipine)', "MEDICATION_START"),
        (r'stop.*?(losartan|lisinopril|metformin|amlodipine)', "MEDICATION_STOP"), 
        (r'switch.*?to.*?(losartan|lisinopril|metformin|amlodipine)', "MEDICATION_SWITCH"),
        (r'change.*?to.*?(losartan|lisinopril|metformin|amlodipine)', "MEDICATION_SWITCH"),
        (r'discontinue.*?(losartan|lisinopril|metformin|amlodipine)', "MEDICATION_STOP"),
    ]
    
    for pattern, action in change_patterns:
        for match in re.finditer(pattern, text.lower()):
            entities.append(MedicalEntity(
                entity=action,
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
    
    return entities

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

# Load spaCy with EntityRuler (unchanged)
def load_spacy_with_ruler():
    try:
        import spacy
        from spacy.pipeline import EntityRuler
        
        nlp = spacy.load("en_core_web_sm")
        
        patterns = []
        for label, keywords in MEDICAL_KEYWORDS.items():
            for keyword in keywords:
                patterns.append({"label": label, "pattern": [{"LOWER": keyword.lower()}]})
        
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(patterns)
        
        logger.info("✓ spaCy model with EntityRuler loaded successfully")
        return nlp
    except Exception as e:
        logger.warning(f"spaCy with EntityRuler failed: {e}")
        return None

# Model loading functions (updated to include medical LLM)
async def load_models_async():
    """Asynchronously load all models including medical LLM"""
    global WHISPER_MODEL, PYANNOTE_PIPELINE ,BIOBERT_MODEL, CLINICALBERT_MODEL, SPACY_MODEL, MEDICAL_LLM, MEDICAL_TOKENIZER, _models_loaded
    global REALTIME_PROCESSOR  # Add this

    with _model_load_lock:
        if _models_loaded:
            return
        
        logger.info("Starting async model loading...")
        
        async def load_whisper():
            try:
                def _load_whisper():
                    import whisper
                    model = whisper.load_model(WHISPER_MODEL_SIZE)
                    model = model.to('cpu')
                    return model
                
                model = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, _load_whisper
                )
                logger.info("✓ Whisper model loaded successfully")
                return model
            except Exception as e:
                logger.warning(f"Whisper failed: {e}")
                return None
        
        async def load_biobert():
            try:
                def _load_biobert():
                    from transformers import pipeline
                    
                    if torch.cuda.is_available():
                        device = 0
                    torch.set_default_device('cpu')
                    torch.set_default_dtype(torch.float32)
                    device = -1
                    biopipe = pipeline(
                        "ner",
                        model="dmis-lab/biobert-v1.1",
                        tokenizer="dmis-lab/biobert-v1.1",
                        aggregation_strategy="simple",
                        device=device,
                        torch_dtype=torch.float32  # Explicit dtype
                    )
                    return biopipe
                
                model = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, _load_biobert
                )
                logger.info("✓ BioBERT model loaded successfully")
                return model
            except Exception as e:
                logger.warning(f"BioBERT failed: {e}")
                return None
            
        async def load_clinicalBert():
            try:
                def _load_clinicalBert():
                    from transformers import pipeline
                    if torch.cuda.is_available():
                        device = 0
                    torch.set_default_device('cpu')
                    torch.set_default_dtype(torch.float32)
                    device = -1
                    clinicPipe = pipeline(
                        "ner",
                        model="emilyalsentzer/Bio_ClinicalBERT",
                        tokenizer="emilyalsentzer/Bio_ClinicalBERT",
                        aggregation_strategy="simple",
                        device=device,
                        torch_dtype=torch.float32
                    )
                    return clinicPipe
                model = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, _load_clinicalBert
                )
                logger.info("✓ ClinicalBERT model loaded successfully")
                return model
            except Exception as e:
                logger.warning(f"ClinicalBERT failed: {e}")
                return None
            
        async def load_spacy():
            try:
                model = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, load_spacy_with_ruler
                )
                return model
            except Exception as e:
                logger.warning(f"spaCy failed: {e}")
                return None
        
        async def load_medical_llm():
            """Load medical LLM with Apple Silicon support"""
            try:
                def _load_llm():
                    # Load tokenizer first
                    tokenizer = AutoTokenizer.from_pretrained(
                        MEDICAL_LLM_NAME,
                        trust_remote_code=True
                    )
                    
                    # Set padding token if not present
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    # Determine the best loading strategy based on hardware
                    if torch.backends.mps.is_available():
                        # Apple Silicon - load without bitsandbytes
                        logger.info("Loading model for Apple Silicon (MPS)")
                        model = AutoModelForCausalLM.from_pretrained(
                            MEDICAL_LLM_NAME,
                            device_map="mps",
                            trust_remote_code=True,
                            torch_dtype=torch.float16,  # Use half precision for better performance
                            low_cpu_mem_usage=True
                        )
                        
                    elif torch.cuda.is_available():
                        # NVIDIA GPU - use bitsandbytes if available
                        try:
                            from transformers import BitsAndBytesConfig
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                            )
                            
                            model = AutoModelForCausalLM.from_pretrained(
                                MEDICAL_LLM_NAME,
                                quantization_config=quantization_config,
                                device_map="auto",
                                trust_remote_code=True,
                                torch_dtype=torch.float16
                            )
                            logger.info("Loaded with CUDA and 4-bit quantization")
                            
                        except ImportError:
                            # Fallback without bitsandbytes
                            model = AutoModelForCausalLM.from_pretrained(
                                MEDICAL_LLM_NAME,
                                device_map="auto",
                                trust_remote_code=True,
                                torch_dtype=torch.float16
                            )
                            logger.info("Loaded with CUDA (no quantization)")
                            
                    else:
                        # CPU fallback
                        logger.info("Loading model for CPU")
                        model = AutoModelForCausalLM.from_pretrained(
                            MEDICAL_LLM_NAME,
                            device_map="cpu",
                            trust_remote_code=True,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True
                        )
                    
                    return model, tokenizer
                
                model, tokenizer = await asyncio.get_event_loop().run_in_executor(
                    LLM_POOL, _load_llm
                )
                logger.info(f"✓ Medical LLM ({MEDICAL_LLM_NAME}) loaded successfully")
                return model, tokenizer
                
            except Exception as e:
                logger.error(f"Medical LLM failed: {e}")
                # Try a simpler loading approach as fallback
                try:
                    logger.info("Trying simple loading fallback...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        MEDICAL_LLM_NAME,
                        trust_remote_code=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        MEDICAL_LLM_NAME,
                        trust_remote_code=True
                    )
                    logger.info(f"✓ Fallback loading successful for {MEDICAL_LLM_NAME}")
                    return model, tokenizer
                except Exception as fallback_error:
                    logger.error(f"Fallback loading also failed: {fallback_error}")
                    return None, None
                
        async def load_pyannote():
            """Load pyannote speaker diarization pipeline"""
            if not PYANNOTE_AUTH_TOKEN:
                logger.warning("Pyannote auth token not set, skipping diarization model")
                return None
            
            try:
                def _load_pyannote():
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=PYANNOTE_AUTH_TOKEN
                    )
                    # Send to GPU if available
                    if torch.cuda.is_available():
                        pipeline = pipeline.to(torch.device("cuda"))
                    return pipeline
                
                pipeline = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, _load_pyannote
                )
                logger.info("✓ Pyannote diarization pipeline loaded successfully")
                return pipeline
            except Exception as e:
                logger.warning(f"Pyannote pipeline failed: {e}")
                return None
        
        # Load models concurrently
        results = await asyncio.gather(
            load_whisper(),
            load_biobert(),
            load_clinicalBert(),
            load_spacy(),
            load_pyannote(),
            load_medical_llm(),
            return_exceptions=True
        )
        
        # Sanitize results
        sanitized_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model loading failed with exception: {result}")
                if i == 4:  # LLM result
                    sanitized_results.append((None, None))
                elif i == 5:  # Pyannote result
                    sanitized_results.append(None)
                else:
                    sanitized_results.append(None)
            else:
                sanitized_results.append(result)
        
        WHISPER_MODEL, BIOBERT_MODEL, CLINICALBERT_MODEL, SPACY_MODEL, PYANNOTE_PIPELINE, llm_result = sanitized_results

        # Then handle the LLM result
        if llm_result and isinstance(llm_result, tuple) and len(llm_result) == 2:
            MEDICAL_LLM, MEDICAL_TOKENIZER = llm_result
        else:
            MEDICAL_LLM, MEDICAL_TOKENIZER = None, None
        
        _models_loaded = True
        logger.info("Model loading completed")

        # Initialize real-time processor AFTER models are loaded
        REALTIME_PROCESSOR = RealTimeMedicalProcessor(
            whisper_model=WHISPER_MODEL,
            medical_llm=MEDICAL_LLM,
            pyannote_pipeline=PYANNOTE_PIPELINE,
            biobert_model=BIOBERT_MODEL,
            clinicalbert_model=CLINICALBERT_MODEL,
            spacy_model=SPACY_MODEL
        )
        logger.info("Real-time medical processor initialized")

async def cleanup_models():
    """Cleanup model resources"""
    logger.info("Cleaning up models and thread pools...")
    
    # Clear LLM memory if using GPU
    global MEDICAL_LLM, PYANNOTE_PIPELINE
    if MEDICAL_LLM is not None and torch.cuda.is_available():
        try:
            MEDICAL_LLM = None
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")
    
    PYANNOTE_PIPELINE = None
    
    # Shutdown thread pools
    WHISPER_POOL.shutdown(wait=False)
    BIOBERT_POOL.shutdown(wait=False)
    SPACY_POOL.shutdown(wait=False)
    GENERAL_POOL.shutdown(wait=False)
    LLM_POOL.shutdown(wait=False)
    PYANNOTE_POOL.shutdown(wait=False)
    
    logger.info("Thread pools shutdown completed")

# =====\========================================================================
# WEBSOCKET ENDPOINT
# =============================================================================
@app.websocket("/ws/realtime-audio")
async def websocket_realtime_audio(websocket: WebSocket):
    websocket._receive_timeout = 300  # 5 minutes
    websocket._send_timeout = 300     # 5 minutes

    """WebSocket handler with aggressive keepalive during final processing"""
    session_id = f"realtime_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    await websocket.accept()
    logger.info(f"🔗 WebSocket connected: {session_id}")
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected", 
            "data": {"session_id": session_id, "message": "Ready for audio streaming"}
        })
        
        last_activity = time.time()
        processing_complete = False
        
        async def audio_generator():
            """Convert WebSocket messages to async generator"""
            nonlocal last_activity
            try:
                while True:
                    try:
                        # Use wait_for to handle timeouts
                        message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                        last_activity = time.time()
                        
                        if message["type"] == "websocket.disconnect":
                            logger.info(f"Client disconnected: {session_id}")
                            break
                        
                        if message["type"] == "websocket.receive":
                            if "text" in message:
                                if message["text"] == "END_STREAM":
                                    logger.info(f"End stream signal received for {session_id}")
                                    break
                                
                                # Decode base64 audio data
                                audio_data = base64.b64decode(message["text"])
                                yield audio_data
                            
                            elif "bytes" in message:
                                yield message["bytes"]
                                
                    except asyncio.TimeoutError:
                        # Send keepalive ping more frequently during processing
                        try:
                            await websocket.send_json({"type": "keepalive", "data": {"message": "ping"}})
                            logger.debug("Sent keepalive ping")
                            
                            # If we haven't had activity in a while and processing should be complete, break
                            if processing_complete and time.time() - last_activity > 10:
                                break
                                
                        except:
                            break  # Client disconnected
                            
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session_id}")
        
        # Process audio stream in real-time
        logger.info(f"🎯 Starting real-time processing for {session_id}")
        
        async for result in REALTIME_PROCESSOR.process_realtime_stream(session_id, audio_generator()):
            # Send each result back to client
            try:
                await websocket.send_json({
                    "type": result.type,
                    "data": result.data,
                    "session_id": result.session_id,
                    "is_partial": result.is_partial,
                    "timestamp": result.timestamp
                })
                last_activity = time.time()
                
                # Check if this is the final result
                if result.type == "soap_note_complete":
                    processing_complete = True
                    logger.info(f"✅ Final SOAP note sent for {session_id}")
                    
            except Exception as e:
                logger.error(f"Error sending result to client: {e}")
                break
                
        logger.info(f"🏁 Processing completed for {session_id}")
                
    except Exception as e:
        logger.error(f"💥 WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error", 
                "data": {"message": f"Server error: {str(e)}"}
            })
        except:
            pass
    finally:
        logger.info(f"🔚 WebSocket session ended: {session_id}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "whisper": WHISPER_MODEL is not None,
            "biobert": BIOBERT_MODEL is not None,
            "clinicalbert": CLINICALBERT_MODEL is not None,
            "spacy": SPACY_MODEL is not None,
            "medical_llm": MEDICAL_LLM is not None
        },
        "thread_pools": {
            "whisper_pool": WHISPER_POOL._max_workers,
            "biobert_pool": BIOBERT_POOL._max_workers,
            "clinical_pool": CLINICALBERT_MODEL.max_workers,
            "spacy_pool": SPACY_POOL._max_workers,
            "general_pool": GENERAL_POOL._max_workers,
            "llm_pool": LLM_POOL._max_workers
        },
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    return {
        "service": "Medical NLP Service with LLM SOAP Generation",
        "version": "2.0.0",
        "endpoints": {
            "/process-audio": "Process audio for medical transcription and SOAP generation",
            "/health": "Service health check",
            "/model-info": "Model information"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting negation detection test...")
    test_negation_detection()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,  ws_ping_interval=5,
        ws_ping_timeout=10,
        timeout_keep_alive=30,
        ws_max_size=16777216)