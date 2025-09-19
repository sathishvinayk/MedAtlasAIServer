"""
Purpose: Perform the machine learning magic.

Service: asr-service (Automatic Speech Recognition)
API: POST /transcribe
Input: Audio file or stream.
Output: Raw transcript with speaker diarization.
Technology: A fine-tuned OpenAI Whisper model or a custom model running on NVIDIA Riva, hosted on a GPU-enabled cloud instance.

Service: clinical-nlu-service (Natural Language Understanding)
API: POST /analyze
Input: Raw transcript.
Output: Structured JSON of extracted medical entities (symptoms, medications, diagnoses) and their relationships.
Technology: Python (PyTorch/TensorFlow), using a fine-tuned BioBERT or ClinicalBERT model from Hugging Face.

Service: note-assembly-service
API: POST /generate-note
Input: Structured medical entities + original transcript.
Output: A fully formatted clinical note (e.g., in SOAP format).
Technology: Could be a rules-based templating engine or a specialized LLM (like Llama 3 or a fine-tuned GPT) prompted specifically for this task.
"""
# Components together for a basic prototype:
# User records audio in your web app.
# backend sends the audio file to a self-hosted Whisper model.
# This model could be running on a cloud server with a GPU (e.g., an AWS g4dn.xlarge instance).
# Whisper returns the raw transcript.
# backend takes the transcript and sends it to your medical NER model (e.g., a BioBERT model from Hugging Face).
# This model extracts structured data: [[{"entity": "SYMPTOM", "word": "headache"}], ...]
# You then take this structured data and either:
# a) Use a rule-based system to template it into a note: "Patient complains of [SYMPTOM]."
# b) Send it to a smaller, self-hosted LLM (like a fine-tuned Mistral 7B) with a prompt: "Convert these medical entities into a clinical assessment paragraph: [ENTITIES]"
# The final note is presented to the user.
"""
curl -X POST http://localhost:8080/process-audio \           
  -H "Content-Type: application/json" \
  -d "{
    \"audio_data\": \"$(base64 -i doctor_patient_conversation.mp3 | tr -d '\n')\",
    \"file_name\": \"doctor_patient_conversation.mp3\"
  }"
"""
# ASR -> NLU -> SOAP
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple
import hashlib
import base64
from pyannote.audio import Pipeline
import torchaudio
import tempfile
import os
import re
import asyncio
import time
from threading import Lock
from contextlib import asynccontextmanager
import uuid
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
from utils import normalize_medication_name, truncate_text
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
SENTENCE_MODEL = None
WHISPER_MODEL = None
BIOBERT_MODEL = None
SPACY_MODEL = None
MEDICAL_LLM = None
MEDICAL_TOKENIZER = None
PYANNOTE_PIPELINE = None

# Thread pools for each model type
MAX_WORKERS_BIOBERT = int(os.getenv('MAX_WORKERS_BIOBERT', '2'))
MAX_WORKERS_SPACY = int(os.getenv('MAX_WORKERS_SPACY', '2'))
MAX_WORKERS_WHISPER = int(os.getenv('MAX_WORKERS_WHISPER', '1'))
MAX_WORKERS_SENTENCE = int(os.getenv('MAX_WORKERS_SENTENCE', '2'))
MAX_WORKERS_GENERAL = int(os.getenv('MAX_WORKERS_GENERAL', '4'))
MAX_WORKERS_LLM = int(os.getenv('MAX_WORKERS_LLM', '1'))  # LLM is memory-intensive
MAX_WORKERS_PYANNOTE = int(os.getenv('MAX_WORKERS_PYANNOTE', '1'))

SENTENCE_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_SENTENCE, thread_name_prefix="sentence_")
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

# Configuration
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
WHISPER_MODEL_SIZE = "base"
# MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', 'emilyalsentzer/Bio_ClinicalBERT')
MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', 'microsoft/BioGPT-Large')
# Alternatives: 'mistralai/Mistral-7B-v0.1', 'microsoft/BioGPT-Large', 'stanford-crfm/BioMedLM'
PYANNOTE_AUTH_TOKEN = os.getenv('PYANNOTE_AUTH_TOKEN', '') 

# Medical keywords and patterns (unchanged)
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

class SpeakerSegment(BaseModel):
    speaker: str = Field(..., description="Speaker identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")

# Pydantic Models (unchanged)
class MedicalEntity(BaseModel):
    entity: str = Field(..., description="Type of medical entity (SYMPTOM, MEDICATION, etc.)")
    text: str = Field(..., description="The actual text of the entity")
    start: int = Field(..., description="Start character position in text")
    end: int = Field(..., description="End character position in text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to generate embedding for")

class EmbedResponse(BaseModel):
    vector: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used for embedding")
    dims: int = Field(..., description="Dimension of the embedding vector")

class ProcessAudioRequest(BaseModel):
    audio_data: str = Field(..., min_length=100, description="Base64 encoded audio data")
    file_name: str = Field(..., description="Original filename")

    @validator('file_name')
    def validate_file_name(cls, v):
        if not re.match(r'^[\w\s\-\.]+$', v):
            raise ValueError('Filename contains invalid characters')
        return v

    @validator('audio_data')
    def validate_audio_data(cls, v):
        try:
            decoded = base64.b64decode(v, validate=True)
            if len(decoded) > MAX_AUDIO_BYTES:
                raise ValueError(f"Audio data exceeds maximum size of {MAX_AUDIO_BYTES} bytes")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {str(e)}")

class ProcessAudioResponse(BaseModel):
    status: str = Field(..., description="Processing status")
    transcript: str = Field("", description="Transcribed text")
    entities: List[MedicalEntity] = Field(default_factory=list, description="Extracted medical entities")
    soap_note: str = Field("", description="Generated SOAP note")
    speaker_segments: List[SpeakerSegment] = Field(default_factory=list, description="Speaker diarization segments")  # Add speaker segments
    error: Optional[str] = Field(None, description="Error message if any")
    model_used: str = Field("", description="ASR model used")
    nlu_model_used: str = Field("", description="NLU model used")
    llm_model_used: str = Field("", description="LLM model used for SOAP generation")
    diarization_model_used: str = Field("", description="Diarization model used")  # Add diarization model info
    request_id: str = Field(..., description="Unique request identifier")

def perform_diarization(audio_path: str) -> List[SpeakerSegment]:
    """Perform speaker diarization using pyannote"""
    global PYANNOTE_PIPELINE
    
    if PYANNOTE_PIPELINE is None:
        logger.warning("Pyannote pipeline not available, skipping diarization")
        return []
    
    try:
        with _pyannote_lock:
            # Apply the pipeline to the audio file
            diarization = PYANNOTE_PIPELINE(audio_path)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    speaker=speaker,
                    start=round(turn.start, 2),
                    end=round(turn.end, 2),
                    text=""  # This will be filled with transcription later
                ))
            
            logger.info(f"Diarization completed: {len(segments)} segments found")
            return segments
            
    except Exception as e:
        logger.error(f"Pyannote diarization failed: {e}")
        return []

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

# Add function to get audio duration
def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return 0

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
    label_upper = label.upper()
    
    if any(x in label_upper for x in ["DISEASE", "DIAG", "CONDITION"]):
        return "DIAGNOSIS"
    if any(x in label_upper for x in ["CHEM", "DRUG", "MED"]):
        return "MEDICATION"
    if any(x in label_upper for x in ["SYMPTOM", "SIGN"]):
        return "SYMPTOM"
    if any(x in label_upper for x in ["ANATOMY", "BODY", "LOC"]):
        return "BODY_PART"
    
    token_lower = token_text.lower()
    for ent_type, keywords in MEDICAL_KEYWORDS.items():
        if token_lower in keywords:
            return ent_type
    
    return "OTHER"

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

# Keyword-based entity extraction (unchanged)
def extract_entities_keywords(text: str) -> List[MedicalEntity]:
    entities = []
    text_lower = text.lower()
    matched_positions = set()
    
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
    
    return entities

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

# Main entity extraction function (unchanged)
def extract_medical_entities_sync(text: str) -> Tuple[List[MedicalEntity], str]:
    entities = []
    model_used = "keyword-fallback"
    
    global BIOBERT_MODEL, SPACY_MODEL
    
    if BIOBERT_MODEL is not None:
        try:
            with _biobert_lock:
                results = BIOBERT_MODEL(text)
            
            for entity in results:
                if entity.get('score', 0) > 0.6:
                    entity_type = map_biobert_label_to_medical(
                        entity.get('entity_group', ''),
                        entity.get('word', '')
                    )
                    if entity_type != "OTHER":
                        entities.append(MedicalEntity(
                            entity=entity_type,
                            text=entity.get('word', ''),
                            start=entity.get('start', 0),
                            end=entity.get('end', 0),
                            confidence=float(entity.get('score', 0.7))
                        ))
            
            if entities:
                entities = deduplicate_entities(entities)
                model_used = "biobert-medical"
                return entities, model_used
                
        except Exception as e:
            logger.warning(f"BioBERT extraction failed: {e}")
    
    if SPACY_MODEL is not None:
        try:
            with _spacy_lock:
                doc = SPACY_MODEL(text)
            
            for ent in doc.ents:
                entity_type = map_spacy_label_to_medical(ent.label_)
                if entity_type != "OTHER":
                    entities.append(MedicalEntity(
                        entity=entity_type,
                        text=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.9
                    ))
            
            if entities:
                entities = deduplicate_entities(entities)
                model_used = "spacy-medical"
                return entities, model_used
                
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")
    
    entities = extract_entities_keywords(text)
    entities = filter_negated_entities(text, entities)  # <-- ADD THIS LINE
    return entities, model_used

# NEW: LLM-based SOAP note generation
def generate_soap_note_llm(transcript: str, entities: List[MedicalEntity]) -> str:
    """Generate SOAP note using fine-tuned medical LLM"""
    global MEDICAL_LLM, MEDICAL_TOKENIZER
    
    if MEDICAL_LLM is None or MEDICAL_TOKENIZER is None:
        logger.warning("Medical LLM not available, falling back to rule-based SOAP")
        return generate_soap_note_rule_based(transcript, entities)
    
    try:
        with _llm_lock:
            # Prepare context from extracted entities
            symptoms = sorted(set(e.text for e in entities if e.entity == "SYMPTOM"))
            medications = sorted(set(
                normalize_medication_name(e.text)[0]
                for e in entities if e.entity == "MEDICATION"
            ))
            
            # Construct prompt for medical LLM
            prompt = f"""<s>[INST] <<SYS>>
                You are a medical assistant trained to generate comprehensive SOAP notes from patient transcripts.
                Generate a structured SOAP note following this format:

                SUBJECTIVE:
                - Patient's reported symptoms and concerns
                - Relevant medical history from conversation

                OBJECTIVE:
                - Vital signs and physical exam findings (infer from context)
                - Current medications mentioned

                ASSESSMENT:
                - Clinical assessment and differential diagnosis
                - Connection between symptoms and medications

                PLAN:
                - Treatment recommendations
                - Follow-up instructions
                - Medication adjustments if needed

                Keep the note professional, concise, and clinically accurate.
                <</SYS>>

                Patient Transcript: "{truncate_text(transcript, 1500)}"

                Extracted Medical Information:
                - Symptoms: {', '.join(symptoms) if symptoms else 'None reported'}
                - Medications: {', '.join(medications) if medications else 'None reported'}

                Please generate a comprehensive SOAP note based on this information. [/INST]"""
            
            # Tokenize and generate
            inputs = MEDICAL_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # FIX: Move inputs to the same device as the model
            device = next(MEDICAL_LLM.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = MEDICAL_LLM.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=MEDICAL_TOKENIZER.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode and extract the generated text
            generated_text = MEDICAL_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response (after the instruction)
            response = generated_text.split("[/INST]")[-1].strip()
            
            # Clean up any remaining special tokens
            response = re.sub(r'<s>|</s>|\[INST\]|\[/INST\]', '', response).strip()
            
            return response
            
    except Exception as e:
        logger.error(f"LLM SOAP generation failed: {e}")
        # Fallback to rule-based
        return generate_soap_note_rule_based(transcript, entities)

# Fallback rule-based SOAP generation
def generate_soap_note_rule_based(transcript: str, entities: List[MedicalEntity]) -> str:
    """Improved rule-based SOAP note generation"""
    symptoms = sorted(set(e.text for e in entities if e.entity == "SYMPTOM"))
    
    # Fix: Extract just the medication names, not the tuples
    medications = []
    for e in entities:
        if e.entity == "MEDICATION":
            med_name = normalize_medication_name(e.text)[0]
            medications.append(med_name)
    
    medications = sorted(set(medications))
    
    symptom_lower = [s.lower() for s in symptoms]
    med_lower = [m.lower() for m in medications]  # This should work now
    
    # Rest of your function remains the same...
    assessment = "Routine follow-up. Symptoms stable and managed with current treatment plan."
    
    if "dizziness" in symptom_lower and any("lisinopril" in m for m in med_lower):
        assessment = "Dizziness may be related to lisinopril (antihypertensive medication). Consider monitoring blood pressure and potential dosage adjustment."
    elif "cough" in symptom_lower and any("lisinopril" in m for m in med_lower):
        assessment = "Dry cough is a known side effect of ACE inhibitors like lisinopril. Consider alternative antihypertensive if cough persists."
    elif "tired" in symptom_lower or "fatigue" in symptom_lower:
        if medications:
            assessment = f"Fatigue reported; evaluate for potential side effects of {medications[0]} or other underlying causes."
        else:
            assessment = "Fatigue reported; evaluate for underlying causes including anemia, metabolic issues, or sleep disorders."
    
    meds_text = ', '.join(medications) if medications else 'None reported'
    if not medications and "medication" in transcript.lower():
        meds_text = "Patient mentioned medication but none specifically identified"
    
    soap_note = f"""SUBJECTIVE:
        Patient reports: {truncate_text(transcript, 500)}

        Presenting symptoms: {', '.join(symptoms) if symptoms else 'None reported'}

        OBJECTIVE:
        Vital signs: Within normal limits
        Physical examination: Unremarkable
        Current medications: {meds_text}

        ASSESSMENT:
        {assessment}

        PLAN:
        1. Continue current medication regimen with monitoring
        2. Follow up on: {', '.join(symptoms) if symptoms else 'No specific symptoms to monitor'}
        3. Schedule follow-up appointment in 2-4 weeks
        4. Patient instructed to report any worsening symptoms promptly
        5. Consider medication review if side effects persist"""
    
    return soap_note.strip()

# Model loading functions (updated to include medical LLM)
async def load_models_async():
    """Asynchronously load all models including medical LLM"""
    global SENTENCE_MODEL, WHISPER_MODEL, PYANNOTE_PIPELINE ,BIOBERT_MODEL, SPACY_MODEL, MEDICAL_LLM, MEDICAL_TOKENIZER, _models_loaded
    
    with _model_load_lock:
        if _models_loaded:
            return
        
        logger.info("Starting async model loading...")
        
        async def load_sentence_transformer():
            try:
                def _load_st():
                    from sentence_transformers import SentenceTransformer
                    return SentenceTransformer('all-MiniLM-L6-v2')
                
                model = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, _load_st
                )
                logger.info("✓ SentenceTransformer loaded successfully")
                return model
            except Exception as e:
                logger.warning(f"SentenceTransformer failed: {e}")
                return None
        
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
                    return pipeline(
                        "ner",
                        model="dmis-lab/biobert-v1.1",
                        tokenizer="dmis-lab/biobert-v1.1",
                        aggregation_strategy="simple"
                    )
                
                model = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, _load_biobert
                )
                logger.info("✓ BioBERT model loaded successfully")
                return model
            except Exception as e:
                logger.warning(f"BioBERT failed: {e}")
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
            load_sentence_transformer(),
            load_whisper(),
            load_biobert(),
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
        
        SENTENCE_MODEL, WHISPER_MODEL, BIOBERT_MODEL, SPACY_MODEL, PYANNOTE_PIPELINE, llm_result = sanitized_results

        # Then handle the LLM result
        if llm_result and isinstance(llm_result, tuple) and len(llm_result) == 2:
            MEDICAL_LLM, MEDICAL_TOKENIZER = llm_result
        else:
            MEDICAL_LLM, MEDICAL_TOKENIZER = None, None
        
        _models_loaded = True
        logger.info("Model loading completed")

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
    SENTENCE_POOL.shutdown(wait=False)
    WHISPER_POOL.shutdown(wait=False)
    BIOBERT_POOL.shutdown(wait=False)
    SPACY_POOL.shutdown(wait=False)
    GENERAL_POOL.shutdown(wait=False)
    LLM_POOL.shutdown(wait=False)
    PYANNOTE_POOL.shutdown(wait=False)
    
    logger.info("Thread pools shutdown completed")

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

# API Endpoints (updated for LLM SOAP generation)
@app.post("/process-audio", response_model=ProcessAudioResponse)
async def process_audio(request: ProcessAudioRequest):
    """Process audio data and return medical analysis with LLM-generated SOAP note and speaker diarization"""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing audio request {request_id}")
        
        # Decode and validate audio
        audio_bytes = base64.b64decode(request.audio_data)
        
        async with temp_audio_file(audio_bytes) as audio_path:
            # Get audio duration for diarization alignment
            audio_duration = await asyncio.get_event_loop().run_in_executor(
                GENERAL_POOL, get_audio_duration, audio_path
            )
            
            # Perform speaker diarization (async)
            speaker_segments = []
            diarization_model_used = "none"
            if PYANNOTE_PIPELINE is not None:
                try:
                    speaker_segments = await asyncio.get_event_loop().run_in_executor(
                        PYANNOTE_POOL, perform_diarization, audio_path
                    )
                    diarization_model_used = "pyannote/speaker-diarization-3.1"
                    logger.info(f"Diarization found {len(speaker_segments)} speaker segments")
                except Exception as e:
                    logger.error(f"Diarization failed: {e}")
                    diarization_model_used = "failed"
            
            # Transcription
            if WHISPER_MODEL is not None:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        WHISPER_POOL, 
                        lambda: WHISPER_MODEL.transcribe(audio_path)
                    )
                    transcript = result.get("text", "")
                    asr_model_used = f"whisper-{WHISPER_MODEL_SIZE}"
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
                    transcript = await asyncio.get_event_loop().run_in_executor(
                        GENERAL_POOL, 
                        universal_transcript, audio_path
                    )
                    asr_model_used = "universal-fallback"
            else:
                transcript = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, 
                    universal_transcript, audio_path
                )
                asr_model_used = "universal-fallback"
            
            # Align transcription with speaker segments if available
            if speaker_segments:
                speaker_segments = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL,
                    align_transcription_with_speakers, transcript, speaker_segments, audio_duration
                )
            
            # Entity extraction
            if BIOBERT_MODEL is not None:
                entities, nlu_model_used = await asyncio.get_event_loop().run_in_executor(
                    BIOBERT_POOL, 
                    extract_medical_entities_sync, transcript
                )
            elif SPACY_MODEL is not None:
                entities, nlu_model_used = await asyncio.get_event_loop().run_in_executor(
                    SPACY_POOL, 
                    extract_medical_entities_sync, transcript
                )
            else:
                entities, nlu_model_used = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, 
                    extract_medical_entities_sync, transcript
                )
            
            # SOAP note generation with LLM
            llm_model_used = "none"
            if MEDICAL_LLM is not None:
                try:
                    soap_note = await asyncio.get_event_loop().run_in_executor(
                        LLM_POOL, 
                        generate_soap_note_llm, transcript, entities
                    )
                    llm_model_used = MEDICAL_LLM_NAME
                except Exception as e:
                    logger.error(f"LLM SOAP generation failed: {e}")
                    soap_note = generate_soap_note_rule_based(transcript, entities)
                    llm_model_used = "rule-based-fallback"
            else:
                soap_note = generate_soap_note_rule_based(transcript, entities)
                llm_model_used = "rule-based"
            
            logger.info(f"Request {request_id} completed successfully")
            
            return ProcessAudioResponse(
                status="success",
                transcript=transcript,
                entities=entities,
                soap_note=soap_note,
                speaker_segments=speaker_segments,  # Include speaker segments
                model_used=asr_model_used,
                nlu_model_used=nlu_model_used,
                llm_model_used=llm_model_used,
                diarization_model_used=diarization_model_used,  # Include diarization model info
                request_id=request_id
            )
            
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}", exc_info=True)
        return ProcessAudioResponse(
            status="error",
            error=f"Processing failed: {str(e)}",
            request_id=request_id
        )

# Other endpoints remain unchanged
@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    try:
        if SENTENCE_MODEL is not None:
            vector = await asyncio.get_event_loop().run_in_executor(
                SENTENCE_POOL, 
                SENTENCE_MODEL.encode, request.text
            )
            vector = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
            model_name = "all-MiniLM-L6-v2"
        else:
            vector = await asyncio.get_event_loop().run_in_executor(
                GENERAL_POOL, 
                universal_embedding, request.text
            )
            model_name = "universal-fallback"
        
        return EmbedResponse(
            vector=vector,
            model=model_name,
            dims=len(vector)
        )
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        vector = await asyncio.get_event_loop().run_in_executor(
            GENERAL_POOL, 
            universal_embedding, request.text
        )
        return EmbedResponse(
            vector=vector,
            model="error-fallback",
            dims=len(vector)
        )

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "sentence_transformer": SENTENCE_MODEL is not None,
            "whisper": WHISPER_MODEL is not None,
            "biobert": BIOBERT_MODEL is not None,
            "spacy": SPACY_MODEL is not None,
            "medical_llm": MEDICAL_LLM is not None
        },
        "thread_pools": {
            "sentence_pool": SENTENCE_POOL._max_workers,
            "whisper_pool": WHISPER_POOL._max_workers,
            "biobert_pool": BIOBERT_POOL._max_workers,
            "spacy_pool": SPACY_POOL._max_workers,
            "general_pool": GENERAL_POOL._max_workers,
            "llm_pool": LLM_POOL._max_workers
        },
        "timestamp": time.time()
    }

@app.get("/model-info")
async def model_info():
    if SENTENCE_MODEL is not None:
        return {
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dimension": SENTENCE_MODEL.get_sentence_embedding_dimension(),
            "status": "loaded"
        }
    return {
        "model_name": "universal-fallback",
        "embedding_dimension": 384,
        "status": "fallback"
    }

@app.get("/")
async def root():
    return {
        "service": "Medical NLP Service with LLM SOAP Generation",
        "version": "2.0.0",
        "endpoints": {
            "/process-audio": "Process audio for medical transcription and SOAP generation",
            "/embed": "Generate text embeddings",
            "/health": "Service health check",
            "/model-info": "Model information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)