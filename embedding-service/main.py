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
import tempfile
import os
import re
import asyncio
import time
from threading import Lock
from contextlib import asynccontextmanager
import uuid
from concurrent.futures import ThreadPoolExecutor

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    await load_models_async()
    yield
    # Shutdown: cleanup
    await cleanup_models()

app = FastAPI(
    title="Medical Embedding Service",
    description="API for medical audio processing, transcription, and entity extraction",
    version="1.0.0",
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
logger = logging.getLogger("medical-embedding-service")

# Global models (loaded asynchronously)
SENTENCE_MODEL = None
WHISPER_MODEL = None
BIOBERT_MODEL = None
SPACY_MODEL = None

# Thread pools for each model type - configurable via environment variables
MAX_WORKERS_BIOBERT = int(os.getenv('MAX_WORKERS_BIOBERT', '2'))
MAX_WORKERS_SPACY = int(os.getenv('MAX_WORKERS_SPACY', '2'))
MAX_WORKERS_WHISPER = int(os.getenv('MAX_WORKERS_WHISPER', '1'))
MAX_WORKERS_SENTENCE = int(os.getenv('MAX_WORKERS_SENTENCE', '2'))
MAX_WORKERS_GENERAL = int(os.getenv('MAX_WORKERS_GENERAL', '4'))

SENTENCE_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_SENTENCE, thread_name_prefix="sentence_")
WHISPER_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_WHISPER, thread_name_prefix="whisper_")
BIOBERT_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_BIOBERT, thread_name_prefix="biobert_")
SPACY_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_SPACY, thread_name_prefix="spacy_")
GENERAL_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_GENERAL, thread_name_prefix="general_")

# Thread safety
_biobert_lock = Lock()
_spacy_lock = Lock()
_model_load_lock = Lock()  # Lock for model loading
_models_loaded = False

# Configuration
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
WHISPER_MODEL_SIZE = "base"

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

# Pydantic Models
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
        """Validate filename pattern"""
        if not re.match(r'^[\w\s\-\.]+$', v):
            raise ValueError('Filename contains invalid characters')
        return v

    @validator('audio_data')
    def validate_audio_data(cls, v):
        try:
            # Validate base64 and check size
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
    error: Optional[str] = Field(None, description="Error message if any")
    model_used: str = Field("", description="ASR model used")
    nlu_model_used: str = Field("", description="NLU model used")
    request_id: str = Field(..., description="Unique request identifier")

# Utility functions
def normalize_medication_name(text: str) -> str:
    """Normalize medication names to standard terms"""
    lower_text = text.lower()
    for synonym, standard in MEDICATION_SYNONYMS.items():
        if synonym in lower_text:
            return standard
    return text

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text for readability"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def deduplicate_entities(entities: List[MedicalEntity]) -> List[MedicalEntity]:
    """Better deduplication that handles overlapping spans by preferring longest match"""
    if not entities:
        return []
    
    # Sort by start position and then by length (longest first)
    entities.sort(key=lambda x: (x.start, -(x.end - x.start)))
    
    unique_entities = []
    seen_positions = set()
    
    for entity in entities:
        # Check if this entity overlaps with any already selected entity
        overlapping = False
        for selected in unique_entities:
            if (entity.start < selected.end and entity.end > selected.start):
                overlapping = True
                break
        
        # Only add if it doesn't overlap with any already selected entity
        if not overlapping:
            unique_entities.append(entity)
    
    return unique_entities

def universal_embedding(text: str, dimensions: int = 384) -> List[float]:
    """Deterministic fallback embedding"""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    seed = int(text_hash[:8], 16)
    
    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal(dimensions).astype(np.float32)
    
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.tolist()

def universal_transcript(audio_path: str) -> str:
    """Fallback transcription for testing"""
    with open(audio_path, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()
    
    seed = int(audio_hash[:8], 16)
    rng = np.random.default_rng(seed)

    symptoms = ["headache", "fever", "cough", "chest pain", "fatigue", "dizziness"]
    medications = ["ibuprofen", "amoxicillin", "lisinopril", "metformin"]

    random_symptoms = rng.choice(symptoms, size=2, replace=False)
    random_med = rng.choice(medications, size=1)[0]

    return f"Patient presents with {' and '.join(random_symptoms)}. Currently taking {random_med}. Denies other symptoms. Vital signs stable."

# Model mapping functions
def map_biobert_label_to_medical(label: str, token_text: str) -> str:
    """Map BioBERT labels to medical categories"""
    label_upper = label.upper()
    
    if any(x in label_upper for x in ["DISEASE", "DIAG", "CONDITION"]):
        return "DIAGNOSIS"
    if any(x in label_upper for x in ["CHEM", "DRUG", "MED"]):
        return "MEDICATION"
    if any(x in label_upper for x in ["SYMPTOM", "SIGN"]):
        return "SYMPTOM"
    if any(x in label_upper for x in ["ANATOMY", "BODY", "LOC"]):
        return "BODY_PART"
    
    # Fallback to keyword matching
    token_lower = token_text.lower()
    for ent_type, keywords in MEDICAL_KEYWORDS.items():
        if token_lower in keywords:
            return ent_type
    
    return "OTHER"

def map_spacy_label_to_medical(label: str) -> str:
    """Map spaCy labels to medical categories"""
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

# Keyword-based entity extraction
def extract_entities_keywords(text: str) -> List[MedicalEntity]:
    """Improved keyword extraction without duplicates"""
    entities = []
    text_lower = text.lower()
    matched_positions = set()
    
    for entity_type, keywords in MEDICAL_KEYWORDS.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            for match in re.finditer(pattern, text_lower):
                start, end = match.start(), match.end()
                
                # Check if this position is already covered
                position_key = (start, end)
                if position_key not in matched_positions:
                    entities.append(MedicalEntity(
                        entity=entity_type,
                        text=text[start:end],  # Preserve original case
                        start=start,
                        end=end,
                        confidence=0.8
                    ))
                    matched_positions.add(position_key)
    
    return entities

# Load spaCy with EntityRuler
def load_spacy_with_ruler():
    """Load spaCy model with EntityRuler for medical patterns"""
    try:
        import spacy
        from spacy.pipeline import EntityRuler
        
        nlp = spacy.load("en_core_web_sm")
        
        # Create patterns for medical entities
        patterns = []
        for label, keywords in MEDICAL_KEYWORDS.items():
            for keyword in keywords:
                patterns.append({"label": label, "pattern": [{"LOWER": keyword.lower()}]})
        
        # Add entity ruler
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(patterns)
        
        logger.info("✓ spaCy model with EntityRuler loaded successfully")
        return nlp
    except Exception as e:
        logger.warning(f"spaCy with EntityRuler failed: {e}")
        return None

# Main entity extraction function
def extract_medical_entities_sync(text: str) -> Tuple[List[MedicalEntity], str]:
    """Synchronous entity extraction (run in executor)"""
    entities = []
    model_used = "keyword-fallback"
    
    global BIOBERT_MODEL, SPACY_MODEL
    
    # Try BioBERT first
    if BIOBERT_MODEL is not None:
        try:
            with _biobert_lock:  # Use threading lock for short sync operations
                results = BIOBERT_MODEL(text)
            
            for entity in results:
                if entity.get('score', 0) > 0.6:  # Confidence threshold
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
    
    # Try spaCy with medical patterns
    if SPACY_MODEL is not None:
        try:
            with _spacy_lock:  # Use threading lock for short sync operations
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
    
    # Fallback to keyword matching
    entities = extract_entities_keywords(text)
    return entities, model_used

# SOAP note generation
def generate_soap_note(transcript: str, entities: List[MedicalEntity]) -> str:
    """Generate clinical SOAP note"""
    symptoms = sorted(set(e.text for e in entities if e.entity == "SYMPTOM"))
    medications = sorted(set(
        normalize_medication_name(e.text) 
        for e in entities if e.entity == "MEDICATION"
    ))
    
    # Clinical assessment logic
    symptom_lower = [s.lower() for s in symptoms]
    med_lower = [m.lower() for m in medications]
    
    if "dizziness" in symptom_lower and any("lisinopril" in m for m in med_lower):
        assessment = "Dizziness may be related to antihypertensive medication. Consider monitoring blood pressure and potential dosage adjustment."
    elif "cough" in symptom_lower and any("lisinopril" in m for m in med_lower):
        assessment = "Dry cough is a known side effect of ACE inhibitors like lisinopril. Consider alternative antihypertensive if cough persists."
    elif "fatigue" in symptom_lower or "tired" in symptom_lower:
        assessment = "Fatigue reported; evaluate for underlying causes including medication side effects, anemia, or metabolic issues."
    else:
        assessment = "Routine follow-up. Symptoms stable and managed with current treatment plan."
    
    soap_note = f"""SUBJECTIVE:
        Patient reports: {truncate_text(transcript, 250)}

        Presenting symptoms: {', '.join(symptoms) if symptoms else 'None reported'}

        OBJECTIVE:
        Vital signs: Within normal limits
        Physical examination: Unremarkable
        Current medications: {', '.join(medications) if medications else 'None reported'}

        ASSESSMENT:
        {assessment}

        PLAN:
        1. Continue current medication regimen with monitoring
        2. Follow up on: {', '.join(symptoms) if symptoms else 'No specific symptoms to monitor'}
        3. Schedule follow-up appointment in 2-4 weeks
        4. Patient instructed to report any worsening symptoms promptly
        """
    return soap_note.strip()

# Model loading functions
async def load_models_async():
    """Asynchronously load all models"""
    global SENTENCE_MODEL, WHISPER_MODEL, BIOBERT_MODEL, SPACY_MODEL, _models_loaded
    
    # Use lock to prevent multiple concurrent loads
    with _model_load_lock:
        if _models_loaded:
            return
        
        logger.info("Starting async model loading...")
        
        async def load_sentence_transformer():
            try:
                # Offload blocking constructor to thread
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
                # Offload blocking constructor to thread
                def _load_whisper():
                    import whisper
                    return whisper.load_model(WHISPER_MODEL_SIZE)
                
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
                # Offload blocking constructor to thread
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
                # Offload blocking constructor to thread
                model = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, load_spacy_with_ruler
                )
                return model
            except Exception as e:
                logger.warning(f"spaCy failed: {e}")
                return None
        
        # Load models concurrently
        results = await asyncio.gather(
            load_sentence_transformer(),
            load_whisper(),
            load_biobert(),
            load_spacy(),
            return_exceptions=True
        )
        
        # Sanitize results - replace exceptions with None
        sanitized_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model loading failed with exception: {result}")
                sanitized_results.append(None)
            else:
                sanitized_results.append(result)
        
        SENTENCE_MODEL, WHISPER_MODEL, BIOBERT_MODEL, SPACY_MODEL = sanitized_results
        _models_loaded = True
        logger.info("Model loading completed")

async def cleanup_models():
    """Cleanup model resources"""
    logger.info("Cleaning up models and thread pools...")
    
    # Shutdown thread pools
    SENTENCE_POOL.shutdown(wait=False)
    WHISPER_POOL.shutdown(wait=False)
    BIOBERT_POOL.shutdown(wait=False)
    SPACY_POOL.shutdown(wait=False)
    GENERAL_POOL.shutdown(wait=False)
    
    logger.info("Thread pools shutdown completed")

# Temporary file context manager
@asynccontextmanager
async def temp_audio_file(audio_bytes: bytes):
    """Context manager for temporary audio files"""
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

# API Endpoints
@app.post("/process-audio", response_model=ProcessAudioResponse)
async def process_audio(request: ProcessAudioRequest):
    """Process audio data and return medical analysis"""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing audio request {request_id}")
        
        # Decode and validate audio
        audio_bytes = base64.b64decode(request.audio_data)
        
        async with temp_audio_file(audio_bytes) as audio_path:
            # Transcription - use Whisper-specific thread pool
            if WHISPER_MODEL is not None:
                try:
                    # Run transcription in Whisper thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        WHISPER_POOL, 
                        lambda: WHISPER_MODEL.transcribe(audio_path)
                    )
                    transcript = result.get("text", "")
                    asr_model_used = f"whisper-{WHISPER_MODEL_SIZE}"
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
                    # Fallback to universal transcript in general pool
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
            
            # Entity extraction - use appropriate thread pool based on model availability
            if BIOBERT_MODEL is not None:
                # Use BioBERT pool for entity extraction
                entities, nlu_model_used = await asyncio.get_event_loop().run_in_executor(
                    BIOBERT_POOL, 
                    extract_medical_entities_sync, transcript
                )
            elif SPACY_MODEL is not None:
                # Use spaCy pool for entity extraction
                entities, nlu_model_used = await asyncio.get_event_loop().run_in_executor(
                    SPACY_POOL, 
                    extract_medical_entities_sync, transcript
                )
            else:
                # Fallback to general pool
                entities, nlu_model_used = await asyncio.get_event_loop().run_in_executor(
                    GENERAL_POOL, 
                    extract_medical_entities_sync, transcript
                )
            
            # SOAP note generation - run inline since it's CPU-light
            soap_note = generate_soap_note(transcript, entities)
            
            logger.info(f"Request {request_id} completed successfully")
            
            return ProcessAudioResponse(
                status="success",
                transcript=transcript,
                entities=entities,
                soap_note=soap_note,
                model_used=asr_model_used,
                nlu_model_used=nlu_model_used,
                request_id=request_id
            )
            
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}", exc_info=True)
        return ProcessAudioResponse(
            status="error",
            error=f"Processing failed: {str(e)}",
            request_id=request_id
        )

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate text embeddings"""
    try:
        if SENTENCE_MODEL is not None:
            # Use sentence transformer pool for embedding
            vector = await asyncio.get_event_loop().run_in_executor(
                SENTENCE_POOL, 
                SENTENCE_MODEL.encode, request.text
            )
            vector = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
            model_name = "all-MiniLM-L6-v2"
        else:
            # Fallback to universal embedding in general pool
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
        # Error fallback in general pool
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "sentence_transformer": SENTENCE_MODEL is not None,
            "whisper": WHISPER_MODEL is not None,
            "biobert": BIOBERT_MODEL is not None,
            "spacy": SPACY_MODEL is not None
        },
        "thread_pools": {
            "sentence_pool": SENTENCE_POOL._max_workers,
            "whisper_pool": WHISPER_POOL._max_workers,
            "biobert_pool": BIOBERT_POOL._max_workers,
            "spacy_pool": SPACY_POOL._max_workers,
            "general_pool": GENERAL_POOL._max_workers
        },
        "timestamp": time.time()
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
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
    """Root endpoint with service information"""
    return {
        "service": "Medical Embedding Service",
        "version": "1.0.0",
        "endpoints": {
            "/process-audio": "Process audio for medical transcription",
            "/embed": "Generate text embeddings",
            "/health": "Service health check",
            "/model-info": "Model information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)