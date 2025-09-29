# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import validator
import logging
from typing import Any, List, Tuple, AsyncGenerator, Dict
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
from dataclasses import dataclass, field
from utils import normalize_medication_name, truncate_text, get_audio_duration, map_spacy_label_to_medical, universal_transcript, temp_audio_file, map_biobert_label_to_medical, align_transcription_with_speakers
from soap_generator import generate_soap_note_rule_based
from audio_models import ProcessAudioRequest, ProcessAudioResponse
from shared_models import MedicalEntity, SpeakerSegment
from entity_extractor import extract_entities_keywords, deduplicate_entities, filter_negated_entities, extract_medication_changes, extract_medical_patterns
from constants import MEDICAL_KEYWORDS
from config import WHISPER_MODEL_SIZE, MEDICAL_LLM_NAME, PYANNOTE_AUTH_TOKEN

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

class AudioStreamProcessor:
    """Process audio streams in real-time without temp files"""
    
    def __init__(self, sample_rate=16000, channels=1, bits_per_sample=16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.bytes_per_sample = bits_per_sample // 8
        
    def add_audio_to_buffer(self, buffer: bytearray, audio_data: bytes):
        """Add audio data to buffer for continuous processing"""
        buffer.extend(audio_data)
        # Keep buffer manageable (max 30 seconds of audio)
        max_buffer_size = 30 * self.sample_rate * self.channels * self.bytes_per_sample
        if len(buffer) > max_buffer_size:
            # Keep the most recent 20 seconds
            keep_size = 20 * self.sample_rate * self.channels * self.bytes_per_sample
            buffer = buffer[-keep_size:]
    
    def create_wav_from_buffer(self, buffer: bytearray) -> bytes:
        """Create valid WAV file from audio buffer"""
        if len(buffer) < 1000:  # Minimum audio size
            return b""
        
        # Calculate sizes
        data_size = len(buffer)
        file_size = data_size + 36  # WAV header size minus 8 bytes
        
        # Create WAV header
        byte_rate = self.sample_rate * self.channels * self.bytes_per_sample
        block_align = self.channels * self.bytes_per_sample
        
        header = b'RIFF'
        header += struct.pack('<I', file_size)
        header += b'WAVE'
        header += b'fmt '
        header += struct.pack('<I', 16)
        header += struct.pack('<H', 1)  # PCM
        header += struct.pack('<H', self.channels)
        header += struct.pack('<I', self.sample_rate)
        header += struct.pack('<I', byte_rate)
        header += struct.pack('<H', block_align)
        header += struct.pack('<H', self.bits_per_sample)
        header += b'data'
        header += struct.pack('<I', data_size)
        
        return header + bytes(buffer)

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
                # LOWER confidence threshold from 0.6 to 0.4 for better recall
                if entity.get('score', 0) > 0.4:
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
    entities = filter_negated_entities(text, entities)
    model_used = "keyword-fallback"
    logger.info(f"Keyword fallback extracted {len(entities)} entities")
    
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

# Model loading functions (updated to include medical LLM)
async def load_models_async():
    """Asynchronously load all models including medical LLM"""
    global WHISPER_MODEL, PYANNOTE_PIPELINE ,BIOBERT_MODEL, SPACY_MODEL, MEDICAL_LLM, MEDICAL_TOKENIZER, _models_loaded
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
        
        WHISPER_MODEL, BIOBERT_MODEL, SPACY_MODEL, PYANNOTE_PIPELINE, llm_result = sanitized_results

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
    extracted_entities: List[MedicalEntity] = field(default_factory=list)
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

class RealTimeMedicalValidator:
    """Validates medical content in real-time"""
    
    def __init__(self):
        self.dangerous_combinations = [
            ("warfarin", "aspirin"),
            ("lisinperol", "ibuprofen"),
            ("metformin", "alcohol"),
            ("simvastatin", "grapefruit")
        ]
        
        self.red_flag_symptoms = [
            "chest pain", "shortness of breath", "severe headache",
            "uncontrolled bleeding", "loss of consciousness"
        ]
    
    def validate_medication_safety(self, medications: List[str]) -> List[Dict[str, str]]:
        """Check for dangerous medication combinations"""
        alerts = []
        meds_lower = [med.lower() for med in medications]
        
        for med1, med2 in self.dangerous_combinations:
            if med1 in meds_lower and med2 in meds_lower:
                alerts.append({
                    "type": "medication_interaction",
                    "message": f"Potential interaction between {med1} and {med2}",
                    "severity": "high"
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
                    "severity": "urgent"
                })
        
        return alerts

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

class RealTimeMedicalProcessor:
    """Optimized real-time medical audio processor"""
    
    def __init__(self, whisper_model, medical_llm=None, pyannote_pipeline=None,
                 biobert_model=None, spacy_model=None):
        self.whisper_model = whisper_model
        self.medical_llm = medical_llm
        self.pyannote_pipeline = pyannote_pipeline
        self.biobert_model = biobert_model
        self.spacy_model = spacy_model
        self.soap_builder = ProgressiveSOAPBuilder()
        self.conversation_analyzer = MedicalConversationAnalyzer()
        
        # Real-time processing with optimizations
        self.audio_processor = AudioStreamProcessor()
        self.processing_pool = ThreadPoolExecutor(max_workers=2)  # Reduced workers
        
        # Session management
        self.active_sessions: Dict[str, PatientContext] = {}
        self.session_lock = Lock()
        
        # Performance tracking
        self.last_processing_time = 0
        self.processing_interval = 5.0  # Process every 5 seconds
        
        logger.info("Optimized RealTimeMedicalProcessor initialized")
    
    def _format_current_soap(self, sections: Dict[str, str]) -> str:
        """Format the current SOAP state for sending to client"""
        soap_note = ""
        for section, content in sections.items():
            if content.strip():
                soap_note += f"{section.upper()}:\n{content}\n\n"
        return soap_note.strip()

    async def _update_progressive_soap(self, patient_context: PatientContext, transcript: str, entities: List[MedicalEntity]):
        """Update SOAP note progressively using the sophisticated rule-based generator"""
        
        # Combine all transcripts so far for context
        all_transcripts = " ".join(patient_context.conversation_history + [transcript])
        
        # Use ALL stored entities (no need to combine manually)
        all_entities = patient_context.extracted_entities  # ← SIMPLIFIED
        
        print(f"🔍 DEBUG: {len(patient_context.conversation_history)} transcripts, {len(entities)} new entities")
        print(f"🔍 DEBUG: Total stored entities: {len(all_entities)}")
        
        # Use your sophisticated rule-based generator
        current_soap = generate_soap_note_rule_based(all_transcripts, all_entities)
        
        print(f"🔍 DEBUG: Generated SOAP length: {len(current_soap)}")
        if current_soap:
            print(f"🔍 DEBUG: SOAP preview: {current_soap[:200]}...")
        
        # Update patient context with current SOAP state
        patient_context.soap_note_sections = self._parse_soap_to_sections(current_soap)
        
        return current_soap

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
    
    async def _process_audio_buffer_optimized(self, patient_context: PatientContext, 
                                            is_final: bool) -> AsyncGenerator[RealtimeResult, None]:
        """Optimized audio buffer processing"""
        try:
            # Convert buffer to WAV format
            wav_data = self.audio_processor.create_wav_from_buffer(patient_context.audio_buffer)
            if not wav_data or len(wav_data) < 1000:
                return
            
            # Quick validation before processing
            if not await self._is_valid_audio_data(wav_data):
                logger.warning("Invalid audio data, skipping processing")
                return
            
            # Transcribe with timeout
            try:
                transcript = await asyncio.wait_for(
                    self._transcribe_audio(wav_data), 
                    timeout=10.0  # 10 second timeout for transcription
                )
            except asyncio.TimeoutError:
                logger.warning("Transcription timeout, skipping chunk")
                return
            
            if not transcript or not transcript.strip():
                return
            
            # Check if this is new content (not duplicate)
            if self._is_duplicate_transcript(patient_context, transcript):
                logger.debug("Duplicate transcript, skipping")
                return
            
            logger.info(f"Processing transcript: '{transcript[:50]}...'")
            
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
            entities = await self._extract_medical_entities_fast(transcript)
            if entities:
                # STORE entities for progressive SOAP building
                patient_context.extracted_entities.extend(entities)
                self._update_patient_context(patient_context, entities)
                
                yield RealtimeResult(
                    type="entities",
                    data={"entities": [entity.dict() for entity in entities]},
                    session_id=patient_context.session_id,
                    is_partial=not is_final
                )
            
            # Final processing
            if is_final:
                full_transcript = " ".join(patient_context.conversation_history)
                if full_transcript.strip():
                    async for result in self._generate_final_results_fast(patient_context, full_transcript):
                        yield result
                        
        except Exception as e:
            logger.error(f"Audio buffer processing error: {e}")
            yield RealtimeResult(
                type="error",
                data={"message": f"Processing error: {str(e)}"},
                session_id=patient_context.session_id,
                is_partial=not is_final
            )
    
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

    def _update_patient_context(self, patient_context: PatientContext, entities: List[MedicalEntity]):
        """Update patient context with new findings"""
        for entity in entities:
            if entity.entity == "SYMPTOM" and entity.text not in patient_context.current_symptoms:
                patient_context.current_symptoms.append(entity.text)
            elif entity.entity == "MEDICATION":
                med_name = normalize_medication_name(entity.text)[0]
                if med_name not in patient_context.medications:
                    patient_context.medications.append(med_name)
            elif entity.entity == "DIAGNOSIS" and entity.text not in patient_context.medical_history:
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
            entities = await self._extract_medical_entities_fast(transcript)
            if entities:
                # STORE ENTITIES CONSISTENTLY
                patient_context.extracted_entities.extend(entities)  # ← ADD THIS
                patient_context.extracted_entities = deduplicate_entities(patient_context.extracted_entities)
                self._update_patient_context(patient_context, entities)
                
                yield RealtimeResult(
                    type="entities",
                    data={
                        "entities": [entity.dict() for entity in entities],
                        "new_entities": len(entities)
                    },
                    session_id=patient_context.session_id,
                    is_partial=not is_final
                )
            
            # DEEPSCRIBE STRATEGY: Progressive SOAP building
            current_soap = await self._update_progressive_soap(patient_context, transcript, entities)
            
            if current_soap:
                yield RealtimeResult(
                    type="soap_update", 
                    data={
                        "current_sections": patient_context.soap_note_sections,
                        "soap_note": current_soap,
                        "is_progressive": True,
                        "new_content": transcript[:100] + "..." if len(transcript) > 100 else transcript
                    },
                    session_id=patient_context.session_id,
                    is_partial=not is_final
                )
            
            # DEEPSCRIBE STRATEGY: If this is final processing, send SOAP immediately
            if is_final:
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

# =============================================================================
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
            "spacy": SPACY_MODEL is not None,
            "medical_llm": MEDICAL_LLM is not None
        },
        "thread_pools": {
            "whisper_pool": WHISPER_POOL._max_workers,
            "biobert_pool": BIOBERT_POOL._max_workers,
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)