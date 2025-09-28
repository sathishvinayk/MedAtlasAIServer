from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import validator
import logging
from typing import List, Tuple
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
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
)
from utils import normalize_medication_name, truncate_text, get_audio_duration, map_spacy_label_to_medical, universal_transcript, temp_audio_file, map_biobert_label_to_medical, align_transcription_with_speakers
from soap_generator import generate_soap_note_rule_based
from entities import MedicalEntity, SpeakerSegment, ProcessAudioRequest, ProcessAudioResponse
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
    global SENTENCE_MODEL, WHISPER_MODEL, PYANNOTE_PIPELINE ,BIOBERT_MODEL, SPACY_MODEL, MEDICAL_LLM, MEDICAL_TOKENIZER, _models_loaded
    
    with _model_load_lock:
        if _models_loaded:
            return
        
        logger.info("Starting async model loading...")
        
        async def load_sentence_transformer():
            try:
                # def _load_st():
                #     from sentence_transformers import SentenceTransformer
                #     return SentenceTransformer('all-MiniLM-L6-v2')
                
                # model = await asyncio.get_event_loop().run_in_executor(
                #     GENERAL_POOL, _load_st
                # )
                logger.info("✓ SentenceTransformer not loaded")
                # return model
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