# model_loader.py
import logging
import os
import torch
import asyncio
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from pyannote.audio import Pipeline
from config import WHISPER_MODEL_SIZE, MEDICAL_LLM_NAME, PYANNOTE_AUTH_TOKEN

logger = logging.getLogger("medical-nlp-service")

# Global models
WHISPER_MODEL = None
BIOBERT_MODEL = None
SPACY_MODEL = None
MEDICAL_LLM = None
MEDICAL_TOKENIZER = None
PYANNOTE_PIPELINE = None
REALTIME_PROCESSOR = None

# Thread pools for all model operations
MAX_WORKERS_WHISPER = int(os.getenv('MAX_WORKERS_WHISPER', '1'))
MAX_WORKERS_BIOBERT = int(os.getenv('MAX_WORKERS_BIOBERT', '2'))
MAX_WORKERS_SPACY = int(os.getenv('MAX_WORKERS_SPACY', '2'))
MAX_WORKERS_SENTENCE = int(os.getenv('MAX_WORKERS_SENTENCE', '2'))
MAX_WORKERS_GENERAL = int(os.getenv('MAX_WORKERS_GENERAL', '4'))
MAX_WORKERS_LLM = int(os.getenv('MAX_WORKERS_LLM', '1'))
MAX_WORKERS_PYANNOTE = int(os.getenv('MAX_WORKERS_PYANNOTE', '1'))

# Thread pools
WHISPER_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_WHISPER, thread_name_prefix="whisper_")
BIOBERT_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_BIOBERT, thread_name_prefix="biobert_")
SPACY_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_SPACY, thread_name_prefix="spacy_")
GENERAL_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_GENERAL, thread_name_prefix="general_")
LLM_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_LLM, thread_name_prefix="llm_")
PYANNOTE_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS_PYANNOTE, thread_name_prefix="pyannote_")

# Thread safety locks
_whisper_lock = Lock()
_biobert_lock = Lock()
_spacy_lock = Lock()
_llm_lock = Lock()
_pyannote_lock = Lock()
_model_load_lock = Lock()
_models_loaded = False

def load_spacy_with_ruler():
    """Load spaCy with EntityRuler for medical entities"""
    try:
        import spacy
        from spacy.pipeline import EntityRuler
        from constants import MEDICAL_KEYWORDS
        
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

async def load_whisper():
    """Load Whisper model for speech recognition"""
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
    """Load BioBERT model for medical NER"""
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
    """Load spaCy model"""
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
                    torch_dtype=torch.float16,
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

async def load_models_async():
    """Asynchronously load all models including medical LLM"""
    global WHISPER_MODEL, BIOBERT_MODEL, SPACY_MODEL, MEDICAL_LLM, MEDICAL_TOKENIZER, PYANNOTE_PIPELINE, _models_loaded

    with _model_load_lock:
        if _models_loaded:
            return
        
        logger.info("Starting async model loading...")
        
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
                else:
                    sanitized_results.append(None)
            else:
                sanitized_results.append(result)
        
        WHISPER_MODEL, BIOBERT_MODEL, SPACY_MODEL, PYANNOTE_PIPELINE, llm_result = sanitized_results

        # Handle the LLM result
        if llm_result and isinstance(llm_result, tuple) and len(llm_result) == 2:
            MEDICAL_LLM, MEDICAL_TOKENIZER = llm_result
        else:
            MEDICAL_LLM, MEDICAL_TOKENIZER = None, None
        
        _models_loaded = True
        logger.info("Model loading completed")

async def cleanup_models():
    """Cleanup model resources and shutdown thread pools"""
    global MEDICAL_LLM, PYANNOTE_PIPELINE
    
    logger.info("Cleaning up models and thread pools...")
    
    # Clear LLM memory if using GPU
    if MEDICAL_LLM is not None and torch.cuda.is_available():
        try:
            MEDICAL_LLM = None
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")
    
    PYANNOTE_PIPELINE = None
    
    # Shutdown all thread pools
    WHISPER_POOL.shutdown(wait=False)
    BIOBERT_POOL.shutdown(wait=False)
    SPACY_POOL.shutdown(wait=False)
    GENERAL_POOL.shutdown(wait=False)
    LLM_POOL.shutdown(wait=False)
    PYANNOTE_POOL.shutdown(wait=False)
    
    logger.info("All thread pools shutdown completed")

# Model execution functions
async def run_whisper_transcription(audio_path: str):
    """Run Whisper transcription in thread pool"""
    if WHISPER_MODEL is None:
        raise ValueError("Whisper model not loaded")
    
    return await asyncio.get_event_loop().run_in_executor(
        WHISPER_POOL, 
        lambda: WHISPER_MODEL.transcribe(audio_path)
    )

async def run_biobert_ner(text: str):
    """Run BioBERT NER in thread pool"""
    if BIOBERT_MODEL is None:
        raise ValueError("BioBERT model not loaded")
    
    with _biobert_lock:
        return await asyncio.get_event_loop().run_in_executor(
            BIOBERT_POOL, 
            lambda: BIOBERT_MODEL(text)
        )

async def run_spacy_processing(text: str):
    """Run spaCy processing in thread pool"""
    if SPACY_MODEL is None:
        raise ValueError("spaCy model not loaded")
    
    with _spacy_lock:
        return await asyncio.get_event_loop().run_in_executor(
            SPACY_POOL, 
            lambda: SPACY_MODEL(text)
        )

async def run_llm_generation(prompt: str, generation_params: dict = None):
    """Run LLM generation in thread pool"""
    if MEDICAL_LLM is None or MEDICAL_TOKENIZER is None:
        raise ValueError("Medical LLM not loaded")
    
    if generation_params is None:
        generation_params = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": MEDICAL_TOKENIZER.eos_token_id,
            "repetition_penalty": 1.1
        }
    
    def _generate():
        with _llm_lock:
            inputs = MEDICAL_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Move inputs to the same device as the model
            device = next(MEDICAL_LLM.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = MEDICAL_LLM.generate(**inputs, **generation_params)
            
            # Decode and extract the generated text
            generated_text = MEDICAL_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            return generated_text
    
    return await asyncio.get_event_loop().run_in_executor(LLM_POOL, _generate)

async def run_pyannote_diarization(audio_path: str):
    """Run pyannote diarization in thread pool"""
    if PYANNOTE_PIPELINE is None:
        raise ValueError("Pyannote pipeline not loaded")
    
    with _pyannote_lock:
        return await asyncio.get_event_loop().run_in_executor(
            PYANNOTE_POOL, 
            lambda: PYANNOTE_PIPELINE(audio_path)
        )

async def run_general_task(func, *args):
    """Run general tasks in thread pool"""
    return await asyncio.get_event_loop().run_in_executor(
        GENERAL_POOL, func, *args
    )

# Model information and status
def get_models():
    """Get all loaded models"""
    return {
        "WHISPER_MODEL": WHISPER_MODEL,
        "BIOBERT_MODEL": BIOBERT_MODEL,
        "SPACY_MODEL": SPACY_MODEL,
        "MEDICAL_LLM": MEDICAL_LLM,
        "MEDICAL_TOKENIZER": MEDICAL_TOKENIZER,
        "PYANNOTE_PIPELINE": PYANNOTE_PIPELINE
    }

def get_thread_pools():
    """Get all thread pools with their status"""
    pools = {
        "WHISPER_POOL": WHISPER_POOL,
        "BIOBERT_POOL": BIOBERT_POOL,
        "SPACY_POOL": SPACY_POOL,
        "GENERAL_POOL": GENERAL_POOL,
        "LLM_POOL": LLM_POOL,
        "PYANNOTE_POOL": PYANNOTE_POOL
    }
    
    return {
        name: {
            "max_workers": pool._max_workers,
            "active_threads": pool._work_queue.qsize() if hasattr(pool, '_work_queue') else 0
        }
        for name, pool in pools.items()
    }

def are_models_loaded():
    """Check if all models are loaded"""
    models = get_models()
    return all(model is not None for model in models.values())

def get_model_status():
    """Get detailed model status"""
    models = get_models()
    return {
        "models_loaded": {
            "whisper": models["WHISPER_MODEL"] is not None,
            "biobert": models["BIOBERT_MODEL"] is not None,
            "spacy": models["SPACY_MODEL"] is not None,
            "medical_llm": models["MEDICAL_LLM"] is not None,
            "pyannote": models["PYANNOTE_PIPELINE"] is not None
        },
        "thread_pools": get_thread_pools(),
        "all_models_loaded": are_models_loaded()
    }