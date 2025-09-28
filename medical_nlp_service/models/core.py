# models/core.py
import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import (
    pipeline, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor
)

# Configure logging
logger = logging.getLogger("medical-nlp-models")

class ModelManager:
    """Centralized model management with ClinicalBERT and optional LLM"""
    
    def __init__(self):
        self.SENTENCE_MODEL = None
        self.WHISPER_MODEL = None
        self.WHISPER_PROCESSOR = None
        self.CLINICALBERT_MODEL = None  # Renamed from BIOBERT_MODEL
        self.SPACY_MODEL = None
        self.MEDICAL_LLM = None
        self.MEDICAL_TOKENIZER = None
        self.PYANNOTE_PIPELINE = None
        
        # Configuration
        self.WHISPER_MODEL_SIZE = "base"
        self.CLINICALBERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # ClinicalBERT
        # self.MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', '')  # Empty = no LLM by default
        self.MEDICAL_LLM_NAME = os.getenv('MEDICAL_LLM_NAME', 'microsoft/BioGPT-Large')

        self.PYANNOTE_AUTH_TOKEN = os.getenv('PYANNOTE_AUTH_TOKEN', '')
        
        # Control flags
        self.USE_LLM = bool(self.MEDICAL_LLM_NAME)  # Only load LLM if explicitly specified
        
        # Thread pools
        self.MAX_WORKERS_GENERAL = int(os.getenv('MAX_WORKERS_GENERAL', '4'))
        self.MAX_WORKERS_LLM = int(os.getenv('MAX_WORKERS_LLM', '1'))
        self.GENERAL_POOL = ThreadPoolExecutor(
            max_workers=self.MAX_WORKERS_GENERAL, 
            thread_name_prefix="model_loader_"
        )
        self.LLM_POOL = ThreadPoolExecutor(
            max_workers=self.MAX_WORKERS_LLM, 
            thread_name_prefix="llm_loader_"
        )
        
        # Thread safety
        self._model_load_lock = threading.Lock()
        self._models_loaded = False
        self._load_callbacks = []
    
    def add_load_callback(self, callback):
        """Add callback function to be called when models are loaded"""
        self._load_callbacks.append(callback)
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._models_loaded
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return {
            "sentence_transformer": self.SENTENCE_MODEL is not None,
            "whisper": self.WHISPER_MODEL is not None,
            "clinicalbert": self.CLINICALBERT_MODEL is not None,  # Updated name
            "spacy": self.SPACY_MODEL is not None,
            "medical_llm": self.MEDICAL_LLM is not None,
            "pyannote": self.PYANNOTE_PIPELINE is not None,
            "all_loaded": self._models_loaded
        }
    
    def _load_sentence_transformer(self):
        """Load SentenceTransformer model"""
        try:
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('all-MiniLM-L6-v2')
            # logger.info("✓ SentenceTransformer loaded successfully")
            # return model
            logger.info("SentenceTransformer disabled - model not loaded")
        except Exception as e:
            logger.warning(f"SentenceTransformer failed: {e}")
            return None
    
    def _load_whisper(self):
        """Load Whisper model using the working approach"""
        try:
            import whisper
            # Use the exact same approach as your working code
            model = whisper.load_model(self.WHISPER_MODEL_SIZE)
            model = model.to('cpu')  # Explicitly move to CPU
            logger.info("✓ Whisper model loaded successfully")
            return {"model": model, "processor": None}
        except Exception as e:
            logger.error(f"Whisper failed: {e}")
            return None
    
    def _load_clinicalbert(self):
        """Load ClinicalBERT model for clinical entity recognition"""
        try:
            clinicalbert_pipeline = pipeline(
                "ner",
                model=self.CLINICALBERT_MODEL_NAME,  # ClinicalBERT
                tokenizer=self.CLINICALBERT_MODEL_NAME,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            logger.info("✓ ClinicalBERT model loaded successfully")
            return clinicalbert_pipeline
        except Exception as e:
            logger.error(f"ClinicalBERT failed: {e}", exc_info=True)
            return None
    
    def _load_spacy_with_ruler(self):
        """Load spaCy model with medical entity ruler (fallback only)"""
        try:
            import spacy
            from spacy.pipeline import EntityRuler
            
            # Medical keywords for entity ruler (minimal fallback only)
            MEDICAL_KEYWORDS = {
                "SYMPTOM": ["headache", "fever", "cough", "pain", "nausea", "dizziness", 
                            "fatigue", "tired", "shortness of breath", "weakness"],
                "MEDICATION": ["ibuprofen", "aspirin", "amoxicillin", "lisinopril", 
                              "metformin", "tylenol", "advil", "atenolol", "amlodipine"],
                "DIAGNOSIS": ["hypertension", "high blood pressure", "diabetes", 
                             "migraine", "infection", "arthritis", "asthma", "pneumonia"],
                "BODY_PART": ["head", "chest", "arm", "leg", "back", "stomach", "throat"]
            }
            
            nlp = spacy.load("en_core_web_sm")
            
            patterns = []
            for label, keywords in MEDICAL_KEYWORDS.items():
                for keyword in keywords:
                    patterns.append({"label": label, "pattern": [{"LOWER": keyword.lower()}]})
            
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(patterns)
            
            logger.info("✓ spaCy model with EntityRuler loaded successfully (fallback)")
            return nlp
        except Exception as e:
            logger.warning(f"spaCy with EntityRuler failed: {e}")
            return None
    
    async def _load_medical_llm(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Load medical LLM only if explicitly requested"""
        if not self.USE_LLM:
            logger.info("LLM loading skipped - using rule-based SOAP generation")
            return None, None
            
        try:
            def _load_llm():
                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(
                    self.MEDICAL_LLM_NAME,
                    trust_remote_code=True
                )
                
                # Set padding token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Determine the best loading strategy based on hardware
                if torch.backends.mps.is_available():
                    # Apple Silicon
                    logger.info("Loading model for Apple Silicon (MPS)")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.MEDICAL_LLM_NAME,
                        device_map="mps",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                    
                elif torch.cuda.is_available():
                    # NVIDIA GPU
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            self.MEDICAL_LLM_NAME,
                            quantization_config=quantization_config,
                            device_map="auto",
                            trust_remote_code=True,
                            torch_dtype=torch.float16
                        )
                        logger.info("Loaded with CUDA and 4-bit quantization")
                        
                    except ImportError:
                        # Fallback without bitsandbytes
                        model = AutoModelForCausalLM.from_pretrained(
                            self.MEDICAL_LLM_NAME,
                            device_map="auto",
                            trust_remote_code=True,
                            torch_dtype=torch.float16
                        )
                        logger.info("Loaded with CUDA (no quantization)")
                        
                else:
                    # CPU fallback
                    logger.info("Loading model for CPU")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.MEDICAL_LLM_NAME,
                        device_map="cpu",
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                
                return model, tokenizer
            
            model, tokenizer = await asyncio.get_event_loop().run_in_executor(
                self.LLM_POOL, _load_llm
            )
            logger.info(f"✓ Medical LLM ({self.MEDICAL_LLM_NAME}) loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Medical LLM failed: {e}")
            return None, None
    
    def _load_pyannote(self):
        """Load pyannote speaker diarization pipeline"""
        if not self.PYANNOTE_AUTH_TOKEN:
            logger.warning("Pyannote auth token not set, skipping diarization model")
            return None
        
        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.PYANNOTE_AUTH_TOKEN
            )
            # Send to GPU if available
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
            logger.info("✓ Pyannote diarization pipeline loaded successfully")
            return pipeline
        except Exception as e:
            logger.warning(f"Pyannote pipeline failed: {e}")
            return None
    
    async def load_models_async(self):
        """Asynchronously load all models with ClinicalBERT as primary"""
        with self._model_load_lock:
            if self._models_loaded:
                logger.info("Models already loaded, skipping")
                return
            
            logger.info("Starting async model loading...")
            logger.info(f"LLM enabled: {self.USE_LLM} (MEDICAL_LLM_NAME: {self.MEDICAL_LLM_NAME})")
            
            # Load models concurrently
            tasks = [
                self.GENERAL_POOL.submit(self._load_sentence_transformer),
                self.GENERAL_POOL.submit(self._load_whisper),
                self.GENERAL_POOL.submit(self._load_clinicalbert),  # ClinicalBERT
                self.GENERAL_POOL.submit(self._load_spacy_with_ruler),
                self.GENERAL_POOL.submit(self._load_pyannote),
            ]
            
            # Only add LLM task if explicitly enabled
            if self.USE_LLM:
                # Convert async function to sync for thread pool
                tasks.append(self.LLM_POOL.submit(
                    lambda: asyncio.run(self._load_medical_llm())
                ))
            else:
                # Add a placeholder for consistent indexing
                tasks.append(lambda: (None, None))
            
            # Wait for all tasks to complete
            completed_tasks = await asyncio.get_event_loop().run_in_executor(
                None, lambda: [task.result() for task in tasks]
            )
            
            # Assign results
            self.SENTENCE_MODEL = completed_tasks[0]
            
            # Handle Whisper result (now returns dict with model and processor)
            whisper_result = completed_tasks[1]
            if whisper_result and isinstance(whisper_result, dict):
                self.WHISPER_MODEL = whisper_result.get("model")
                self.WHISPER_PROCESSOR = whisper_result.get("processor")
            else:
                self.WHISPER_MODEL = None
                self.WHISPER_PROCESSOR = None
            
            self.CLINICALBERT_MODEL = completed_tasks[2]  # ClinicalBERT
            self.SPACY_MODEL = completed_tasks[3]
            self.PYANNOTE_PIPELINE = completed_tasks[4]
            
            # Handle LLM result
            llm_result = completed_tasks[5]
            if llm_result and isinstance(llm_result, tuple) and len(llm_result) == 2:
                self.MEDICAL_LLM, self.MEDICAL_TOKENIZER = llm_result
            else:
                self.MEDICAL_LLM, self.MEDICAL_TOKENIZER = None, None
            
            self._models_loaded = True
            
            # Call registered callbacks
            for callback in self._load_callbacks:
                try:
                    callback(self)
                except Exception as e:
                    logger.error(f"Callback failed: {e}")
            
            logger.info("Model loading completed")
            logger.info(f"Model status: {self.get_model_status()}")
    
    async def cleanup(self):
        """Cleanup model resources"""
        logger.info("Cleaning up models...")
        
        # Clear LLM memory if using GPU
        if self.MEDICAL_LLM is not None and torch.cuda.is_available():
            try:
                self.MEDICAL_LLM = None
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")
        
        self.PYANNOTE_PIPELINE = None
        
        # Shutdown thread pools
        self.GENERAL_POOL.shutdown(wait=False)
        self.LLM_POOL.shutdown(wait=False)
        
        logger.info("Model cleanup completed")

# Global instance
model_manager = ModelManager()

# Convenience functions for direct access
async def load_models():
    """Convenience function to load models using global instance"""
    return await model_manager.load_models_async()

async def cleanup_models():
    """Convenience function to cleanup models"""
    return await model_manager.cleanup()

def get_model_status():
    """Get model loading status"""
    return model_manager.get_model_status()

def is_models_loaded():
    """Check if models are loaded"""
    return model_manager.is_loaded()

def is_llm_enabled():
    """Check if LLM is enabled"""
    return model_manager.USE_LLM

def get_clinicalbert_model():
    """Get ClinicalBERT model instance"""
    return model_manager.CLINICALBERT_MODEL

def get_whisper_model():
    """Get Whisper model instance"""
    return model_manager.WHISPER_MODEL

def get_whisper_processor():
    """Get Whisper processor instance"""
    return model_manager.WHISPER_PROCESSOR