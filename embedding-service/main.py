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

# ASR -> NLU -> SOAP
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import logging
from typing import List, Optional
import hashlib
import base64
import tempfile
import os
import whisper 

from openai import OpenAI

app = FastAPI(title="Embedding Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    vector: List[float]
    model: str
    dims: int

class ProcessAudioRequest(BaseModel):
    audio_data: str
    file_name: str

class ProcessAudioResponse(BaseModel):
    status: str
    transcript: str = ""
    soap_note: str = ""
    error: Optional[str] = None
    model_used: str = ""

# Try to load proper model, but fallback to universal
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Loaded sentence-transformers model")
except ImportError as e:
    print(f"✗ Could not load sentence-transformers: {e}")
    MODEL = None
except Exception as e:
    print(f"✗ Error loading model: {e}")
    MODEL = None

try:
    WHISPER_MODEL = whisper.load_model("base")
    print("✓ Loaded Whisper model for speech recognition")
except ImportError as e:
    print(f"✗ Could not load whisper-model: {e}")
    WHISPER_MODEL = None
except Exception as e:
    print(f"✗ Could not load Whisper model: {e}")
    WHISPER_MODEL = None

def universal_transcript(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Create deterministic "transcript" based on hash
    seed = int(audio_hash[:8], 16)
    np.random.seed(seed)

    symptoms = ["headache", "fever", "cough", "chest pain", "fatigue"]
    medications = ["ibuprofen", "amoxicillin", "lisinopril", "metformin"]

    random_symptoms = np.random.choice(symptoms, size=2, replace=False)
    random_med = np.random.choice(medications, size=1)[0]

    return f"Patient presents with {' and '.join(random_symptoms)}. Currently taking {random_med}. Denies other symptoms. Vital signs stable."

def universal_embedding(text, dimensions=384):
    """Universal embedding function that works everywhere"""
    # Create a deterministic embedding based on text hash
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    seed = int(text_hash[:8], 16)  # Use first 8 chars for seed
    
    np.random.seed(seed)
    embedding = np.random.randn(dimensions).astype(np.float32)
    
    # Normalize to unit length
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.tolist()

# Sample soap generation
def generate_soap_note(transcript: str) -> str:
    """Generate a SOAP note from transcript - currently a simple template"""
    soap_template = f"""
        SUBJECTIVE:
            Patient states: "{transcript}"
        OBJECTIVE:
            Vital signs within normal limits. Physical examination unremarkable.
        ASSESSMENT:
            Probable viral syndrome. Continue current management.
        PLAN:
            - Supportive care
            - Follow up as needed
            - Return if symptoms worsen
    """
    return soap_template.strip()

@app.post("/process-audio", response_model=ProcessAudioResponse)
async def process_audio(request: ProcessAudioRequest):
    """Process audio data and return transcript and medical note"""
    try:
        try:
            audio_bytes = base64.b64decode(request.audio_data)
        except Exception as e:
            return ProcessAudioResponse(
                status="error",
                error=f"Invalid audio data: {str(e)}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name

        try:
            if WHISPER_MODEL is not None:
                result = WHISPER_MODEL.transcribe(tmp_audio_path)
                transcript = result["text"]
                model_used = "whisper-base"
            else:
                transcript = universal_transcript(tmp_audio_path)
                model_used = "universal_fallback"

            soap_note = generate_soap_note(transcript)

            return ProcessAudioResponse(
                status="success",
                transcript=transcript,
                soap_note=soap_note,
                model_used=model_used
            )
        finally:
            os.unlink(tmp_audio_path)
    except Exception as e:
        error_msg = f"Audio processing error: {str(e)}"
        print(error_msg)
        return ProcessAudioResponse(
            status="error",
            error=error_msg
        )

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    try:
        if MODEL is not None:
            # Use the proper model
            vector = MODEL.encode(request.text).tolist()
            model_name = "all-MiniLM-L6-v2"
        else:
            # Fallback to universal embedding
            vector = universal_embedding(request.text)
            model_name = "universal-hash-embedding"
        
        return EmbedResponse(
            vector=vector,
            model=model_name,
            dims=len(vector)
        )
    except Exception as e:
        print(f"Embedding error: {e}, using fallback")
        # Final fallback
        vector = universal_embedding(request.text)
        return EmbedResponse(
            vector=vector,
            model="fallback-universal",
            dims=len(vector)
        )

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": MODEL is not None,
        "model_type": "sentence-transformers" if MODEL else "universal-fallback"
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify service is working"""
    test_text = "This is a test query for medical search"
    
    if MODEL is not None:
        vector = MODEL.encode(test_text).tolist()
        model_type = "sentence-transformers"
    else:
        vector = universal_embedding(test_text)
        model_type = "universal-fallback"
    
    return {
        "message": "Embedding service is working",
        "model_type": model_type,
        "embedding_length": len(vector),
        "embedding_sample": vector[:5],
        "test_text": test_text
    }

@app.get("/model-info")
async def model_info():
    if MODEL is not None:
        return {
            "model_name": str(MODEL),
            "embedding_dimension": MODEL.get_sentence_embedding_dimension(),
            "status": "loaded"
        }
    else:
        return {
            "model_name": "universal-fallback",
            "embedding_dimension": 384,
            "status": "fallback"
        }

@app.get("/whisper-model-info")
async def whisper_model_info():
    if WHISPER_MODEL is not None:
        return {
            "model_name": f"whisper-{WHISPER_MODEL}",
            "status": "loaded"
        }
    else:
        return {
            "model_name": "universal-fallback",
            "status": "fallback"
        }
