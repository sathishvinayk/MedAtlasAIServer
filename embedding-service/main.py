from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import logging
from typing import List
import hashlib

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