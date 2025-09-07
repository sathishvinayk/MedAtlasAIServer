from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Initially load None as model on startup
model = None

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    vector: list[float]
    model: str
    dims: int

@app.on_event("startup")
async def load_model():
    global model
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e
    
@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        embedding = model.encode(request.text)
        embedding_list = embedding.tolist()

        return EmbedResponse(
            vector=embedding_list,
            model=model._modules['0'].auto_model.config.name_or_path,
            dims = len(embedding_list)
        )
    except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail="Error generating embedding")
    

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}