from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple

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
