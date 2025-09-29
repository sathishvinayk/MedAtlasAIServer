# Pydantic Models (unchanged)
from pydantic import BaseModel, Field, validator
from typing import Dict, Any
from dataclasses import dataclass, field
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
import time

class MedicalEntity(BaseModel):
    entity: str = Field(..., description="Type of medical entity (SYMPTOM, MEDICATION, etc.)")
    text: str = Field(..., description="The actual text of the entity")
    start: int = Field(..., description="Start character position in text")
    end: int = Field(..., description="End character position in text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

class SpeakerSegment(BaseModel):
    speaker: str = Field(..., description="Speaker identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")

@dataclass
class RealtimeResult:
    """Internal real-time processing result"""
    type: str  # "transcript", "entities", "soap_update", "medical_alert"
    data: Dict[str, Any]
    session_id: str
    is_partial: bool = True
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.9