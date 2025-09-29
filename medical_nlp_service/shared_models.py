# Pydantic Models (unchanged)
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from typing import List, Optional
import base64
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
import re
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

class StartStreamingRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    sample_rate: int = Field(16000, description="Audio sample rate")
    channels: int = Field(1, description="Audio channels")

class StartStreamResponse(BaseModel):
    status: str
    session_id: str
    message: str

class StreamAudioRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    audio_chunk: str = Field(..., description="Base64 encoded audio chunk")
    is_final: bool = Field(False, description="Is this the final chunk?")

class StreamAudioResponse(BaseModel):
    status: str
    partial_transcript: str = ""
    entities: List[MedicalEntity] = []
    soap_note: str = ""
    is_complete: bool = False

class EndStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session to end")

class EndStreamResponse(BaseModel):
    status: str
    final_transcript: str = ""
    entities: List[MedicalEntity] = []
    soap_note: str = ""
    session_duration: float

@dataclass
class RealtimeResult:
    """Internal real-time processing result"""
    type: str  # "transcript", "entities", "soap_update", "medical_alert"
    data: Dict[str, Any]
    session_id: str
    is_partial: bool = True
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.9