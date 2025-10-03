# Pydantic Models (unchanged)
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
import time

class MedicalEntity(BaseModel):
    entity: str = Field(..., description="Type of medical entity (SYMPTOM, MEDICATION, etc.)")
    text: str = Field(..., description="The actual text of the entity")
    start: int = Field(..., description="Start character position in text")
    end: int = Field(..., description="End character position in text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

class EnhancedMedicalEntity(MedicalEntity):
    """Enhanced medical entity with negation and temporal attributes"""
    negated: bool = Field(default=False, description="Whether the entity is negated")
    negation_confidence: float = Field(default=0.0, description="Confidence of negation detection")
    negation_phrase: Optional[str] = Field(default=None, description="The negation phrase found")
    duration: Optional[str] = Field(default=None, description="Duration mentioned with entity")
    normalized_duration: Optional[str] = Field(default=None, description="Normalized duration")
    onset: Optional[str] = Field(default=None, description="Onset time mentioned")
    normalized_onset: Optional[str] = Field(default=None, description="Normalized onset time")
    temporal_context: Optional[str] = Field(default=None, description="Temporal context")

class SpeakerSegment(BaseModel):
    speaker: str = Field(..., description="Speaker identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")

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