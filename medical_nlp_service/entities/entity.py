# Pydantic Models (unchanged)
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import base64
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
import re

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

class ProcessAudioRequest(BaseModel):
    audio_data: str = Field(..., min_length=100, description="Base64 encoded audio data")
    file_name: str = Field(..., description="Original filename")

    @validator('file_name')
    def validate_file_name(cls, v):
        if not re.match(r'^[\w\s\-\.]+$', v):
            raise ValueError('Filename contains invalid characters')
        return v

    @validator('audio_data')
    def validate_audio_data(cls, v):
        try:
            decoded = base64.b64decode(v, validate=True)
            if len(decoded) > MAX_AUDIO_BYTES:
                raise ValueError(f"Audio data exceeds maximum size of {MAX_AUDIO_BYTES} bytes")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {str(e)}")

class ProcessAudioResponse(BaseModel):
    status: str = Field(..., description="Processing status")
    transcript: str = Field("", description="Transcribed text")
    entities: List[MedicalEntity] = Field(default_factory=list, description="Extracted medical entities")
    soap_note: str = Field("", description="Generated SOAP note")
    speaker_segments: List[SpeakerSegment] = Field(default_factory=list, description="Speaker diarization segments")  # Add speaker segments
    error: Optional[str] = Field(None, description="Error message if any")
    model_used: str = Field("", description="ASR model used")
    nlu_model_used: str = Field("", description="NLU model used")
    llm_model_used: str = Field("", description="LLM model used for SOAP generation")
    diarization_model_used: str = Field("", description="Diarization model used")  # Add diarization model info
    request_id: str = Field(..., description="Unique request identifier")