# models/__init__.py
from .entity import MedicalEntity, SpeakerSegment, EmbedRequest, EmbedResponse, ProcessAudioRequest, ProcessAudioResponse

__all__ = [
    'MedicalEntity',
    'SpeakerSegment',
    'EmbedRequest',
    'EmbedResponse',
    'ProcessAudioRequest',
    'ProcessAudioResponse'
]