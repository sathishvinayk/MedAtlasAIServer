import re
class MedicalConversationAnalyzer:
    """Analyzes clinical conversation patterns in real-time"""
    
    def __init__(self):
        self.doctor_phrases = [
            r"how are you feeling", r"any pain", r"describe the", r"when did",
            r"where does it hurt", r"rate your pain", r"any medications",
            r"medical history", r"any allergies", r"what brings you"
        ]
        self.patient_phrases = [
            r"i have", r"i feel", r"my pain", r"it hurts", r"i take",
            r"i was diagnosed", r"my doctor said", r"i've been"
        ]
        self.history_phrases = [
            r"history of", r"diagnosed with", r"previous", r"past medical",
            r"for the past", r"since last"
        ]
    
    def analyze_conversation_phase(self, transcript: str) -> str:
        """Detect the phase of clinical conversation"""
        transcript_lower = transcript.lower()
        
        doctor_matches = sum(1 for phrase in self.doctor_phrases 
                           if re.search(phrase, transcript_lower))
        patient_matches = sum(1 for phrase in self.patient_phrases 
                            if re.search(phrase, transcript_lower))
        history_matches = sum(1 for phrase in self.history_phrases 
                            if re.search(phrase, transcript_lower))
        
        if history_matches > 2:
            return "history_taking"
        elif doctor_matches > patient_matches:
            return "questioning"
        elif patient_matches > doctor_matches:
            return "symptom_reporting"
        else:
            return "general"
