from typing import Dict, List
from shared_models import MedicalEntity

class ProgressiveSOAPBuilder:
    """Builds SOAP notes progressively as conversation unfolds"""
    
    def __init__(self):
        self.section_templates = {
            "subjective": {
                "symptom_reporting": "Patient reports {symptoms}.",
                "history_sharing": "Relevant history: {history}.",
                "medication_discussion": "Current medications: {medications}."
            }
        }
    
    def update_soap_sections(self, current_sections: Dict[str, str], 
                           transcript: str, entities: List[MedicalEntity],
                           conversation_phase: str) -> Dict[str, str]:
        """Update SOAP sections based on new content and conversation context"""
        updated_sections = current_sections.copy()
        
        # Analyze content and update appropriate sections
        if conversation_phase == "symptom_reporting":
            updated_sections["subjective"] = self._add_to_section(
                updated_sections["subjective"], 
                self._extract_symptom_content(transcript, entities)
            )
        
        elif conversation_phase == "history_taking":
            updated_sections["subjective"] = self._add_to_section(
                updated_sections["subjective"],
                self._extract_history_content(transcript, entities)
            )
        
        elif conversation_phase == "questioning":
            # Doctor questions might go to objective or help structure assessment
            updated_sections["objective"] = self._add_to_section(
                updated_sections["objective"],
                self._extract_clinical_findings(transcript, entities)
            )
        
        # Always update assessment and plan based on new information
        updated_sections["assessment"] = self._update_assessment(updated_sections, entities)
        updated_sections["plan"] = self._update_plan(updated_sections, entities)
        
        return updated_sections
    
    def _add_to_section(self, current_content: str, new_content: str) -> str:
        """Add new content to a section without duplication"""
        if not new_content.strip():
            return current_content
        
        if new_content in current_content:
            return current_content
        
        if current_content:
            return current_content + ". " + new_content
        else:
            return new_content
    
    def _extract_symptom_content(self, transcript: str, entities: List[MedicalEntity]) -> str:
        """Extract symptom-related content for subjective section"""
        symptoms = [e.text for e in entities if e.entity == "SYMPTOM"]
        if symptoms:
            return f"Reports {', '.join(symptoms)}"
        return ""
    
    def _extract_history_content(self, transcript: str, entities: List[MedicalEntity]) -> str:
        """Extract history-related content"""
        medications = [e.text for e in entities if e.entity == "MEDICATION"]
        conditions = [e.text for e in entities if e.entity in ["DIAGNOSIS", "CONDITION"]]
        
        parts = []
        if medications:
            parts.append(f"Current medications: {', '.join(medications)}")
        if conditions:
            parts.append(f"History of {', '.join(conditions)}")
        
        return ". ".join(parts)
    
    def _extract_clinical_findings(self, transcript: str, entities: List[MedicalEntity]) -> str:
        """Extract objective findings from clinical discussion"""
        # This would be enhanced with vital signs, exam findings, etc.
        return "Clinical discussion regarding patient condition."
    
    def _update_assessment(self, sections: Dict[str, str], entities: List[MedicalEntity]) -> str:
        """Update assessment based on current information"""
        symptoms = [e.text for e in entities if e.entity == "SYMPTOM"]
        if symptoms:
            return f"Assessment of {', '.join(symptoms[:2])}"
        return "Ongoing assessment"
    
    def _update_plan(self, sections: Dict[str, str], entities: List[MedicalEntity]) -> str:
        """Update plan based on current information"""
        medications = [e.text for e in entities if e.entity == "MEDICATION"]
        if medications:
            return f"Consider {medications[0]} management"
        return "Continue evaluation and management"