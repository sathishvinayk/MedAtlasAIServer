from typing import Tuple

# Utility functions (unchanged except for SOAP generation)
def normalize_medication_name(med_name: str) -> Tuple[str, float]:
    """Normalize medication name and return with adjusted confidence"""
    original_confidence = 0.9  # Default confidence
    
    med_name_lower = med_name.lower().strip()
    
    # Common medication misspellings and corrections
    medication_corrections = {
        "laciniprol": ("lisinopril", 0.8),  # Lower confidence for corrected spellings
        "metforman": ("metformin", 0.8),
        "ibuprofin": ("ibuprofen", 0.8),
        "amoxicilin": ("amoxicillin", 0.8),
        "asprin": ("aspirin", 0.8),
    }
    
    if med_name_lower in medication_corrections:
        corrected_name, confidence = medication_corrections[med_name_lower]
        return corrected_name, confidence
    
    # For properly spelled medications, keep original confidence
    return med_name, original_confidence

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, preserving word boundaries"""
    if len(text) <= max_length:
        return text
    
    # Find the last space within the limit
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."