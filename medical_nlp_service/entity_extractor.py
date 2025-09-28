from typing import List
from entities import MedicalEntity
from constants import MEDICAL_KEYWORDS, MEDICATION_SYNONYMS
import re
import logging

logger = logging.getLogger("medical-nlp-service")

# Keyword-based entity extraction (unchanged)
def extract_entities_keywords(text: str) -> List[MedicalEntity]:
    """Enhanced keyword extraction with partial matching"""
    entities = []
    text_lower = text.lower()
    matched_positions = set()
    
    # Check for medication synonyms first
    for misspelling, canonical in MEDICATION_SYNONYMS.items():
        pattern = r"\b" + re.escape(misspelling) + r"\b"
        for match in re.finditer(pattern, text_lower):
            start, end = match.start(), match.end()
            position_key = (start, end)
            if position_key not in matched_positions:
                entities.append(MedicalEntity(
                    entity="MEDICATION",
                    text=canonical,  # Use canonical name
                    start=start,
                    end=end,
                    confidence=0.85  # High confidence for known synonyms
                ))
                matched_positions.add(position_key)
    
    # Then check regular medical keywords
    for entity_type, keywords in MEDICAL_KEYWORDS.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            for match in re.finditer(pattern, text_lower):
                start, end = match.start(), match.end()
                
                position_key = (start, end)
                if position_key not in matched_positions:
                    entities.append(MedicalEntity(
                        entity=entity_type,
                        text=text[start:end],
                        start=start,
                        end=end,
                        confidence=0.8
                    ))
                    matched_positions.add(position_key)
    
    return entities

def deduplicate_entities(entities: List[MedicalEntity]) -> List[MedicalEntity]:
    if not entities:
        return []
    
    # Sort by start position and length (longer first)
    entities.sort(key=lambda x: (x.start, -(x.end - x.start)))
    
    unique_entities = []
    seen_texts = set()
    
    for entity in entities:
        # Normalize text for comparison
        normalized_text = entity.text.lower().strip()
        
        # Check for exact duplicates
        if normalized_text in seen_texts:
            continue
            
        # Check for overlapping entities (keep the longer one)
        overlapping = False
        for selected in unique_entities:
            if (entity.start < selected.end and entity.end > selected.start):
                # If current entity is longer, replace the existing one
                if (entity.end - entity.start) > (selected.end - selected.start):
                    unique_entities.remove(selected)
                    seen_texts.discard(selected.text.lower().strip())
                else:
                    overlapping = True
                break
        
        if not overlapping:
            unique_entities.append(entity)
            seen_texts.add(normalized_text)
    
    return unique_entities

def filter_negated_entities(transcript: str, entities: List[MedicalEntity]) -> List[MedicalEntity]:
    """Filter out entities that are mentioned in negative context"""
    filtered_entities = []
    transcript_lower = transcript.lower()
    
    negation_phrases = [
        "no ", "not ", "denies ", "denied ", "without ", "negative for ",
        "never ", "none ", "doesn't have ", "haven't had "
    ]
    
    for entity in entities:
        entity_text = entity.text.lower()
        start, end = entity.start, entity.end
        
        # Check if the entity appears in a negative context
        context_start = max(0, start - 50)
        context_end = min(len(transcript), end + 20)
        context = transcript_lower[context_start:context_end]
        
        is_negated = any(neg in context for neg in negation_phrases)
        
        if not is_negated:
            filtered_entities.append(entity)
        else:
            logger.info(f"Filtered out negated entity: {entity.text}")
    
    return filtered_entities