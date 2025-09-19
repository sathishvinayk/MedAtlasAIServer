def map_spacy_label_to_medical(label: str) -> str:
    mapping = {
        "DISEASE": "DIAGNOSIS",
        "CONDITION": "DIAGNOSIS",
        "SYMPTOM": "SYMPTOM",
        "MEDICATION": "MEDICATION",
        "DRUG": "MEDICATION",
        "BODY_PART": "BODY_PART",
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "DATE": "DATE",
        "TIME": "TIME"
    }
    return mapping.get(label, "OTHER")