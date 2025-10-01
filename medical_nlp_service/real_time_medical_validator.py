from typing import List, Dict
class RealTimeMedicalValidator:
    """Validates medical content in real-time"""
    
    def __init__(self):
        self.dangerous_combinations = [
            ("warfarin", "aspirin"),
            ("lisinopril", "ibuprofen"), 
            ("lisinopril", "naproxen"),  # Add more NSAIDs
            ("ace_inhibitor", "nsaid"),   # Class-level interactions
            ("metformin", "alcohol"),
            ("simvastatin", "grapefruit"),
            ("digoxin", "furosemide"),
            ("levothyroxine", "calcium"),
            ("phenytoin", "warfarin"),
            # Add ACE inhibitor specific interactions
            ("ace_inhibitor", "potassium_sparing_diuretics"),
            ("ace_inhibitor", "lithium")
        ]
        
        self.red_flag_symptoms = [
            "chest pain", "shortness of breath", "severe headache",
            "uncontrolled bleeding", "loss of consciousness",
            "sudden weakness", "difficulty speaking", "severe abdominal pain"
        ]

        self.contraindications = {
            "beta_blockers": ["asthma", "copd"],
            "ace_inhibitors": ["pregnancy", "angioedema"],
            "nsaids": ["peptic ulcer", "kidney disease"],
            "statins": ["liver disease", "pregnancy"]
        }
    
    def _map_to_medication_class(self, medication: str) -> List[str]:
        """Map specific medications to their drug classes"""
        medication = medication.lower()
        classes = []
        
        if any(ace in medication for ace in ['lisinopril', 'enalapril', 'ramipril', 'ace inhibitor']):
            classes.append('ace_inhibitor')
        if any(arb in medication for arb in ['losartan', 'valsartan', 'arb']):
            classes.append('arb')
        if any(nsaid in medication for nsaid in ['ibuprofen', 'naproxen', 'nsaid']):
            classes.append('nsaid')
        if 'statin' in medication:
            classes.append('statin')
        
        return classes

    def check_ace_inhibitor_safety(self, medications: List[str], symptoms: List[str]) -> List[Dict[str, str]]:
        """Specific safety checks for ACE inhibitors"""
        alerts = []
        
        # Convert to lowercase for case-insensitive matching
        meds_lower = [med.lower() for med in medications]
        symptoms_lower = [symptom.lower() for symptom in symptoms]
        
        # Check if any ACE inhibitors are present
        ace_medications = []
        for med in medications:
            med_lower = med.lower()
            if any(ace in med_lower for ace in ['lisinopril', 'enalapril', 'ramipril', 'ace inhibitor', 'ace']):
                ace_medications.append(med)
        
        if ace_medications:
            # Check for ACE inhibitor cough
            if 'cough' in symptoms_lower:
                alerts.append({
                    "type": "side_effect_alert",
                    "message": f"ACE inhibitor ({', '.join(ace_medications)}) may be causing persistent cough",
                    "severity": "moderate",
                    "entities": ace_medications,
                    "recommendation": "Consider switching to ARB if cough persists"
                })
            
            # Check for hyperkalemia risk symptoms
            hyperkalemia_symptoms = ['weakness', 'fatigue', 'palpitations', 'tired', 'dizziness']
            if any(symptom in ' '.join(symptoms_lower) for symptom in hyperkalemia_symptoms):
                alerts.append({
                    "type": "safety_alert", 
                    "message": f"ACE inhibitor use with these symptoms may indicate hyperkalemia",
                    "severity": "high",
                    "entities": ace_medications,
                    "recommendation": "Check potassium levels and renal function"
                })
            
            # Check for angioedema risk
            angioedema_terms = ['swelling', 'swollen', 'angioedema', 'face swelling', 'lip swelling']
            if any(term in ' '.join(symptoms_lower) for term in angioedema_terms):
                alerts.append({
                    "type": "safety_alert",
                    "message": "Possible angioedema with ACE inhibitor use",
                    "severity": "urgent",
                    "entities": ace_medications,
                    "recommendation": "Discontinue ACE inhibitor immediately and seek emergency care"
                })
        
        return alerts
    
    def validate_medication_safety(self, medications: List[str]) -> List[Dict[str, str]]:
        """Check for dangerous medication combinations with class mapping"""
        alerts = []
        meds_lower = [med.lower() for med in medications]
        
        # Check exact medication matches
        for med1, med2 in self.dangerous_combinations:
            if med1 in meds_lower and med2 in meds_lower:
                alerts.append({
                    "type": "drug_interaction",
                    "message": f"Potential dangerous interaction between {med1} and {med2}",
                    "severity": "high",
                    "entities": [med1, med2],
                    "recommendation": "Monitor closely or consider alternative medications"
                })
        
        # Check medication class interactions
        all_classes = []
        for med in medications:
            all_classes.extend(self._map_to_medication_class(med))
        
        for class1, class2 in self.dangerous_combinations:
            if class1 in all_classes and class2 in all_classes:
                involved_meds = [med for med in medications 
                            if class1 in self._map_to_medication_class(med) or 
                                class2 in self._map_to_medication_class(med)]
                alerts.append({
                    "type": "drug_interaction",
                    "message": f"Potential class interaction between {class1} and {class2}",
                    "severity": "moderate", 
                    "entities": involved_meds,
                    "recommendation": f"Monitor for {class1}-{class2} interactions"
                })
        
        return alerts
    
    def check_red_flags(self, symptoms: List[str], transcript: str) -> List[Dict[str, str]]:
        """Check for red flag symptoms requiring urgent attention"""
        alerts = []
        transcript_lower = transcript.lower()
        
        for red_flag in self.red_flag_symptoms:
            if red_flag in transcript_lower:
                alerts.append({
                    "type": "red_flag_symptom", 
                    "message": f"Red flag symptom detected: {red_flag}",
                    "severity": "urgent",
                    "entities": [red_flag],
                    "recommendation": "Requires immediate medical attention"
                })
        
        return alerts

    def check_contraindications(self, medication: str, conditions: List[str]) -> List[Dict[str, str]]:
        """Check medication contraindications with patient conditions"""
        alerts = []
        med_lower = medication.lower()
        
        for med_class, contra_conditions in self.contraindications.items():
            if med_class in med_lower:
                for condition in conditions:
                    if condition.lower() in contra_conditions:
                        alerts.append({
                            "type": "contraindication",
                            "message": f"Medication {medication} may be contraindicated with {condition}",
                            "severity": "high",
                            "entities": [medication, condition],
                            "recommendation": "Consult prescribing guidelines"
                        })
        
        return alerts