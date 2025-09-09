package safety

type SafetyResult struct {
	IsSafe    bool     `json:"is_safe"`
	Reasons   []string `json:"reasons,omitempty"`
	RiskLevel string   `json:"risk_level"` //Low, medium and high
}

type MedicalSafetyChecker struct {
	BlockedTopics      []string
	HighRiskKeywords   []string
	MediumRiskKeywords []string
}

func NewMedicalSafetyChecker() *MedicalSafetyChecker {
	return &MedicalSafetyChecker{
		BlockedTopics: []string{
			"emergency", "911", "suicide", "self-harm", "overdose",
			"immediate help", "right now", "urgent", "dying",
		},
		HighRiskKeywords: []string{
			"prescription", "dosage", "how much", "how many", "take",
			"dose", "mg", "milligram", "self-medicate", "without doctor",
			"diagnose me", "what do I have", "what's wrong with me",
		},
		MediumRiskKeywords: []string{
			"treatment for", "cure for", "medicine for", "drug for",
			"should I take", "recommend medication", "best drug",
		},
	}
}
