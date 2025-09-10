package safety

import "strings"

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

func (msc *MedicalSafetyChecker) CheckMessage(message string) SafetyResult {
	lowerMessage := strings.ToLower(message)

	// Check for emergency situations

	for _, topic := range msc.BlockedTopics {
		if strings.Contains(lowerMessage, topic) {
			return SafetyResult{
				IsSafe:    false,
				Reasons:   []string{"emergency_or_crisis_content"},
				RiskLevel: "high",
			}
		}
	}

	// Check for high-risk requests
	for _, keywords := range msc.HighRiskKeywords {
		if strings.Contains(lowerMessage, keywords) {
			return SafetyResult{
				IsSafe:    false,
				Reasons:   []string{"treatment_prescription_request"},
				RiskLevel: "high",
			}
		}
	}

	mediumRiskFound := false
	for _, keyword := range msc.MediumRiskKeywords {
		if strings.Contains(lowerMessage, keyword) {
			mediumRiskFound = true
			break
		}
	}

	if mediumRiskFound {
		return SafetyResult{
			IsSafe:    true,
			RiskLevel: "medium",
			Reasons:   []string{"treatment_inquiry_detected"},
		}
	}
	return SafetyResult{
		IsSafe:    true,
		RiskLevel: "low",
	}
}

func (msc *MedicalSafetyChecker) GenerateSafetyResponse(riskLevel string, reasons []string) string {
	switch riskLevel {
	case "high":
		return "I'm sorry, I cannot provide specific medical advice or emergency guidance. Please contact emergency services (911) or your healthcare provider immediately for urgent medical concerns."
	case "medium":
		return "I can provide general information about medical topics, but I cannot recommend specific treatments or medications. It's important to consult with a healthcare professional for personalized medical advice."
	default:
		return ""
	}
}
