package ai

import (
	"MedAtlasAIServer/internal/embeddingClient"
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/qdrant/go-client/qdrant"
)

type MedicalChat struct {
	Embedder     *embeddingClient.Client
	QdrantClient qdrant.PointsClient
}

type ChatMessage struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

type ChatResponse struct {
	Response    string   `json:"response"`
	Suggestions []string `json:"suggestions,omitempty"`
}

func NewMedicalChat(embedder *embeddingClient.Client, qdrantClient qdrant.PointsClient) *MedicalChat {
	return &MedicalChat{
		Embedder:     embedder,
		QdrantClient: qdrantClient,
	}
}

func (mc *MedicalChat) ProcessMessage(ctx context.Context, userMessage string, chatHistory []ChatMessage) (*ChatResponse, error) {
	intent := mc.UnderstandIntent(userMessage, chatHistory)
	contextualHistory := mc.ExtractRelevantHistory(chatHistory)

	// Search for relevant medical information
	searchResults, err := mc.SearchMedicalKnowledge(ctx, userMessage, intent)
	if err != nil {
		// Fallback to general response if search fails
		return mc.GenerateFallbackResponse(userMessage, intent), nil
	}

	// Generate conversational AI response
	response := mc.GenerateAIResponse(userMessage, searchResults, intent, contextualHistory)
	suggestions := mc.GenerateHelpfulSuggestions(userMessage, intent, searchResults)
	return &ChatResponse{
		Response:    response,
		Suggestions: suggestions,
	}, nil
}

func (mc *MedicalChat) GenerateAIResponse(userMessage string, searchResults []string, intent string, history []ChatMessage) string {
	switch intent {
	case "symptom_inquiry":
		return mc.GenerateSymptomResponse(userMessage, searchResults)
	case "treatment_info":
		return mc.GenerateTreatmentResponse(userMessage, searchResults)
	case "prevention":
		return mc.GeneratePreventionResponse(userMessage, searchResults)
	case "causes":
		return mc.GenerateCausesResponse(userMessage, searchResults)
	case "diagnosis":
		return mc.GenerateDiagnosisResponse(userMessage, searchResults)
	case "risks":
		return mc.GenerateRisksResponse(userMessage, searchResults)
	case "comparison":
		return mc.GenerateComparisonResponse(userMessage, searchResults)
	case "how_to":
		return mc.GenerateHowToResponse(userMessage, searchResults)
	default:
		return mc.GenerateGeneralResponse(userMessage, searchResults)
	}
}

func (mc *MedicalChat) SearchMedicalKnowledge(ctx context.Context, query string, intent string) ([]string, error) {
	enhancedQuery := mc.EnhanceQueryForIntent(query, intent)

	vector, err := mc.Embedder.GetEmbedding(enhancedQuery)
	if err != nil {
		return nil, err
	}

	searchResult, err := mc.QdrantClient.Search(ctx, &qdrant.SearchPoints{
		CollectionName: "medical_abstracts",
		Vector:         vector,
		Limit:          1, // Fewer, more focused results for chat
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Include{
				Include: &qdrant.PayloadIncludeSelector{
					Fields: []string{"title", "abstract", "journal"},
				},
			},
		},
	})
	if err != nil {
		return nil, err
	}

	var results []string
	for _, point := range searchResult.Result {
		payload := point.Payload
		abstract := safeGetString(payload, "abstract")
		title := safeGetString(payload, "title")
		journal := safeGetString(payload, "journal")

		if abstract != "" {
			results = append(results, fmt.Sprintf("Study: %s (%s) - %s", title, journal, abstract))
		}

	}
	return results, nil
}

func (mc *MedicalChat) EnhanceQueryForIntent(query string, intent string) string {
	intentModifiers := map[string]string{
		"symptom_inquiry": "symptoms clinical presentation signs",
		"treatment_info":  "treatment therapy management clinical trial",
		"prevention":      "prevention risk reduction prophylaxis",
		"causes":          "causes etiology risk factors",
		"diagnosis":       "diagnosis testing assessment criteria",
		"risks":           "risks complications side effects",
		"comparison":      "comparison differences versus",
		"how_to":          "procedure steps process",
	}

	modifier := intentModifiers[intent]
	if modifier != "" {
		return query + " " + modifier
	}
	return query
}

func (mc *MedicalChat) UnderstandIntent(message string, history []ChatMessage) string {
	message = strings.ToLower(message)

	// Analyze intent based on message content
	switch {
	case containsAny(message, []string{"symptom", "pain", "hurt", "feel", "experience", "sign"}):
		return "symptom_inquiry"
	case containsAny(message, []string{"treatment", "medicine", "drug", "therapy", "medication", "cure"}):
		return "treatment_info"
	case containsAny(message, []string{"prevent", "avoid", "reduce risk", "lower chance", "protection"}):
		return "prevention"
	case containsAny(message, []string{"cause", "reason", "why", "how get", "trigger"}):
		return "causes"
	case containsAny(message, []string{"diagnos", "test", "scan", "x-ray", "mri", "blood test"}):
		return "diagnosis"
	case containsAny(message, []string{"side effect", "complication", "risk", "danger"}):
		return "risks"
	case containsAny(message, []string{"what is", "tell me about", "explain", "information about"}):
		return "general_info"
	case containsAny(message, []string{"difference between", "compare", "vs", "versus"}):
		return "comparison"
	case containsAny(message, []string{"how to", "steps", "process", "procedure"}):
		return "how_to"
	default:
		return "general_chat"
	}
}

func (mc *MedicalChat) ExtractRelevantHistory(history []ChatMessage) []ChatMessage {
	if len(history) <= 3 {
		return history
	}
	return history[len(history)-3:]
}

func containsAny(s string, substrs []string) bool {
	for _, substr := range substrs {
		if strings.Contains(s, substr) {
			return true
		}
	}
	return false
}

func safeGetString(payload map[string]*qdrant.Value, key string) string {
	if value, exists := payload[key]; exists && value != nil {
		return value.GetStringValue()
	}
	return ""
}

func (mc *MedicalChat) GenerateSymptomResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific information about those symptoms yet. Could you describe them in more detail?"
	}

	responses := []string{
		"Based on medical research, here's what I found about those symptoms:\n\n",
		"Medical literature discusses several aspects of those symptoms:\n\n",
		"Researchers have studied similar symptoms and found:\n\n",
	}

	response := responses[rand.Intn(len(responses))]

	for i, result := range results {
		if i >= 2 { // Limit to top 2 results
			break
		}
		response += fmt.Sprintf("• %s\n", extractKeyInfo(result, 120))
	}

	response += "\n💡 Remember: I can share general information, but a healthcare professional should evaluate specific symptoms. Would you like me to suggest when to consider seeing a doctor?"

	return response
}

func (mc *MedicalChat) GenerateTreatmentResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific treatment information about that yet. Could you tell me more about what you're looking for?"
	}

	response := "Here's what recent medical research says about treatment approaches:\n\n"

	treatments := make(map[string]bool)
	for _, result := range results {
		if treatment := extractTreatmentInfo(result); treatment != "" && !treatments[treatment] {
			response += fmt.Sprintf("• %s\n", treatment)
			treatments[treatment] = true
		}
		if len(treatments) >= 3 {
			break
		}
	}

	response += "\n🔬 These are general approaches discussed in research. Treatment decisions should always be made with a healthcare provider."

	return response
}

func (mc *MedicalChat) GenerateHelpfulSuggestions(userMessage string, intent string, results []string) []string {
	suggestions := []string{
		"Consult with a healthcare professional for personalized advice",
		"Keep track of your questions for your next medical appointment",
	}

	// Intent-specific suggestions
	switch intent {
	case "symptom_inquiry":
		suggestions = append(suggestions,
			"Consider noting when symptoms occur and what makes them better or worse",
			"Research shows that symptom diaries can be very helpful for medical consultations",
		)
	case "treatment_info":
		suggestions = append(suggestions,
			"Discuss potential treatment options and their benefits/risks with your doctor",
			"Ask about both traditional and newer approaches that might be available",
		)
	case "prevention":
		suggestions = append(suggestions,
			"Consider working with a healthcare provider on a personalized prevention plan",
			"Ask about screening tests that might be appropriate for your situation",
		)
	case "causes":
		suggestions = append(suggestions,
			"Discuss your specific risk factors with a healthcare provider",
			"Ask about lifestyle modifications that might address underlying causes",
		)
	case "diagnosis":
		suggestions = append(suggestions,
			"Prepare a list of your symptoms and concerns before your appointment",
			"Ask your doctor about the diagnostic process and what to expect",
		)
	case "risks":
		suggestions = append(suggestions,
			"Discuss your personal risk profile with a healthcare provider",
			"Ask about risk reduction strategies tailored to your situation",
		)
	case "comparison":
		suggestions = append(suggestions,
			"Discuss the pros and cons of different options with your doctor",
			"Consider which factors are most important for your specific situation",
		)
	case "how_to":
		suggestions = append(suggestions,
			"Ask a healthcare professional to demonstrate the procedure",
			"Request written instructions or resources for proper technique",
		)
	case "general_info":
		suggestions = append(suggestions,
			"Ask your doctor for reliable resources to learn more",
			"Consider discussing this information at your next check-up",
		)
	}

	return suggestions
}

func (mc *MedicalChat) GenerateFallbackResponse(userMessage string, intent string) *ChatResponse {
	fallbackResponses := map[string]string{
		"symptom_inquiry": "I'm having trouble finding specific information about those symptoms. Could you describe them in more detail? For example, when they started, what makes them better or worse, and any other symptoms you're experiencing?",
		"treatment_info":  "I don't have detailed treatment information about that specific condition right now. You might want to discuss treatment options with a healthcare provider who can consider your individual situation.",
		"prevention":      "I'm still learning about prevention strategies for that specific concern. Prevention approaches often depend on individual risk factors that would be best discussed with a healthcare provider.",
		"causes":          "I don't have specific information about the causes of that condition yet. The causes of medical conditions can be complex and multifactorial, often requiring professional evaluation.",
		"diagnosis":       "I don't have specific diagnostic information about that condition. Diagnosis typically involves medical evaluation, so this would be best discussed with a healthcare provider.",
		"risks":           "I don't have detailed risk information about that yet. Risk assessment usually requires consideration of individual factors that would be best evaluated by a healthcare professional.",
		"comparison":      "I don't have specific comparison information about those topics yet. Comparisons in medicine often depend on individual circumstances and latest research findings.",
		"how_to":          "I don't have specific procedural information about that. Medical procedures should always be demonstrated and supervised by qualified healthcare professionals.",
		"general_info":    "I'm still learning about that topic. Could you ask me about something else, or try rephrasing your question?",
		"general_chat":    "I'm here to help with medical information and health-related questions. Would you like to ask about symptoms, treatments, prevention, or general health topics?",
	}

	response := fallbackResponses[intent]
	if response == "" {
		response = "I'm not sure how to help with that yet. Could you try asking about symptoms, treatments, prevention, or general medical information?"
	}

	return &ChatResponse{
		Response: response,
		Suggestions: []string{
			"Try asking about specific symptoms or conditions",
			"Ask about prevention strategies or general health information",
			"Consult a healthcare professional for personalized advice",
			"Try rephrasing your question or providing more details",
		},
	}
}

func extractKeyInfo(text string, maxLength int) string {
	sentences := strings.Split(text, ".")
	if len(sentences) > 0 {
		return strings.TrimSpace(sentences[0])
	}
	if len(text) > maxLength {
		return text[:maxLength] + "..."
	}
	return text
}

func extractTreatmentInfo(text string) string {
	// Simple extraction of treatment information
	lowerText := strings.ToLower(text)
	if strings.Contains(lowerText, "treatment") || strings.Contains(lowerText, "therapy") {
		// Extract the sentence containing treatment info
		sentences := strings.Split(text, ".")
		for _, sentence := range sentences {
			if strings.Contains(strings.ToLower(sentence), "treatment") ||
				strings.Contains(strings.ToLower(sentence), "therapy") {
				return strings.TrimSpace(sentence)
			}
		}
	}
	return ""
}

// GeneratePreventionResponse creates responses for prevention questions
func (mc *MedicalChat) GeneratePreventionResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific prevention information about that yet. Could you tell me what specific aspect you're interested in preventing?"
	}

	responses := []string{
		"Medical research suggests several prevention strategies:\n\n",
		"Here are evidence-based prevention approaches from recent studies:\n\n",
		"Based on clinical research, these prevention methods show promise:\n\n",
	}

	response := responses[rand.Intn(len(responses))]

	preventionMethods := make(map[string]bool)
	for _, result := range results {
		if method := extractPreventionInfo(result); method != "" && !preventionMethods[method] {
			response += fmt.Sprintf("• %s\n", method)
			preventionMethods[method] = true
		}
		if len(preventionMethods) >= 3 {
			break
		}
	}

	response += "\n🛡️ Prevention strategies are most effective when tailored to individual risk factors and implemented consistently."

	return response
}

// GenerateCausesResponse creates responses for cause-related questions
func (mc *MedicalChat) GenerateCausesResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific information about the causes of that condition yet. Could you provide more details about what you're wondering about?"
	}

	response := "Medical research has identified several potential causes and risk factors:\n\n"

	causes := make(map[string]bool)
	for _, result := range results {
		if cause := extractCauseInfo(result); cause != "" && !causes[cause] {
			response += fmt.Sprintf("• %s\n", cause)
			causes[cause] = true
		}
		if len(causes) >= 3 {
			break
		}
	}

	response += "\n🔍 Understanding causes helps researchers develop better treatments and prevention strategies."

	return response
}

// GenerateDiagnosisResponse creates responses for diagnosis questions
func (mc *MedicalChat) GenerateDiagnosisResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific diagnostic information about that condition. Diagnosis typically involves medical evaluation, so this would be best discussed with a healthcare provider."
	}

	responses := []string{
		"Diagnostic approaches discussed in medical literature include:\n\n",
		"Clinical guidelines suggest these diagnostic methods:\n\n",
		"Research indicates these diagnostic criteria are commonly used:\n\n",
	}

	response := responses[rand.Intn(len(responses))]

	diagnosticMethods := make(map[string]bool)
	for _, result := range results {
		if method := extractDiagnosticInfo(result); method != "" && !diagnosticMethods[method] {
			response += fmt.Sprintf("• %s\n", method)
			diagnosticMethods[method] = true
		}
		if len(diagnosticMethods) >= 3 {
			break
		}
	}

	response += "\n🏥 Diagnosis should always be made by qualified healthcare professionals using comprehensive evaluation."

	return response
}

// GenerateRisksResponse creates responses for risk-related questions
func (mc *MedicalChat) GenerateRisksResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific risk information about that yet. Risk factors can vary widely depending on individual circumstances."
	}

	response := "Medical studies have identified these potential risks and considerations:\n\n"

	risks := make(map[string]bool)
	for _, result := range results {
		if risk := extractRiskInfo(result); risk != "" && !risks[risk] {
			response += fmt.Sprintf("• %s\n", risk)
			risks[risk] = true
		}
		if len(risks) >= 3 {
			break
		}
	}

	response += "\n⚠️ Understanding risks helps in making informed decisions and discussing concerns with healthcare providers."

	return response
}

// GenerateComparisonResponse creates responses for comparison questions
func (mc *MedicalChat) GenerateComparisonResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific comparison information about those topics yet. Comparisons in medicine often depend on individual factors and latest research."
	}

	response := "Based on medical literature, here's how these compare:\n\n"

	// Extract comparison points
	comparisonPoints := extractComparisonInfo(results)
	for i, point := range comparisonPoints {
		if i >= 4 { // Limit comparison points
			break
		}
		response += fmt.Sprintf("• %s\n", point)
	}

	response += "\n📊 Comparisons in medicine require considering individual circumstances and latest evidence."

	return response
}

// GenerateHowToResponse creates responses for procedural questions
func (mc *MedicalChat) GenerateHowToResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific procedural information about that. Medical procedures should always be demonstrated and supervised by qualified professionals."
	}

	response := "Medical protocols typically involve these steps:\n\n"

	steps := extractProcedureSteps(results)
	for i, step := range steps {
		response += fmt.Sprintf("%d. %s\n", i+1, step)
		if i >= 4 { // Limit to 5 steps
			break
		}
	}

	response += "\n👩‍⚕️ Medical procedures require proper training and should only be performed by qualified healthcare professionals."

	return response
}

// GenerateGeneralResponse creates responses for general information questions
func (mc *MedicalChat) GenerateGeneralResponse(userMessage string, results []string) string {
	if len(results) == 0 {
		return "I don't have specific information about that topic yet. Could you ask about something else, or try rephrasing your question?"
	}

	responses := []string{
		"Here's what medical research shows about that:\n\n",
		"Based on current medical understanding:\n\n",
		"Medical literature discusses this topic in these ways:\n\n",
	}

	response := responses[rand.Intn(len(responses))]

	// Use the most relevant result
	if len(results) > 0 {
		keyInfo := extractKeyInfo(results[0], 150)
		response += fmt.Sprintf("• %s\n", keyInfo)
	}

	if len(results) > 1 {
		response += fmt.Sprintf("• %s\n", extractKeyInfo(results[1], 100))
	}

	response += "\n📚 Medical knowledge evolves rapidly, so current understanding may change with new research."

	return response
}

// Helper extraction functions
func extractPreventionInfo(text string) string {
	lowerText := strings.ToLower(text)
	if strings.Contains(lowerText, "prevent") || strings.Contains(lowerText, "reduc risk") ||
		strings.Contains(lowerText, "avoid") || strings.Contains(lowerText, "prophylaxis") {

		sentences := strings.Split(text, ".")
		for _, sentence := range sentences {
			lowerSentence := strings.ToLower(sentence)
			if strings.Contains(lowerSentence, "prevent") || strings.Contains(lowerSentence, "reduc risk") ||
				strings.Contains(lowerSentence, "avoid") {
				return strings.TrimSpace(sentence)
			}
		}
	}
	return ""
}

func extractCauseInfo(text string) string {
	lowerText := strings.ToLower(text)
	if strings.Contains(lowerText, "cause") || strings.Contains(lowerText, "due to") ||
		strings.Contains(lowerText, "because of") || strings.Contains(lowerText, "risk factor") {

		sentences := strings.Split(text, ".")
		for _, sentence := range sentences {
			lowerSentence := strings.ToLower(sentence)
			if strings.Contains(lowerSentence, "cause") || strings.Contains(lowerSentence, "due to") ||
				strings.Contains(lowerSentence, "because of") {
				return strings.TrimSpace(sentence)
			}
		}
	}
	return ""
}

func extractDiagnosticInfo(text string) string {
	lowerText := strings.ToLower(text)
	if strings.Contains(lowerText, "diagnos") || strings.Contains(lowerText, "test") ||
		strings.Contains(lowerText, "scan") || strings.Contains(lowerText, "criteria") {

		sentences := strings.Split(text, ".")
		for _, sentence := range sentences {
			lowerSentence := strings.ToLower(sentence)
			if strings.Contains(lowerSentence, "diagnos") || strings.Contains(lowerSentence, "test") ||
				strings.Contains(lowerSentence, "scan") {
				return strings.TrimSpace(sentence)
			}
		}
	}
	return ""
}

func extractRiskInfo(text string) string {
	lowerText := strings.ToLower(text)
	if strings.Contains(lowerText, "risk") || strings.Contains(lowerText, "complication") ||
		strings.Contains(lowerText, "side effect") || strings.Contains(lowerText, "adverse") {

		sentences := strings.Split(text, ".")
		for _, sentence := range sentences {
			lowerSentence := strings.ToLower(sentence)
			if strings.Contains(lowerSentence, "risk") || strings.Contains(lowerSentence, "complication") ||
				strings.Contains(lowerSentence, "side effect") {
				return strings.TrimSpace(sentence)
			}
		}
	}
	return ""
}

func extractComparisonInfo(results []string) []string {
	var comparisons []string
	for _, result := range results {
		lowerResult := strings.ToLower(result)
		if strings.Contains(lowerResult, "vs") || strings.Contains(lowerResult, "versus") ||
			strings.Contains(lowerResult, "compare") || strings.Contains(lowerResult, "difference") {

			sentences := strings.Split(result, ".")
			for _, sentence := range sentences {
				lowerSentence := strings.ToLower(sentence)
				if strings.Contains(lowerSentence, "vs") || strings.Contains(lowerSentence, "versus") ||
					strings.Contains(lowerSentence, "compare") || strings.Contains(lowerSentence, "difference") {
					comparisons = append(comparisons, strings.TrimSpace(sentence))
				}
			}
		}
	}
	return comparisons
}

func extractProcedureSteps(results []string) []string {
	var steps []string
	stepPatterns := []string{
		"first", "second", "third", "then", "next", "finally", "step", "procedure",
	}

	for _, result := range results {
		lowerResult := strings.ToLower(result)
		if strings.Contains(lowerResult, "procedure") || strings.Contains(lowerResult, "method") ||
			strings.Contains(lowerResult, "technique") || strings.Contains(lowerResult, "protocol") {

			sentences := strings.Split(result, ".")
			for _, sentence := range sentences {
				lowerSentence := strings.ToLower(sentence)
				for _, pattern := range stepPatterns {
					if strings.Contains(lowerSentence, pattern) {
						steps = append(steps, strings.TrimSpace(sentence))
						break
					}
				}
			}
		}
	}
	return steps
}
