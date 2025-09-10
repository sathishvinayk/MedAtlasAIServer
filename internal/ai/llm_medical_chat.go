package ai

import (
	"MedAtlasAIServer/internal/embeddingClient"
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/qdrant/go-client/qdrant"
)

type LLMMedicalChat struct {
	Embedder     *embeddingClient.Client
	QdrantClient qdrant.PointsClient
	LLMClient    *LLMClient
	UseRealAI    bool
}

func NewLLMMedicalChat(embedder *embeddingClient.Client, qdrantClient qdrant.PointsClient, llmClient *LLMClient) *LLMMedicalChat {
	return &LLMMedicalChat{
		Embedder:     embedder,
		QdrantClient: qdrantClient,
		LLMClient:    llmClient,
		UseRealAI:    llmClient != nil,
	}
}

func (llm *LLMMedicalChat) ProcessMessage(ctx context.Context, userMessage string, chatHistory []ChatMessage) (*ChatResponse, error) {
	intent := llm.UnderstandIntent(userMessage, chatHistory)

	// Search for relevant medical information
	searchResults, err := llm.SearchMedicalKnowledge(ctx, userMessage, intent)
	if err != nil {
		log.Printf("Search failed: %v, using fallback", err)
		searchResults = []string{} // Empty results for fallback
	}
	conversationContext := llm.BuildConversationContext(chatHistory)

	var response string
	var suggestions []string

	if llm.UseRealAI && llm.LLMClient != nil {
		aiResponse, err := llm.LLMClient.GenerateResponse(conversationContext, userMessage, searchResults)
		if err != nil {
			log.Printf("AI generation failed: %v, using local fallback", err)
			response = llm.GenerateLocalResponse(userMessage, searchResults, intent)
		} else {
			response = aiResponse
		}
	} else {
		response = llm.GenerateLocalResponse(userMessage, searchResults, intent)
	}
	suggestions = llm.GenerateHelpfulSuggestions(intent)
	return &ChatResponse{
		Response:    response,
		Suggestions: suggestions,
	}, nil
}

func (llm *LLMMedicalChat) BuildConversationContext(history []ChatMessage) string {
	if len(history) == 0 {
		return ""
	}
	var context strings.Builder
	context.WriteString("Previous conversation:\n")

	start := len(history) - 3
	if start < 0 {
		start = 0
	}
	for i := start; i < len(history); i++ {
		msg := history[i]
		context.WriteString(fmt.Sprintf("%s: %s\n", strings.ToUpper(msg.Role[:1]), msg.Content))
	}

	return context.String()
}

func (llm *LLMMedicalChat) UnderstandIntent(message string, history []ChatMessage) string {
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

func (llm *LLMMedicalChat) SearchMedicalKnowledge(ctx context.Context, query string, intent string) ([]string, error) {
	enhancedQuery := llm.EnhanceQueryForIntent(query, intent)

	vector, err := llm.Embedder.GetEmbedding(enhancedQuery)
	if err != nil {
		return nil, err
	}

	searchResult, err := llm.QdrantClient.Search(ctx, &qdrant.SearchPoints{
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

func (llm *LLMMedicalChat) EnhanceQueryForIntent(query string, intent string) string {
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

// GenerateLocalResponse creates responses without external AI
func (llm *LLMMedicalChat) GenerateLocalResponse(userMessage string, medicalData []string, intent string) string {
	// Enhanced local response generation with medical data
	if len(medicalData) > 0 {
		return llm.GenerateDataDrivenResponse(userMessage, medicalData, intent)
	}

	// Fallback responses
	switch intent {
	case "symptom_inquiry":
		return "I understand you're asking about symptoms. Symptoms can provide important clues about health, but they need to be evaluated in context. Have you discussed these symptoms with a healthcare provider?"
	case "treatment_info":
		return "Treatment approaches vary based on many factors including the specific condition, its severity, and individual health considerations. Medical research emphasizes personalized treatment plans developed with healthcare professionals."
	case "prevention":
		return "Prevention strategies are most effective when tailored to individual risk factors. Research shows that lifestyle modifications, regular screenings, and proactive health management can significantly reduce risks for many conditions."
	default:
		return "I'd be happy to help you with health information. For personalized medical advice, consulting with a healthcare professional who can consider your specific situation would be most appropriate."
	}
}

// GenerateDataDrivenResponse creates responses based on actual medical data
func (llm *LLMMedicalChat) GenerateDataDrivenResponse(userMessage string, medicalData []string, intent string) string {
	var response strings.Builder

	response.WriteString("Based on medical research, ")

	switch intent {
	case "symptom_inquiry":
		response.WriteString("here's what I found about those symptoms:\n\n")
	case "treatment_info":
		response.WriteString("here are some treatment approaches discussed in recent studies:\n\n")
	case "prevention":
		response.WriteString("these prevention strategies show promise according to research:\n\n")
	case "causes":
		response.WriteString("research has identified these potential causes and risk factors:\n\n")
	default:
		response.WriteString("here's relevant information from medical literature:\n\n")
	}

	// Include top medical findings
	for i, data := range medicalData {
		if i >= 2 { // Limit to top 2 findings
			break
		}
		response.WriteString(fmt.Sprintf("• %s\n", llm.SummarizeMedicalFinding(data)))
	}

	response.WriteString("\n💡 This information comes from published medical research. For personalized advice, please consult with a healthcare professional.")

	return response.String()
}

// SummarizeMedicalFinding creates concise summaries of medical data
func (llm *LLMMedicalChat) SummarizeMedicalFinding(data string) string {
	// Extract the most relevant part of the medical finding
	sentences := strings.Split(data, ".")
	if len(sentences) > 0 {
		// Find the most informative sentence
		for _, sentence := range sentences {
			if len(sentence) > 20 && len(sentence) < 150 {
				if strings.Contains(strings.ToLower(sentence), "study") ||
					strings.Contains(strings.ToLower(sentence), "research") ||
					strings.Contains(strings.ToLower(sentence), "found") {
					return strings.TrimSpace(sentence) + "."
				}
			}
		}
		return strings.TrimSpace(sentences[0]) + "."
	}
	return data
}

func (llm *LLMMedicalChat) GenerateHelpfulSuggestions(intent string) []string {
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
