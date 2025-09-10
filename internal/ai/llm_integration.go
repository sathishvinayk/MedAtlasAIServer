package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

// LLMClient handles communication with OpenRouter.ai
type LLMClient struct {
	APIKey     string
	BaseURL    string
	Model      string
	HTTPClient *http.Client
}

// NewLLMClient creates a new OpenRouter.ai client
func NewLLMClient(apiKey, model string) *LLMClient {
	return &LLMClient{
		APIKey:     apiKey,
		BaseURL:    "https://openrouter.ai/api/v1",
		Model:      model,
		HTTPClient: &http.Client{Timeout: 60 * time.Second},
	}
}

// OpenRouterRequest represents the request to OpenRouter.ai
type OpenRouterRequest struct {
	Model       string            `json:"model"`
	Messages    []ChatMessage     `json:"messages"`
	Temperature float64           `json:"temperature"`
	MaxTokens   int               `json:"max_tokens"`
	Stream      bool              `json:"stream"`
	Headers     map[string]string `json:"headers,omitempty"`
}

// OpenRouterResponse represents the response from OpenRouter.ai
type OpenRouterResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
	Model string `json:"model"`
}

// GenerateResponse generates AI-powered response using OpenRouter.ai
func (lc *LLMClient) GenerateResponse(context string, userMessage string, medicalData []string) (string, error) {
	// Build the prompt with medical context
	prompt := lc.buildMedicalPrompt(context, userMessage, medicalData)

	messages := []ChatMessage{
		{
			Role:    "system",
			Content: "You are a medical AI assistant that provides general health information and suggestions based on medical research. You are helpful, cautious, and always recommend consulting healthcare professionals for personal medical advice. Never provide prescriptions or specific dosage advice.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	request := OpenRouterRequest{
		Model:       lc.Model,
		Messages:    messages,
		Temperature: 0.7,
		MaxTokens:   1024,
		Stream:      false,
		Headers: map[string]string{
			"HTTP-Referer": "https://medical-chat-app.com",
			"X-Title":      "Medical AI Assistant",
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", lc.BaseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+lc.APIKey)
	req.Header.Set("HTTP-Referer", "https://medical-chat-app.com")
	req.Header.Set("X-Title", "Medical AI Assistant")

	log.Printf("🤖 Sending request to OpenRouter.ai with model: %s", lc.Model)

	resp, err := lc.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("OpenRouter.ai returned status %d", resp.StatusCode)
	}

	var response OpenRouterResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if response.Error.Message != "" {
		return "", fmt.Errorf("OpenRouter.ai error: %s", response.Error.Message)
	}

	if len(response.Choices) == 0 || response.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("empty response from AI model")
	}

	log.Printf("✅ Received response from model: %s", response.Model)
	return response.Choices[0].Message.Content, nil
}

// buildMedicalPrompt creates a comprehensive prompt for medical conversations
func (lc *LLMClient) buildMedicalPrompt(context, userMessage string, medicalData []string) string {
	var prompt strings.Builder

	prompt.WriteString("MEDICAL AI ASSISTANT ROLE:\n")
	prompt.WriteString("You are a helpful medical AI assistant. Provide evidence-based health information while being cautious and ethical.\n")
	prompt.WriteString("KEY RULES:\n")
	prompt.WriteString("1. NEVER give prescriptions, dosages, or specific medical advice\n")
	prompt.WriteString("2. ALWAYS recommend consulting healthcare professionals\n")
	prompt.WriteString("3. Base responses on medical research when available\n")
	prompt.WriteString("4. Be empathetic and clear in your communication\n")
	prompt.WriteString("5. If unsure, say so and suggest professional consultation\n\n")

	if context != "" {
		prompt.WriteString("CONVERSATION CONTEXT:\n")
		prompt.WriteString(context)
		prompt.WriteString("\n\n")
	}

	prompt.WriteString("USER'S QUESTION: ")
	prompt.WriteString(userMessage)
	prompt.WriteString("\n\n")

	if len(medicalData) > 0 {
		prompt.WriteString("RELEVANT MEDICAL RESEARCH FINDINGS:\n")
		for i, data := range medicalData {
			if i < 3 { // Limit to top 3 findings
				prompt.WriteString(fmt.Sprintf("%d. %s\n", i+1, data))
			}
		}
		prompt.WriteString("\n")
	}

	prompt.WriteString("INSTRUCTIONS:\n")
	prompt.WriteString("1. Provide helpful information based on the context above\n")
	prompt.WriteString("2. Be cautious and avoid giving medical advice\n")
	prompt.WriteString("3. Suggest consulting healthcare professionals\n")
	prompt.WriteString("4. Keep responses conversational and empathetic\n")
	prompt.WriteString("5. If research findings are available, reference them appropriately\n")
	prompt.WriteString("6. Use plain language, avoid overly technical terms\n\n")

	prompt.WriteString("YOUR RESPONSE:")

	return prompt.String()
}

// GetAvailableModels returns available models from OpenRouter.ai
func (lc *LLMClient) GetAvailableModels() ([]string, error) {
	req, err := http.NewRequest("GET", lc.BaseURL+"/models", nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", "Bearer "+lc.APIKey)

	resp, err := lc.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var modelsResponse struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&modelsResponse); err != nil {
		return nil, err
	}

	var models []string
	for _, model := range modelsResponse.Data {
		models = append(models, model.ID)
	}

	return models, nil
}
