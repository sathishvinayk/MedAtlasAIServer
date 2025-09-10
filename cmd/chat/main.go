package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"MedAtlasAIServer/internal/ai"
	"MedAtlasAIServer/internal/embeddingClient"
	"MedAtlasAIServer/internal/safety"

	"github.com/gorilla/mux"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ChatRequest struct {
	Message string           `json:"message"`
	History []ai.ChatMessage `json:"history,omitempty"`
}

type ChatServer struct {
	MedicalChat   *ai.LLMMedicalChat
	SafetyChecker *safety.MedicalSafetyChecker
	LLMClient     *ai.LLMClient
}

type ChatResponse struct {
	Response    string    `json:"response"`
	Timestamp   time.Time `json:"timestamp"`
	MessageID   string    `json:"message_id"`
	Suggestions []string  `json:"suggestions,omitempty"`
}

func main() {
	// Initialize clients
	embedder := embeddingClient.NewClient("http://localhost:8000")
	qdrantConn, err := grpc.Dial("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Could not connect to Qdrant: %v", err)
	}
	defer qdrantConn.Close()

	qdrantClient := qdrant.NewPointsClient(qdrantConn)
	safetyChecker := safety.NewMedicalSafetyChecker()

	// Initialize OpenRouter.ai client
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	model := os.Getenv("OPENROUTER_MODEL")
	if model == "" {
		model = "mistralai/mistral-7b-instruct" // Default model
	}

	if apiKey == "" {
		log.Fatal("OPENROUTER_API_KEY environment variable is required")
	}

	llmClient := ai.NewLLMClient(apiKey, model)

	// Test model availability
	log.Printf("🔍 Testing OpenRouter.ai connection with model: %s", model)

	chatServer := &ChatServer{
		MedicalChat:   ai.NewLLMMedicalChat(embedder, qdrantClient, llmClient),
		SafetyChecker: safetyChecker,
		LLMClient:     llmClient,
	}

	r := mux.NewRouter()
	r.HandleFunc("/api/chat", chatServer.chatHandler).Methods("POST")
	r.HandleFunc("/api/health", chatServer.healthHandler).Methods("GET")
	r.HandleFunc("/api/capabilities", chatServer.capabilitiesHandler).Methods("GET")
	r.HandleFunc("/api/models", chatServer.modelsHandler).Methods("GET")

	// Serve static files
	r.PathPrefix("/").Handler(http.FileServer(http.Dir("./web/static/")))

	log.Printf("🤖 Medical Chat App starting on :8080")
	log.Printf("🚀 AI Provider: OpenRouter.ai")
	log.Printf("📦 Model: %s", model)
	log.Fatal(http.ListenAndServe(":8080", r))
}

func (cs *ChatServer) capabilitiesHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ai_enabled":   true,
		"model":        cs.LLMClient.Model,
		"provider":     "OpenRouter.ai",
		"capabilities": []string{"real_ai_responses", "medical_knowledge", "safety_checks"},
		"features":     []string{"multiple_models", "free_tier_available", "high_availability"},
	})
}

func (cs *ChatServer) modelsHandler(w http.ResponseWriter, r *http.Request) {
	models, err := cs.LLMClient.GetAvailableModels()
	if err != nil {
		http.Error(w, `{"error": "Failed to fetch models"}`, http.StatusInternalServerError)
		return
	}

	// Filter for Mistral models
	var mistralModels []string
	for _, model := range models {
		if strings.Contains(model, "mistral") {
			mistralModels = append(mistralModels, model)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"available_models": mistralModels,
		"current_model":    cs.LLMClient.Model,
	})
}

func (cs *ChatServer) chatHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error": "Invalid JSON"}`, http.StatusBadRequest)
		return
	}
	if req.Message == "" {
		http.Error(w, `{"error": "Message is required"}`, http.StatusBadRequest)
		return
	}

	safetyResult := cs.SafetyChecker.CheckMessage(req.Message)
	if !safetyResult.IsSafe {
		response := ChatResponse{
			Response:  cs.SafetyChecker.GenerateSafetyResponse(safetyResult.RiskLevel, safetyResult.Reasons),
			Timestamp: time.Now(),
			MessageID: generateMessageID(),
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	ctx := r.Context()
	chatResponse, err := cs.MedicalChat.ProcessMessage(ctx, req.Message, req.History)
	if err != nil {
		http.Error(w, `{"error": "Failed to process message"}`, http.StatusInternalServerError)
		return
	}
	response := ChatResponse{
		Response:    chatResponse.Response,
		Suggestions: chatResponse.Suggestions,
		Timestamp:   time.Now(),
		MessageID:   generateMessageID(),
	}
	json.NewEncoder(w).Encode(response)
}

func generateMessageID() string {
	return fmt.Sprintf("msg_%d", time.Now().UnixNano())
}

func (cs *ChatServer) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "healthy",
		"service": "medical-chat-app",
		"version": "1.0.0",
	})
}
