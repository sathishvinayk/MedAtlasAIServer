package main

import (
	"MedAtlasAIServer/internal/ai"
	"MedAtlasAIServer/internal/embeddingClient"
	"MedAtlasAIServer/internal/safety"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ChatServer struct {
	MedicalChat   *ai.MedicalChat
	SafetyChecker *safety.MedicalSafetyChecker
}

type ChatRequest struct {
	Message string           `json:"message"`
	History []ai.ChatMessage `json:"history,omitempty"`
}

type ChatResponse struct {
	Response    string    `json:"response"`
	Timestamp   time.Time `json:"timestamp"`
	MessageID   string    `json:"message_id"`
	Suggestions []string  `json:"suggestions,omitempty"`
}

func main() {
	embedder := embeddingClient.NewClient("http://localhost:8000")
	qdrantConn, err := grpc.Dial("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Could not connect to Qdrant: %v", err)
	}
	defer qdrantConn.Close()

	qdrantClient := qdrant.NewPointsClient(qdrantConn)
	safetyChecker := safety.NewMedicalSafetyChecker()

	chatServer := &ChatServer{
		MedicalChat:   ai.NewMedicalChat(embedder, qdrantClient),
		SafetyChecker: safetyChecker,
	}

	r := mux.NewRouter()
	r.HandleFunc("/api/chat", chatServer.chatHandler).Methods("POST")
	r.HandleFunc("/api/health", chatServer.healthHandler).Methods("GET")

	log.Println("🤖 Medical Chat App starting on :8080")
	log.Fatal(http.ListenAndServe(":8080", r))
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
