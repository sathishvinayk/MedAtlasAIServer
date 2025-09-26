package main

import (
	"MedAtlasAIServer/internal/embeddingClient"
	"MedAtlasAIServer/internal/models"
	"encoding/base64"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type SearchRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type SearchResponse struct {
	ID            string  `json:"id"`
	Title         string  `json:"title"`
	Abstract      string  `json:"abstract"`
	Authors       string  `json:"authors"`
	PublishedDate string  `json:"published_date"`
	DOI           string  `json:"doi"`
	Score         float32 `json:"score"`
}

type Server struct {
	QdrantClient  qdrant.PointsClient
	Embedder      *embeddingClient.Client
	Sessions      map[string]*models.StreamingSession
	SessionsMutex sync.RWMutex
}

func main() {
	embeddedHost := os.Getenv("EMBEDDING_SERVICE_HOST")
	if embeddedHost == "" { //Keep localhost for now
		embeddedHost = "http://localhost:8000"
	}
	embedder := embeddingClient.NewClient(embeddedHost)
	qdrantHost := os.Getenv("QDRANT_HOST")
	if qdrantHost == "" { //Keep localhost for now
		qdrantHost = "localhost:6334"
	}

	conn, err := grpc.Dial(qdrantHost, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Could not connect to Qdrant: %v", err)
	}
	defer conn.Close()
	qdrantClient := qdrant.NewPointsClient(conn)

	server := &Server{
		QdrantClient: qdrantClient,
		Embedder:     embedder,
		Sessions:     make(map[string]*models.StreamingSession),
	}

	// Routing
	r := mux.NewRouter()
	r.HandleFunc("/search", server.searchHandler).Methods("POST")
	r.HandleFunc("/health", server.healthHandler).Methods("GET")
	r.HandleFunc("/ready", server.readyHandler).Methods("GET")

	r.HandleFunc("/process-audio", server.processAudioHandler).Methods("POST", "OPTIONS")

	r.HandleFunc("/stream/start", server.startStreamingHandler).Methods("POST")
	r.HandleFunc("/stream/chunk", server.processStreamingChunkHandler).Methods("POST")
	r.HandleFunc("/ws/stream", server.websocketStreamHandler).Methods("GET")

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Cors
	corsMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
	log.Printf("Server starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, corsMiddleware(r)))
}

func generateSessionID() string {
	return strconv.FormatInt(time.Now().UnixNano(), 36)
}

func formatPointID(pointID *qdrant.PointId) string {
	if pointID == nil {
		return ""
	}

	switch id := pointID.PointIdOptions.(type) {
	case *qdrant.PointId_Num:
		return strconv.FormatUint(id.Num, 10)
	case *qdrant.PointId_Uuid:
		return id.Uuid
	default:
		return ""
	}
}

func (s *Server) startStreamingHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	sessionID := generateSessionID()

	s.SessionsMutex.Lock()

	s.Sessions[sessionID] = &models.StreamingSession{
		SessionID: sessionID,
		Status:    "active",
		CreatedAt: time.Now().Unix(),
	}
	s.SessionsMutex.Unlock()
	log.Printf("Started streaming session: %s", sessionID)

	json.NewEncoder(w).Encode(map[string]string{
		"session_id": sessionID,
		"status":     "active",
		"message":    "Streaming session started",
	})
}

func (s *Server) processStreamingChunkHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("🚀 Received streaming chunk request")

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	var req models.StreamingAudioRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("❌ JSON decode error: %v", err)

		http.Error(w, `{"error": "Invalid JSON"}`, http.StatusBadRequest)
		return
	}

	s.SessionsMutex.RLock()
	session, exists := s.Sessions[req.SessionID]
	s.SessionsMutex.RUnlock()

	if !exists || session.Status != "active" {
		http.Error(w, `{"error": "Invalid or inactive session"}`, http.StatusBadRequest)
		return
	}
	audioData, err := base64.StdEncoding.DecodeString(req.AudioChunk)
	if err != nil {
		http.Error(w, `{"error": "Invalid audio data"}`, http.StatusBadRequest)
		return
	}
	// ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	ctx := r.Context() // Use request context instead of short timeout
	// defer cancel()

	audioChunks := make(chan []byte, 10)

	resultChan, err := s.Embedder.ProcessAudioStream(ctx, req.SessionID, audioChunks, 16000)
	if err != nil {
		log.Printf("gRPC streaming error: %v", err)
		http.Error(w, `{"error": "Streaming unavailable"}`, http.StatusServiceUnavailable)
		return
	}

	go func() {
		audioChunks <- audioData
		if req.IsFinal {
			close(audioChunks)
		}
	}()

	var results []models.StreamingResult
	for result := range resultChan {
		results = append(results, *result)
		switch result.Type {
		case "transcript":
			if text, ok := result.Data["text"].(string); ok {
				session.CurrentTranscript += " " + text
			}
		case "speaker":
		case "entities":
		case "soap_note":
			if req.IsFinal {
				session.Status = "completed"
			}
		}
	}
	if req.IsFinal {
		s.SessionsMutex.Lock()
		session.Status = "completed"
		s.SessionsMutex.Unlock()
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"session_id":  req.SessionID,
		"status":      "processed",
		"chunk_index": req.ChunkIndex,
		"is_final":    req.IsFinal,
		"results":     results,
	})
}

func (s *Server) websocketStreamHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
	json.NewEncoder(w).Encode(map[string]string{
		"error": "WebSocket streaming not implemented yet",
	})
}

func safeGetString(payload map[string]*qdrant.Value, key string) string {
	if value, exists := payload[key]; exists && value != nil {
		return value.GetStringValue()
	}
	return ""
}

func (s *Server) searchHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error": "Invalid JSON"}`, http.StatusBadRequest)
		return
	}
	if req.Query == "" {
		http.Error(w, `{"error": "Query parameter is required"}`, http.StatusBadRequest)
		return
	}
	if req.Limit == 0 {
		req.Limit = 10
	}

	// Convert User query to a vector
	queryVector, err := s.Embedder.GetEmbedding(req.Query)
	if err != nil {
		log.Printf("Embedding error: %v", err)
		http.Error(w, `{"error": "Error processing query"}`, http.StatusInternalServerError)
		return
	}
	searchResult, err := s.QdrantClient.Search(r.Context(), &qdrant.SearchPoints{
		CollectionName: "medical_abstracts",
		Vector:         queryVector,
		Limit:          uint64(req.Limit),
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Include{
				Include: &qdrant.PayloadIncludeSelector{Fields: []string{"title", "abstract", "authors", "published_date", "doi"}},
			},
		},
	})

	if err != nil {
		log.Printf("Qdrant search error: %v", err)
		http.Error(w, `{"error": "Search failed"}`, http.StatusInternalServerError)
		return
	}

	results := make([]SearchResponse, len(searchResult.Result))
	for i, point := range searchResult.Result {
		payload := point.Payload
		results[i] = SearchResponse{
			ID:            formatPointID(point.Id),
			Title:         safeGetString(payload, "title"),
			Abstract:      safeGetString(payload, "abstract"),
			Authors:       safeGetString(payload, "authors"),
			PublishedDate: safeGetString(payload, "published_date"),
			DOI:           safeGetString(payload, "doi"),
			Score:         point.Score,
		}
	}

	if err := json.NewEncoder(w).Encode(results); err != nil {
		log.Printf("JSON encoding error: %v", err)
		http.Error(w, `{"error": "Error formatting response"}`, http.StatusInternalServerError)
	}
}

func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "service": "medical-Atlas-api"})
}

func (s *Server) readyHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	ctx := r.Context()
	_, err := s.QdrantClient.Search(ctx, &qdrant.SearchPoints{
		CollectionName: "medical_abstracts",
		Vector:         make([]float32, 384), //Dummy vector
		Limit:          1,
	})

	_, embedErr := s.Embedder.GetEmbedding("test")
	status := "ready"
	if err != nil || embedErr != nil {
		status = "not ready"
		w.WriteHeader(http.StatusServiceUnavailable)
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":           status,
		"qdrant_connected": err == nil,
		"embedder_ready":   embedErr == nil,
	})
}

func (s *Server) processAudioHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	var audioReq models.ProcessAudioRequest
	if err := json.NewDecoder(r.Body).Decode(&audioReq); err != nil {
		http.Error(w, `{"error": "Invlalid json"}`, http.StatusBadRequest)
		return
	}

	aiResult, err := s.Embedder.ProcessAudio(&audioReq)
	if err != nil {
		log.Printf("AI service error: %v", err)
		http.Error(w, `{"error": "Error processing audio"}`, http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(aiResult)
}
