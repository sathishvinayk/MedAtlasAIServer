package ai

import (
	"MedAtlasAIServer/internal/embeddingClient"
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
