package chat

import (
	"MedAtlasAIServer/internal/ai"
	"MedAtlasAIServer/internal/safety"
)

type ChatServer struct {
	MedicalChat   *ai.MedicalChat
	SafetyChecker *safety.MedicalSafetyChecker
}

type ChatRequest struct {
	Message string           `json:"message"`
	History []ai.ChatMessage `json:"history,omitempty"`
}
