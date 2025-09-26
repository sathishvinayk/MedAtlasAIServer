package embeddingClient

import (
	"MedAtlasAIServer/internal/models"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	pb "MedAtlasAIServer/gen"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
	GRPCConn   *grpc.ClientConn
	GRPCClient pb.AudioProcessorClient
}

func NewClient(baseURL string) *Client {
	client := &Client{
		BaseURL:    baseURL,
		HTTPClient: &http.Client{},
	}

	grpcHost := "localhost:50051"
	conn, err := grpc.Dial(grpcHost, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err == nil {
		client.GRPCConn = conn
		client.GRPCClient = pb.NewAudioProcessorClient(conn)
	}
	return client
}

func (c *Client) Close() {
	if c.GRPCConn != nil {
		c.GRPCConn.Close()
	}
}

func (c *Client) CreateStream(ctx context.Context, sessionID string, sampleRate int32) (pb.AudioProcessor_ProcessAudioStreamClient, error) {
	if c.GRPCClient == nil {
		return nil, fmt.Errorf("gRPC client not available")
	}
	stream, err := c.GRPCClient.ProcessAudioStream(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC stream: %w", err)
	}

	return stream, nil
}

func (c *Client) SendChunk(ctx context.Context, stream pb.AudioProcessor_ProcessAudioStreamClient, sessionID string, audioData []byte, chunkIndex int, isFinal bool, sampleRate int32) (*models.StreamingResult, error) {
	audioChunk := &pb.AudioChunk{
		AudioData:  audioData,
		SessionId:  sessionID,
		SampleRate: sampleRate,
		ChunkIndex: int32(chunkIndex),
		IsFinal:    isFinal,
	}
	if err := stream.Send(audioChunk); err != nil {
		return nil, fmt.Errorf("failed to send chunk: %w", err)
	}
	ctx, cancel := context.WithTimeout(ctx, 50*time.Second)
	defer cancel()

	result, err := stream.Recv()
	if err != nil {
		if err == io.EOF {
			return nil, fmt.Errorf("stream closed by server")
		}
		return nil, fmt.Errorf("failed to receive result: %w", err)
	}

	// Convert to StreamingResult
	streamingResult := &models.StreamingResult{
		SessionID: result.SessionId,
	}

	switch res := result.Result.(type) {
	case *pb.ProcessingResult_Transcript:
		streamingResult.Type = "transcript"
		streamingResult.Data = map[string]interface{}{
			"text":       res.Transcript.Text,
			"is_partial": res.Transcript.IsPartial,
			"start_ms":   res.Transcript.StartTimeMs,
			"end_ms":     res.Transcript.EndTimeMs,
		}
		streamingResult.IsPartial = res.Transcript.IsPartial

	case *pb.ProcessingResult_Speaker:
		streamingResult.Type = "speaker"
		var segments []map[string]interface{}
		for _, seg := range result.GetSpeaker().Segments {
			segments = append(segments, map[string]interface{}{
				"speaker_id": seg.SpeakerId,
				"start_time": seg.StartTime,
				"end_time":   seg.EndTime,
				"confidence": seg.Confidence,
			})
		}
		streamingResult.Data = map[string]interface{}{"segments": segments}
		streamingResult.IsPartial = false

	case *pb.ProcessingResult_Entities:
		streamingResult.Type = "entities"
		var entities []map[string]interface{}
		for _, ent := range res.Entities.Entities {
			entities = append(entities, map[string]interface{}{
				"entity_type": ent.EntityType,
				"text":        ent.Text,
				"start":       ent.Start,
				"end":         ent.End,
				"confidence":  ent.Confidence,
			})
		}
		streamingResult.Data = map[string]interface{}{"entities": entities}
		streamingResult.IsPartial = false

	case *pb.ProcessingResult_SoapNote:
		streamingResult.Type = "soap_note"
		streamingResult.Data = map[string]interface{}{
			"content":     res.SoapNote.Content,
			"is_complete": res.SoapNote.IsComplete,
		}
		streamingResult.IsPartial = !res.SoapNote.IsComplete
	}

	return streamingResult, nil

}

// Grpc Streaming
func (c *Client) ProcessAudioStream(ctx context.Context, sessionID string, audioChunks <-chan []byte, sampleRate int32) (<-chan *models.StreamingResult, error) {
	if c.GRPCClient == nil {
		return nil, fmt.Errorf("gRPC client not available")
	}
	stream, err := c.GRPCClient.ProcessAudioStream(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC stream: %w", err)
	}

	resultChan := make(chan *models.StreamingResult, 100)

	go func() {
		defer close(resultChan)
		defer stream.CloseSend()

		chunkIndex := 0
		for {
			select {
			case chunk, ok := <-audioChunks:
				if !ok {
					// Send final chunk
					finalChunk := &pb.AudioChunk{
						AudioData:  []byte{}, // Empty final chunk
						SessionId:  sessionID,
						SampleRate: sampleRate,
						ChunkIndex: int32(chunkIndex),
						IsFinal:    true,
					}
					if err := stream.Send(finalChunk); err != nil {
						log.Printf("Failed to send final chunk: %v", err)
					}
					log.Printf("Sent final chunk for session: %s", sessionID)
					return
				}

				if len(chunk) == 0 {
					continue // Skip empty chunks
				}

				audioChunk := &pb.AudioChunk{
					AudioData:  chunk,
					SessionId:  sessionID,
					SampleRate: sampleRate,
					ChunkIndex: int32(chunkIndex),
					IsFinal:    false,
				}

				log.Printf("Sending chunk %d, size: %d bytes", chunkIndex, len(chunk))
				if err := stream.Send(audioChunk); err != nil {
					log.Printf("Failed to send chunk %d: %v", chunkIndex, err)
					return
				}
				chunkIndex++

			case <-ctx.Done():
				log.Printf("Context cancelled, stopping stream")
				return
			}
		}
	}()

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				result, err := stream.Recv()
				if err != nil {
					if err == io.EOF {
						log.Printf("Stream closed by server")
					} else {
						log.Printf("Stream receive error: %v", err)
					}
					return
				}

				streamingResult := &models.StreamingResult{
					SessionID: result.SessionId,
				}

				switch res := result.Result.(type) {
				case *pb.ProcessingResult_Transcript:
					streamingResult.Type = "transcript"
					streamingResult.Data = map[string]interface{}{
						"text":       res.Transcript.Text,
						"is_partial": res.Transcript.IsPartial,
						"start_ms":   res.Transcript.StartTimeMs,
						"end_ms":     res.Transcript.EndTimeMs,
					}
					streamingResult.IsPartial = res.Transcript.IsPartial
					log.Printf("Received transcript: %s", res.Transcript.Text)

				case *pb.ProcessingResult_Speaker:
					streamingResult.Type = "speaker"
					var segments []map[string]interface{}
					for _, seg := range result.GetSpeaker().Segments {
						segments = append(segments, map[string]interface{}{
							"speaker_id": seg.SpeakerId,
							"start_time": seg.StartTime,
							"end_time":   seg.EndTime,
							"confidence": seg.Confidence,
						})
					}
					streamingResult.Data = map[string]interface{}{"segments": segments}
					streamingResult.IsPartial = false

				case *pb.ProcessingResult_Entities:
					streamingResult.Type = "entities"
					var entities []map[string]interface{}
					for _, ent := range res.Entities.Entities {
						entities = append(entities, map[string]interface{}{
							"entity_type": ent.EntityType,
							"text":        ent.Text,
							"start":       ent.Start,
							"end":         ent.End,
							"confidence":  ent.Confidence,
						})
					}
					streamingResult.Data = map[string]interface{}{"entities": entities}
					streamingResult.IsPartial = false

				case *pb.ProcessingResult_SoapNote:
					streamingResult.Type = "soap_note"
					streamingResult.Data = map[string]interface{}{
						"content":     res.SoapNote.Content,
						"is_complete": res.SoapNote.IsComplete,
					}
					streamingResult.IsPartial = !res.SoapNote.IsComplete
				}

				resultChan <- streamingResult
			}
		}
	}()
	return resultChan, nil
}

func (c *Client) GetEmbedding(text string) ([]float32, error) {
	reqBody := models.EmbedRequest{Text: text}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", c.BaseURL+"/embed", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding service returned error: %s - %s", resp.Status, string(body))
	}
	var embedResp models.EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	return embedResp.Vector, nil
}

func (c *Client) ProcessAudio(audioRequest *models.ProcessAudioRequest) (*models.ProcessAudioResponse, error) {
	jsonData, err := json.Marshal(audioRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal audio request: %w", err)
	}
	req, err := http.NewRequest("POST", c.BaseURL+"/process-audio", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding service returned error: %s - %s", resp.Status, string(body))
	}

	var aiResp models.ProcessAudioResponse
	if err := json.NewDecoder(resp.Body).Decode(&aiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	return &aiResp, nil
}
