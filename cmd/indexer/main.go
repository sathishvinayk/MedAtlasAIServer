package main

import (
	"MedAtlasAIServer/internal/embeddingClient"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/qdrant/go-client/qdrant"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Document struct {
	ID            string `json:"id"`
	Title         string `json:"title"`
	Abstract      string `json:"abstract"`
	Authors       string `json:"authors"`
	PublishedDate string `json:"published_date"`
	DOI           string `json:"doi"`
}

func main() {
	embedder := embeddingClient.NewClient("http://localhost:8000")
	qdrantConn, err := grpc.Dial("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Could not connect to Qdrant: %v", err)
	}
	defer qdrantConn.Close()

	collectionsClient := qdrant.NewCollectionsClient(qdrantConn)
	pointsClient := qdrant.NewPointsClient(qdrantConn)

	testVector, err := embedder.GetEmbedding("cardiovascular disease treatment")
	if err != nil {
		log.Fatalf("Embedding service test failed: %v", err)
	}
	vectorSize := len(testVector)

	fmt.Printf("Embedding service connected. Vector size: %d\n", vectorSize)

	_, err = collectionsClient.Create(context.Background(), &qdrant.CreateCollection{
		CollectionName: "medical_abstracts",
		VectorsConfig: &qdrant.VectorsConfig{Config: &qdrant.VectorsConfig_Params{
			Params: &qdrant.VectorParams{
				Size:     uint64(vectorSize), // Use actual dimension later
				Distance: qdrant.Distance_Cosine,
			},
		}},
	})

	if err != nil {
		log.Printf("Collection creation error (might exist): %v", err)
	}

	file, err := os.Open("data/pubmed_real_articles.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	var points []*qdrant.PointStruct
	batchSize := 2
	processed := 0
	batchCount := 0

	for decoder.More() {
		var doc Document
		if err := decoder.Decode(&doc); err != nil {
			log.Printf("Error decoding JSON: %v", err)
			continue
		}

		textToEmbed := doc.Title + " " + doc.Abstract
		vector, err := embedder.GetEmbedding(textToEmbed)

		if err != nil {
			log.Printf("Error creating embedding for doc %s: %v", doc.ID, err)
			continue
		}
		// Verify vector dimension matches our collection
		if len(vector) != vectorSize {
			log.Printf("Warning: Vector dimension mismatch for doc %s. Expected %d, got %d",
				doc.ID, vectorSize, len(vector))
			// Continue anyway, but this might cause issues
		}

		// Point building
		point := &qdrant.PointStruct{
			Id: &qdrant.PointId{
				PointIdOptions: &qdrant.PointId_Num{Num: parseID(doc.ID)},
			},
			Vectors: &qdrant.Vectors{
				VectorsOptions: &qdrant.Vectors_Vector{Vector: &qdrant.Vector{Data: vector}},
			},
			Payload: map[string]*qdrant.Value{
				"title":          {Kind: &qdrant.Value_StringValue{StringValue: doc.Title}},
				"abstract":       {Kind: &qdrant.Value_StringValue{StringValue: doc.Abstract}},
				"authors":        {Kind: &qdrant.Value_StringValue{StringValue: doc.Authors}},
				"published_date": {Kind: &qdrant.Value_StringValue{StringValue: doc.PublishedDate}},
				"doi":            {Kind: &qdrant.Value_StringValue{StringValue: doc.DOI}},
			},
		}

		points = append(points, point)
		processed++

		if len(points) >= batchSize {
			batchCount++
			fmt.Printf("Uploading batch %d with %d points...\n", batchCount, len(points))

			if err := uploadBatch(pointsClient, points); err != nil {
				log.Printf("Batch upload failed: %v", err)
			} else {
				fmt.Printf("Processed %d documents...\n", processed)
			}
			points = points[:0] // Clear the slice
		}
	}

	if len(points) > 0 {
		batchCount++
		fmt.Printf("Uploading final batch %d with %d points...\n", batchCount, len(points))

		if err := uploadBatch(pointsClient, points); err != nil {
			log.Printf("Final batch upload failed: %v", err)
		} else {
			fmt.Printf("Successfully uploaded final batch %d\n", batchCount)
		}
	}
	log.Println("\nIndexing complete!")
}

func uploadBatch(client qdrant.PointsClient, points []*qdrant.PointStruct) error {
	_, err := client.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: "medical_abstracts",
		Points:         points,
		// Wait:           &qdrant.Wait{Wait: true}, // Wait for confirmation
	})
	return err
}

func parseID(idStr string) uint64 {
	// Simple implementation - in production, use a proper ID scheme
	var idNum uint64
	_, err := fmt.Sscanf(idStr, "%d", &idNum)
	if err != nil {
		// If ID is not numeric, create a hash-based ID
		hash := fnvHash(idStr)
		return hash
	}
	return idNum
}

func fnvHash(s string) uint64 {
	// Simple FNV hash for string IDs
	const prime uint64 = 1099511628211
	hash := uint64(14695981039346656037)
	for i := 0; i < len(s); i++ {
		hash ^= uint64(s[i])
		hash *= prime
	}
	return hash
}
