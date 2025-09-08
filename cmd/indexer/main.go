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

	testVector, err := embedder.GetEmbedding("test query")
	if err != nil {
		log.Fatalf("Embedding service test failed: %v", err)
	}
	fmt.Printf("Embedding service connected. Vector size: %d\n", len(testVector))

	_, err = collectionsClient.Create(context.Background(), &qdrant.CreateCollection{
		CollectionName: "medical_abstracts",
		VectorsConfig: &qdrant.VectorsConfig{Config: &qdrant.VectorsConfig_Params{
			Params: &qdrant.VectorParams{
				Size:     uint64(len(testVector)), // Use actual dimension later
				Distance: qdrant.Distance_Cosine,
			},
		}},
	})

	if err != nil {
		log.Printf("Collection creation error (might exist): %v", err)
	}

	file, err := os.Open("data/sample_data.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	var points []*qdrant.PointStruct
	batchSize := 50

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

		if len(points) >= batchSize {
			if err := uploadBatch(pointsClient, points); err != nil {
				log.Printf("Batch upload failed: %v", err)
			}
			points = points[:0]
			fmt.Print(".") // Show some progress
		}
	}

	if len(points) > 0 {
		if err := uploadBatch(pointsClient, points); err != nil {
			log.Printf("Final batch upload failed: %v", err)
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

func parseID(id string) uint64 {
	var idNum uint64
	fmt.Scanf(id, "%d", &idNum)
	return idNum
}
