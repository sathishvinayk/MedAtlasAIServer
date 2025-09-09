package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"MedAtlasAIServer/internal/embeddingClient"
	"MedAtlasAIServer/internal/models"
	"MedAtlasAIServer/pkg/data"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Global counter for processed documents
var totalProcessed int64

func main() {
	log.Println("🚀 Starting Medical Document Indexer...")
	log.Println("📊 Initializing services...")

	// Initialize clients
	embedder := embeddingClient.NewClient("http://localhost:8000")
	qdrantConn, err := grpc.Dial("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("❌ Could not connect to Qdrant: %v", err)
	}
	defer qdrantConn.Close()

	collectionsClient := qdrant.NewCollectionsClient(qdrantConn)
	pointsClient := qdrant.NewPointsClient(qdrantConn)

	// Test embedding service and get dimension
	log.Println("🔍 Testing embedding service...")
	testVector, err := embedder.GetEmbedding("medical research treatment cancer immunotherapy")
	if err != nil {
		log.Fatalf("❌ Embedding service test failed: %v", err)
	}
	vectorSize := len(testVector)
	fmt.Printf("✅ Embedding dimension: %d\n", vectorSize)

	// Setup collection
	ctx := context.Background()
	setupCollection(ctx, collectionsClient, vectorSize)

	// Find all PubMed data files
	dataFiles, err := filepath.Glob("data/raw/pubmed_*.jsonl")
	if err != nil {
		log.Fatalf("❌ Error finding data files: %v", err)
	}

	// Add sample data if exists
	if fileExists("data/medical_sample_large.jsonl") {
		dataFiles = append(dataFiles, "data/medical_sample_large.jsonl")
	}

	if len(dataFiles) == 0 {
		log.Fatalf("❌ No data files found in data/raw/ directory")
	}

	log.Printf("📁 Found %d data files to process", len(dataFiles))

	// Track seen IDs to avoid duplicates
	seenIDs := make(map[string]bool)
	duplicateCount := 0

	// Process each file
	for _, dataFile := range dataFiles {
		log.Printf("📄 Processing file: %s", dataFile)
		fileProcessed, fileDuplicates := processFile(ctx, dataFile, embedder, pointsClient, vectorSize, seenIDs)
		atomic.AddInt64(&totalProcessed, int64(fileProcessed))
		duplicateCount += fileDuplicates
		log.Printf("✅ Processed %d documents from %s (%d duplicates skipped)",
			fileProcessed, dataFile, fileDuplicates)
	}

	log.Printf("🎉 Indexing complete! Total documents processed: %d", totalProcessed)
	log.Printf("🔁 Duplicates skipped: %d", duplicateCount)

	// Verify the final count
	countResp, err := pointsClient.Count(ctx, &qdrant.CountPoints{
		CollectionName: "medical_abstracts",
		// Exact:          &qdrant.Exact{Exact: true},
	})
	if err != nil {
		log.Printf("⚠️  Error counting points: %v", err)
	} else {
		log.Printf("📈 Total points in collection: %d", countResp.Result.Count)
	}

	// Check for discrepancy
	if countResp.Result.Count != uint64(totalProcessed) {
		log.Printf("⚠️  WARNING: Collection count (%d) doesn't match processed count (%d)",
			countResp.Result.Count, totalProcessed)
		log.Printf("💡 Some documents may have failed to index or were duplicates")
	}
}

func setupCollection(ctx context.Context, client qdrant.CollectionsClient, vectorSize int) {
	log.Println("🔄 Setting up Qdrant collection...")

	// First, check if collection exists
	listResp, err := client.List(ctx, &qdrant.ListCollectionsRequest{})
	if err != nil {
		log.Printf("⚠️  Error listing collections: %v", err)
		return
	}

	collectionExists := false
	for _, coll := range listResp.Collections {
		if coll.Name == "medical_abstracts" {
			collectionExists = true
			break
		}
	}

	if collectionExists {
		log.Println("📊 Collection already exists. Checking if we need to recreate...")
		// For now, let's keep the existing collection to avoid data loss
		log.Println("💡 Using existing collection - new documents will be added/updated")
		return
	}

	// Create new collection if it doesn't exist
	log.Printf("🆕 Creating new collection with vector size: %d", vectorSize)
	_, err = client.Create(ctx, &qdrant.CreateCollection{
		CollectionName: "medical_abstracts",
		VectorsConfig: &qdrant.VectorsConfig{Config: &qdrant.VectorsConfig_Params{
			Params: &qdrant.VectorParams{
				Size:     uint64(vectorSize),
				Distance: qdrant.Distance_Cosine,
			},
		}},
	})
	if err != nil {
		log.Fatalf("❌ Failed to create collection: %v", err)
	}
	log.Println("✅ Collection created successfully")
}

func processFile(ctx context.Context, filename string, embedder *embeddingClient.Client,
	pointsClient qdrant.PointsClient, vectorSize int, seenIDs map[string]bool) (int, int) {

	file, err := os.Open(filename)
	if err != nil {
		log.Printf("❌ Error opening file %s: %v", filename, err)
		return 0, 0
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	batchSize := 10 // Increased batch size for efficiency
	processed := 0
	duplicateCount := 0
	batchCount := 0
	var points []*qdrant.PointStruct

	for decoder.More() {
		var article models.MedicalArticle
		if err := decoder.Decode(&article); err != nil {
			log.Printf("❌ Error decoding JSON in %s: %v", filename, err)
			continue
		}

		// Skip duplicates across files
		if seenIDs[article.ID] {
			duplicateCount++
			continue
		}
		seenIDs[article.ID] = true

		// Clean and enhance the data first
		article.Title = data.CleanMedicalText(article.Title)
		article.Abstract = data.CleanMedicalText(article.Abstract)
		article.Abstract = data.NormalizeMedicalTerms(article.Abstract)

		// Validate after cleaning
		if valid, reason := data.ValidateArticleWithReason(article); !valid {
			log.Printf("⚠️  Skipping article %s: %s", article.ID, reason)
			continue
		}

		// Create embedding from title and abstract
		textToEmbed := article.Title + ". " + article.Abstract
		vector, err := embedder.GetEmbedding(textToEmbed)
		if err != nil {
			log.Printf("❌ Error creating embedding for %s: %v", article.ID, err)
			continue
		}

		// Verify vector dimension matches our collection
		if len(vector) != vectorSize {
			log.Printf("⚠️  Vector dimension mismatch for %s. Expected %d, got %d",
				article.ID, vectorSize, len(vector))
			// Skip this article if dimension doesn't match
			continue
		}

		// Prepare payload for Qdrant
		payload := map[string]*qdrant.Value{
			"title":          {Kind: &qdrant.Value_StringValue{StringValue: article.Title}},
			"abstract":       {Kind: &qdrant.Value_StringValue{StringValue: article.Abstract}},
			"authors":        {Kind: &qdrant.Value_StringValue{StringValue: data.FormatAuthors(article.Authors)}},
			"published_date": {Kind: &qdrant.Value_StringValue{StringValue: article.PublishedDate.Format("2006-01-02")}},
			"doi":            {Kind: &qdrant.Value_StringValue{StringValue: article.DOI}},
			"journal":        {Kind: &qdrant.Value_StringValue{StringValue: article.Journal}},
			"source":         {Kind: &qdrant.Value_StringValue{StringValue: article.Source}},
			"id":             {Kind: &qdrant.Value_StringValue{StringValue: article.ID}},
		}

		// Add MeSH headings if available
		if len(article.MeshHeadings) > 0 {
			payload["mesh_headings"] = &qdrant.Value{
				Kind: &qdrant.Value_ListValue{
					ListValue: &qdrant.ListValue{
						Values: convertToValueList(article.MeshHeadings),
					},
				},
			}
		}

		point := &qdrant.PointStruct{
			Id:      &qdrant.PointId{PointIdOptions: &qdrant.PointId_Num{Num: parseID(article.ID)}},
			Vectors: &qdrant.Vectors{VectorsOptions: &qdrant.Vectors_Vector{Vector: &qdrant.Vector{Data: vector}}},
			Payload: payload,
		}

		points = append(points, point)
		processed++

		// Upload batch when full
		if len(points) >= batchSize {
			batchCount++
			success := uploadBatchWithRetry(ctx, pointsClient, points, batchCount, 3) // 3 retries
			if !success {
				log.Printf("❌ Batch %d failed after retries, skipping %d documents", batchCount, len(points))
				// Reset points but don't count them as processed
				processed -= len(points)
			}
			points = make([]*qdrant.PointStruct, 0, batchSize)
		}
	}

	// Upload final batch
	if len(points) > 0 {
		batchCount++
		success := uploadBatchWithRetry(ctx, pointsClient, points, batchCount, 3)
		if !success {
			log.Printf("❌ Final batch failed after retries, skipping %d documents", len(points))
			processed -= len(points)
		}
	}

	return processed, duplicateCount
}

func uploadBatchWithRetry(ctx context.Context, client qdrant.PointsClient,
	points []*qdrant.PointStruct, batchNumber int, maxRetries int) bool {

	if len(points) == 0 {
		return true
	}

	for attempt := 1; attempt <= maxRetries; attempt++ {
		log.Printf("📤 Uploading batch %d (attempt %d/%d) with %d points...",
			batchNumber, attempt, maxRetries, len(points))

		start := time.Now()
		_, err := client.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: "medical_abstracts",
			Points:         points,
			// Wait:           &qdrant.Wait{Enabled: true},
		})

		if err != nil {
			log.Printf("❌ Batch %d attempt %d failed: %v", batchNumber, attempt, err)
			if attempt < maxRetries {
				time.Sleep(time.Duration(attempt) * time.Second) // Exponential backoff
				continue
			}
			return false
		}

		log.Printf("✅ Successfully uploaded batch %d in %v", batchNumber, time.Since(start))
		return true
	}

	return false
}

func parseID(idStr string) uint64 {
	// First try to parse as number
	var idNum uint64
	_, err := fmt.Sscanf(idStr, "%d", &idNum)
	if err == nil {
		return idNum
	}

	// If not numeric, create a hash-based ID
	hash := uint64(0)
	for _, char := range idStr {
		hash = hash*31 + uint64(char)
	}
	return hash
}

func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func convertToValueList(strings []string) []*qdrant.Value {
	values := make([]*qdrant.Value, len(strings))
	for i, s := range strings {
		values[i] = &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: s}}
	}
	return values
}
