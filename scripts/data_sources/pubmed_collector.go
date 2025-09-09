package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"MedAtlasAIServer/internal/models"
	"MedAtlasAIServer/pkg/data"
)

func main() {
	log.Println("Starting PubMed data collection...")

	// Create output directory
	if err := os.MkdirAll("data/raw", 0755); err != nil {
		log.Fatalf("Failed to create data directory: %v", err)
	}

	// Medical research topics
	medicalTopics := []string{
		"cancer immunotherapy",
		"cardiovascular disease treatment",
		"neurology research",
		"medical artificial intelligence",
		"clinical trials",
		"precision medicine",
		"genomic medicine",
		"infectious diseases",
		"mental health treatment",
		"surgery innovations",
	}

	client := data.NewPubMedClient()
	totalArticles := 0

	for _, topic := range medicalTopics {
		fmt.Printf("\n🔍 Searching PubMed for: %s\n", topic)

		// Search for articles
		articleIDs, err := client.SearchArticles(topic, 50) // Get 50 articles per topic
		if err != nil {
			log.Printf("❌ Search failed for '%s': %v", topic, err)
			continue
		}

		fmt.Printf("   Found %d articles\n", len(articleIDs))

		if len(articleIDs) == 0 {
			continue
		}

		// Fetch article details
		articles, err := client.FetchArticleDetails(articleIDs)
		if err != nil {
			log.Printf("❌ Failed to fetch details for '%s': %v", topic, err)
			continue
		}

		// Process and save articles
		processed := processAndSaveArticles(articles, client, topic)
		totalArticles += processed

		fmt.Printf("   ✅ Processed %d articles for %s\n", processed, topic)

		// Be respectful to PubMed API
		time.Sleep(1 * time.Second)
	}

	fmt.Printf("\n🎉 Collection complete! Total articles processed: %d\n", totalArticles)
}

func processAndSaveArticles(pubmedArticles []models.PubMedArticle, client *data.PubMedClient, topic string) int {
	outputFile := fmt.Sprintf("data/raw/pubmed_%s.jsonl", sanitizeFilename(topic))

	file, err := os.OpenFile(outputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("❌ Failed to open file %s: %v", outputFile, err)
		return 0
	}
	defer file.Close()

	processed := 0

	for _, pubmedArticle := range pubmedArticles {
		// Normalize and validate article
		article := client.NormalizeArticle(pubmedArticle)

		if !data.ValidateArticle(article) {
			log.Printf("⚠️  Skipping invalid article: %s", article.ID)
			continue
		}

		// Clean and enhance the data
		article.Title = data.CleanMedicalText(article.Title)
		article.Abstract = data.CleanMedicalText(article.Abstract)
		article.Abstract = data.NormalizeMedicalTerms(article.Abstract)

		// Convert to JSON and save
		jsonData, err := json.Marshal(article)
		if err != nil {
			log.Printf("❌ Error marshaling article %s: %v", article.ID, err)
			continue
		}

		file.Write(jsonData)
		file.WriteString("\n")
		processed++
	}

	return processed
}

func sanitizeFilename(name string) string {
	// Remove special characters for safe filenames
	var result []rune
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			result = append(result, r)
		} else if r == ' ' {
			result = append(result, '_')
		}
	}
	return string(result)
}
