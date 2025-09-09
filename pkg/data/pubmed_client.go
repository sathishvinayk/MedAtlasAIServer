package data

import (
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"MedAtlasAIServer/internal/models"
)

type PubMedClient struct {
	BaseURL    string
	HTTPClient *http.Client
	BatchSize  int
	Delay      time.Duration
}

func NewPubMedClient() *PubMedClient {
	return &PubMedClient{
		BaseURL:    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
		HTTPClient: &http.Client{Timeout: 30 * time.Second},
		BatchSize:  100,                    // PubMed API limit per request
		Delay:      200 * time.Millisecond, // Respect API rate limits
	}
}

func (c *PubMedClient) SearchArticles(query string, maxResults int) ([]string, error) {
	// ESearch: Get article IDs
	esearchURL := fmt.Sprintf("%s/esearch.fcgi?db=pubmed&term=%s&retmax=%d&retmode=xml",
		c.BaseURL, url.QueryEscape(query), maxResults)

	resp, err := c.HTTPClient.Get(esearchURL)
	if err != nil {
		return nil, fmt.Errorf("ESearch request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read ESearch response: %w", err)
	}

	var esearchResult struct {
		IdList struct {
			IDs []string `xml:"Id"`
		} `xml:"IdList"`
	}

	if err := xml.Unmarshal(body, &esearchResult); err != nil {
		return nil, fmt.Errorf("failed to parse ESearch XML: %w", err)
	}

	return esearchResult.IdList.IDs, nil
}

func (c *PubMedClient) FetchArticleDetails(articleIDs []string) ([]models.PubMedArticle, error) {
	var allArticles []models.PubMedArticle

	// Process in batches to respect API limits
	for i := 0; i < len(articleIDs); i += c.BatchSize {
		end := i + c.BatchSize
		if end > len(articleIDs) {
			end = len(articleIDs)
		}

		batchIDs := articleIDs[i:end]
		articles, err := c.fetchBatch(batchIDs)
		if err != nil {
			log.Printf("Failed to fetch batch %d-%d: %v", i, end, err)
			continue
		}

		allArticles = append(allArticles, articles...)

		// Respect API rate limits
		if i+c.BatchSize < len(articleIDs) {
			time.Sleep(c.Delay)
		}
	}

	return allArticles, nil
}

func (c *PubMedClient) fetchBatch(articleIDs []string) ([]models.PubMedArticle, error) {
	if len(articleIDs) == 0 {
		return nil, nil
	}

	idParam := strings.Join(articleIDs, ",")
	efetchURL := fmt.Sprintf("%s/efetch.fcgi?db=pubmed&id=%s&retmode=xml",
		c.BaseURL, idParam)

	resp, err := c.HTTPClient.Get(efetchURL)
	if err != nil {
		return nil, fmt.Errorf("EFetch request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read EFetch response: %w", err)
	}

	// Parse XML response
	var result models.PubMedResult
	if err := xml.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse EFetch XML: %w", err)
	}

	return result.Articles, nil
}

func (c *PubMedClient) NormalizeArticle(pubmedArticle models.PubMedArticle) models.MedicalArticle {
	article := pubmedArticle.MedlineCitation.Article

	// Extract abstract text
	var abstractBuilder strings.Builder
	for _, abstractText := range article.Abstract.AbstractText {
		if abstractBuilder.Len() > 0 {
			abstractBuilder.WriteString(" ")
		}
		abstractBuilder.WriteString(abstractText.Text)
	}

	// Extract authors
	var authors []models.Author
	for _, auth := range article.AuthorList.Authors {
		fullName := fmt.Sprintf("%s %s", auth.ForeName, auth.LastName)
		if strings.TrimSpace(fullName) == "" {
			fullName = auth.LastName
		}

		authors = append(authors, models.Author{
			LastName: auth.LastName,
			ForeName: auth.ForeName,
			Initials: auth.Initials,
			FullName: strings.TrimSpace(fullName),
		})
	}

	// Extract MeSH headings
	var meshHeadings []string
	for _, mesh := range pubmedArticle.MedlineCitation.MeshHeadingList.MeshHeadings {
		meshHeadings = append(meshHeadings, mesh.DescriptorName.Text)
	}

	// Extract publication types
	var pubTypes []string
	for _, pubType := range article.PublicationTypeList.PublicationTypes {
		pubTypes = append(pubTypes, pubType)
	}

	// Parse publication date
	publicationDate := parsePubMedDate(
		pubmedArticle.MedlineCitation.ArticleDate.Year,
		pubmedArticle.MedlineCitation.ArticleDate.Month,
		pubmedArticle.MedlineCitation.ArticleDate.Day,
	)

	// Extract DOI from ArticleIdList
	doi := ""
	for _, articleID := range pubmedArticle.PubmedData.ArticleIdList.ArticleIds {
		if articleID.IdType == "doi" {
			doi = articleID.Text
			break
		}
	}

	return models.MedicalArticle{
		ID:               pubmedArticle.MedlineCitation.PMID,
		Title:            CleanMedicalText(article.ArticleTitle),
		Abstract:         CleanMedicalText(abstractBuilder.String()),
		Authors:          authors,
		PublishedDate:    publicationDate,
		DOI:              doi,
		Journal:          article.Journal.Title,
		JournalAbbr:      article.Journal.ISOAbbreviation,
		Source:           "pubmed",
		MeshHeadings:     meshHeadings,
		PublicationTypes: pubTypes,
		Affiliation:      getFirstAffiliation(article.AuthorList.Authors),
	}
}

func parsePubMedDate(yearStr, monthStr, dayStr string) time.Time {
	// Default to January if no month
	if monthStr == "" {
		monthStr = "01"
	}
	// Default to 1st if no day
	if dayStr == "" {
		dayStr = "01"
	}

	// Convert to integers
	year, _ := strconv.Atoi(yearStr)
	month, _ := strconv.Atoi(monthStr)
	day, _ := strconv.Atoi(dayStr)

	// Validate and create date
	if year == 0 {
		return time.Time{} // Zero time for invalid dates
	}
	if month < 1 || month > 12 {
		month = 1
	}
	if day < 1 || day > 31 {
		day = 1
	}

	return time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.UTC)
}

func getFirstAffiliation(authors []struct {
	LastName    string `xml:"LastName"`
	ForeName    string `xml:"ForeName"`
	Initials    string `xml:"Initials"`
	Affiliation string `xml:"Affiliation"`
}) string {
	for _, author := range authors {
		if author.Affiliation != "" {
			return author.Affiliation
		}
	}
	return ""
}
