package main

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

type PubMedSearchResponse struct {
	ESearchResult struct {
		Count    string   `json:"count"`
		IdList   []string `json:"idlist"`
		WebEnv   string   `json:"webenv"`
		QueryKey string   `json:"querykey"`
	} `json:"esearchresult"`
}

// Enhanced PubMed article structure with better XML tags
type PubMedArticle struct {
	XMLName         xml.Name `xml:"PubmedArticle"`
	MedlineCitation struct {
		PMID struct {
			Text string `xml:",chardata" json:"pmid"`
		} `xml:"PMID" json:"pmid"`
		Article struct {
			Journal struct {
				Title           string `xml:"Title" json:"title"`
				ISOAbbreviation string `xml:"ISOAbbreviation" json:"iso_abbreviation"`
			} `xml:"Journal" json:"journal"`
			ArticleTitle struct {
				Text string `xml:",chardata" json:"article_title"`
			} `xml:"ArticleTitle" json:"article_title"`
			Abstract struct {
				AbstractText []struct {
					Text        string `xml:",chardata" json:"text"`
					Label       string `xml:"Label,attr" json:"label,omitempty"`
					NlmCategory string `xml:"NlmCategory,attr" json:"nlm_category,omitempty"`
				} `xml:"AbstractText" json:"abstract_text"`
			} `xml:"Abstract" json:"abstract"`
			AuthorList struct {
				Author []struct {
					LastName        string `xml:"LastName" json:"last_name"`
					ForeName        string `xml:"ForeName" json:"fore_name"`
					Initials        string `xml:"Initials" json:"initials"`
					AffiliationInfo struct {
						Affiliation string `xml:"Affiliation" json:"affiliation"`
					} `xml:"AffiliationInfo" json:"affiliation_info,omitempty"`
				} `xml:"Author" json:"author"`
			} `xml:"AuthorList" json:"author_list"`
		} `xml:"Article" json:"article"`
		DateCompleted struct {
			Year  string `xml:"Year" json:"year"`
			Month string `xml:"Month" json:"month"`
			Day   string `xml:"Day" json:"day"`
		} `xml:"DateCompleted" json:"date_completed"`
		DateRevised struct {
			Year  string `xml:"Year" json:"year"`
			Month string `xml:"Month" json:"month"`
			Day   string `xml:"Day" json:"day"`
		} `xml:"DateRevised" json:"date_revised"`
	} `xml:"MedlineCitation" json:"medline_citation"`
	PubmedData struct {
		History struct {
			PubMedPubDate []struct {
				PubStatus string `xml:"PubStatus,attr" json:"pub_status"`
				Year      string `xml:"Year" json:"year"`
				Month     string `xml:"Month" json:"month"`
				Day       string `xml:"Day" json:"day"`
			} `xml:"PubMedPubDate" json:"pubmed_pub_date"`
		} `xml:"History" json:"history"`
		ArticleIdList struct {
			ArticleId []struct {
				IdType string `xml:"IdType,attr" json:"id_type"`
				Text   string `xml:",chardata" json:"text"`
			} `xml:"ArticleId" json:"article_id"`
		} `xml:"ArticleIdList" json:"article_id_list"`
	} `xml:"PubmedData" json:"pubmed_data"`
}

type PubMedArticleSet struct {
	XMLName  xml.Name        `xml:"PubmedArticleSet"`
	Articles []PubMedArticle `xml:"PubmedArticle"`
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	fmt.Println("Fetching real PubMed articles with debug logging...")

	// Medical topics to search
	medicalTopics := []string{
		"cancer immunotherapy",
		"cardiovascular disease treatment",
		"diabetes management",
		"mental health therapy",
		"infectious diseases",
	}

	outputFile := "data/pubmed_real_articles.jsonl"
	maxResultsPerTopic := 5 // Start small for debugging

	var allArticles []map[string]interface{}

	for _, topic := range medicalTopics {
		fmt.Printf("\n=== Searching for: %s ===\n", topic)

		articleIDs, err := searchPubMed(topic, maxResultsPerTopic)
		if err != nil {
			log.Printf("Error searching for %s: %v", topic, err)
			continue
		}

		fmt.Printf("Found %d article IDs: %v\n", len(articleIDs), articleIDs)

		if len(articleIDs) > 0 {
			articles, err := fetchPubMedArticles(articleIDs)
			if err != nil {
				log.Printf("Error fetching articles for %s: %v", topic, err)
				continue
			}

			fmt.Printf("Successfully parsed %d articles for: %s\n", len(articles), topic)
			allArticles = append(allArticles, articles...)

			// Be nice to PubMed API
			time.Sleep(2 * time.Second)
		}
	}

	// Save to JSONL file
	if err := saveArticlesToFile(allArticles, outputFile); err != nil {
		log.Fatal("Error saving articles:", err)
	}

	fmt.Printf("\n=== FINAL RESULTS ===\n")
	fmt.Printf("Successfully saved %d real PubMed articles to %s\n", len(allArticles), outputFile)

	if len(allArticles) > 0 {
		fmt.Printf("\nFirst article sample:\n")
		jsonData, _ := json.MarshalIndent(allArticles[0], "", "  ")
		fmt.Println(string(jsonData))
	}
}

func searchPubMed(query string, maxResults int) ([]string, error) {
	fmt.Printf("Searching PubMed for: %s (max: %d)\n", query, maxResults)

	baseURL := "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

	params := url.Values{}
	params.Add("db", "pubmed")
	params.Add("term", query)
	params.Add("retmax", strconv.Itoa(maxResults))
	params.Add("retmode", "json")
	params.Add("sort", "relevance")
	params.Add("field", "title/abstract")

	url := baseURL + "?" + params.Encode()
	fmt.Printf("API URL: %s\n", url)

	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response status: %s\n", resp.Status)

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response failed: %w", err)
	}

	fmt.Printf("Response body size: %d bytes\n", len(body))

	// Debug: log first 200 chars of response
	if len(body) > 0 {
		fmt.Printf("Response preview: %.200s...\n", string(body))
	}

	var result PubMedSearchResponse
	if err := json.Unmarshal(body, &result); err != nil {
		fmt.Printf("Raw JSON response: %s\n", string(body))
		return nil, fmt.Errorf("JSON parsing failed: %w", err)
	}

	fmt.Printf("Parsed result: Count=%s, IDs=%v\n", result.ESearchResult.Count, result.ESearchResult.IdList)
	return result.ESearchResult.IdList, nil
}

func fetchPubMedArticles(articleIDs []string) ([]map[string]interface{}, error) {
	fmt.Printf("Fetching details for %d articles: %v\n", len(articleIDs), articleIDs)

	if len(articleIDs) == 0 {
		return nil, nil
	}

	baseURL := "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

	params := url.Values{}
	params.Add("db", "pubmed")
	params.Add("id", strings.Join(articleIDs, ","))
	params.Add("retmode", "xml")
	params.Add("rettype", "abstract")

	url := baseURL + "?" + params.Encode()
	fmt.Printf("Fetch URL: %s\n", url)

	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Fetch response status: %s\n", resp.Status)

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response failed: %w", err)
	}

	fmt.Printf("Fetch response size: %d bytes\n", len(body))

	// Save raw XML for debugging
	if err := os.WriteFile("debug_pubmed_response.xml", body, 0644); err != nil {
		log.Printf("Warning: Could not save debug XML: %v", err)
	} else {
		fmt.Println("Saved raw XML response to debug_pubmed_response.xml")
	}

	// Debug: log XML structure
	if len(body) > 0 {
		fmt.Printf("XML preview: %.500s...\n", string(body))
	}

	articles, err := parsePubMedXML(body)
	if err != nil {
		fmt.Printf("XML parsing error: %v\n", err)
		fmt.Printf("Problematic XML: %.1000s...\n", string(body))
		return nil, err
	}

	fmt.Printf("Successfully parsed %d articles from XML\n", len(articles))
	return articles, nil
}

func parsePubMedXML(xmlData []byte) ([]map[string]interface{}, error) {
	fmt.Println("Parsing PubMed XML...")

	// Check if XML contains expected structure
	if !strings.Contains(string(xmlData), "<PubmedArticleSet>") {
		fmt.Println("Warning: XML doesn't contain PubmedArticleSet tag")
		fmt.Printf("XML starts with: %.200s...\n", string(xmlData))
	}

	var articleSet PubMedArticleSet
	if err := xml.Unmarshal(xmlData, &articleSet); err != nil {
		fmt.Printf("XML Unmarshal error: %v\n", err)

		// Try to find where the error occurs
		lines := strings.Split(string(xmlData), "\n")
		for i, line := range lines {
			if strings.Contains(line, "<PubmedArticle>") {
				fmt.Printf("Found PubmedArticle at line %d: %s\n", i+1, strings.TrimSpace(line))
			}
		}

		return nil, fmt.Errorf("XML parsing failed: %w", err)
	}

	fmt.Printf("Found %d articles in XML\n", len(articleSet.Articles))

	var articles []map[string]interface{}

	for i, pubmedArticle := range articleSet.Articles {
		fmt.Printf("Processing article %d/%d\n", i+1, len(articleSet.Articles))

		article := convertPubMedArticleToDocument(pubmedArticle)
		if article != nil {
			fmt.Printf("  -> Valid article: %s\n", article["title"])
			articles = append(articles, article)
		} else {
			fmt.Printf("  -> Skipped article (missing required fields)\n")
		}
	}

	return articles, nil
}

func convertPubMedArticleToDocument(pubmedArticle PubMedArticle) map[string]interface{} {
	fmt.Printf("Converting article with PMID: %s\n", pubmedArticle.MedlineCitation.PMID.Text)

	// Extract DOI from article IDs
	var doi string
	for _, articleID := range pubmedArticle.PubmedData.ArticleIdList.ArticleId {
		if articleID.IdType == "doi" {
			doi = articleID.Text
			fmt.Printf("  Found DOI: %s\n", doi)
			break
		}
	}

	// Combine abstract text
	var abstractBuilder strings.Builder
	for _, abstractText := range pubmedArticle.MedlineCitation.Article.Abstract.AbstractText {
		if abstractBuilder.Len() > 0 {
			abstractBuilder.WriteString(" ")
		}
		abstractBuilder.WriteString(abstractText.Text)
	}
	abstract := abstractBuilder.String()
	fmt.Printf("  Abstract length: %d characters\n", len(abstract))

	// Format authors
	var authorsBuilder strings.Builder
	for i, author := range pubmedArticle.MedlineCitation.Article.AuthorList.Author {
		if i > 0 {
			authorsBuilder.WriteString(", ")
		}
		authorsBuilder.WriteString(author.LastName)
		if author.Initials != "" {
			authorsBuilder.WriteString(" ")
			authorsBuilder.WriteString(author.Initials)
		}
	}
	authors := authorsBuilder.String()
	fmt.Printf("  Authors: %s\n", authors)

	// Use publication date from history (most reliable)
	var pubDate string
	for _, historyDate := range pubmedArticle.PubmedData.History.PubMedPubDate {
		if historyDate.PubStatus == "pubmed" {
			pubDate = fmt.Sprintf("%s-%s-%s", historyDate.Year, historyDate.Month, historyDate.Day)
			fmt.Printf("  Publication date: %s\n", pubDate)
			break
		}
	}

	// Fallback to other dates
	if pubDate == "" && pubmedArticle.MedlineCitation.DateRevised.Year != "" {
		pubDate = fmt.Sprintf("%s-%s-%s",
			pubmedArticle.MedlineCitation.DateRevised.Year,
			pubmedArticle.MedlineCitation.DateRevised.Month,
			pubmedArticle.MedlineCitation.DateRevised.Day,
		)
	}

	// Ensure we have required fields
	pmid := pubmedArticle.MedlineCitation.PMID.Text
	title := pubmedArticle.MedlineCitation.Article.ArticleTitle.Text

	if pmid == "" || title == "" {
		fmt.Printf("  SKIPPING - Missing required fields: PMID=%s, Title=%s\n", pmid, title)
		return nil
	}

	article := map[string]interface{}{
		"id":             pmid,
		"title":          title,
		"abstract":       abstract,
		"authors":        authors,
		"published_date": pubDate,
		"doi":            doi,
		"journal":        pubmedArticle.MedlineCitation.Article.Journal.Title,
		"journal_abbr":   pubmedArticle.MedlineCitation.Article.Journal.ISOAbbreviation,
		"source":         "pubmed",
		"pmid":           pmid,
	}

	fmt.Printf("  Successfully converted article: %s\n", title[:min(50, len(title))]+"...")
	return article
}

func saveArticlesToFile(articles []map[string]interface{}, filename string) error {
	fmt.Printf("Saving %d articles to %s\n", len(articles), filename)

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("creating file failed: %w", err)
	}
	defer file.Close()

	for i, article := range articles {
		jsonData, err := json.Marshal(article)
		if err != nil {
			log.Printf("Error marshaling article %d: %v", i, err)
			continue
		}
		file.Write(jsonData)
		file.WriteString("\n")
	}

	fmt.Printf("Finished saving articles to file\n")
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
