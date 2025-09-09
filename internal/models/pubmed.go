package models

import (
	"encoding/xml"
	"time"
)

// PubMed XML Response Structures
type PubMedResult struct {
	XMLName  xml.Name        `xml:"PubmedArticleSet"`
	Articles []PubMedArticle `xml:"PubmedArticle"`
}

type PubMedArticle struct {
	XMLName         xml.Name `xml:"PubmedArticle"`
	MedlineCitation struct {
		PMID    string `xml:"PMID"`
		Article struct {
			Journal struct {
				Title           string `xml:"Title"`
				ISOAbbreviation string `xml:"ISOAbbreviation"`
			} `xml:"Journal"`
			ArticleTitle string `xml:"ArticleTitle"`
			Abstract     struct {
				AbstractText []struct {
					Text        string `xml:",chardata"`
					Label       string `xml:"Label,attr"`
					NlmCategory string `xml:"NlmCategory,attr"`
				} `xml:"AbstractText"`
			} `xml:"Abstract"`
			AuthorList struct {
				Authors []struct {
					LastName    string `xml:"LastName"`
					ForeName    string `xml:"ForeName"`
					Initials    string `xml:"Initials"`
					Affiliation string `xml:"Affiliation"`
				} `xml:"Author"`
			} `xml:"AuthorList"`
			PublicationTypeList struct {
				PublicationTypes []string `xml:"PublicationType"`
			} `xml:"PublicationTypeList"`
		} `xml:"Article"`
		ArticleDate struct {
			Year  string `xml:"Year"`
			Month string `xml:"Month"`
			Day   string `xml:"Day"`
		} `xml:"ArticleDate"`
		MeshHeadingList struct {
			MeshHeadings []struct {
				DescriptorName struct {
					Text string `xml:",chardata"`
					UI   string `xml:"UI,attr"`
				} `xml:"DescriptorName"`
			} `xml:"MeshHeading"`
		} `xml:"MeshHeadingList"`
	} `xml:"MedlineCitation"`
	PubmedData struct {
		ArticleIdList struct {
			ArticleIds []struct {
				IdType string `xml:"IdType,attr"`
				Text   string `xml:",chardata"`
			} `xml:"ArticleId"`
		} `xml:"ArticleIdList"`
	} `xml:"PubmedData"`
}

// Normalized Article Structure
type MedicalArticle struct {
	ID               string    `json:"id"`
	Title            string    `json:"title"`
	Abstract         string    `json:"abstract"`
	Authors          []Author  `json:"authors"`
	PublishedDate    time.Time `json:"published_date"`
	DOI              string    `json:"doi"`
	Journal          string    `json:"journal"`
	JournalAbbr      string    `json:"journal_abbr"`
	Source           string    `json:"source"`
	MeshHeadings     []string  `json:"mesh_headings"`
	PublicationTypes []string  `json:"publication_types"`
	Affiliation      string    `json:"affiliation"`
	KeyConcepts      []string  `json:"key_concepts,omitempty"` // Now used!
	HasMedicalTerms  bool      `json:"has_medical_terms"`
}

type Author struct {
	LastName string `json:"last_name"`
	ForeName string `json:"fore_name"`
	Initials string `json:"initials"`
	FullName string `json:"full_name"`
}

// ESearch Response
type ESearchResult struct {
	Count    string   `xml:"Count"`
	IdList   []string `xml:"IdList>Id"`
	WebEnv   string   `xml:"WebEnv"`
	QueryKey string   `xml:"QueryKey"`
}
