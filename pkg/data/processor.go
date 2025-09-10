package data

import (
	"fmt"
	"regexp"
	"strings"
	"time"

	"MedAtlasAIServer/internal/models"

	strip "github.com/grokify/html-strip-tags-go"
)

// CleanMedicalText removes HTML tags, special characters, and normalizes text
func CleanMedicalText(text string) string {
	if text == "" {
		return ""
	}

	// Remove HTML tags
	text = strip.StripTags(text)

	// Remove special characters but keep medical terminology, hyphens, and parentheses
	text = regexp.MustCompile(`[^\w\s\-\.\,\(\)\&\/]`).ReplaceAllString(text, " ")

	// Normalize whitespace
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// NormalizeMedicalTerms expands common medical abbreviations
func NormalizeMedicalTerms(text string) string {
	if text == "" {
		return ""
	}

	// Common medical abbreviations expansion
	abbreviations := map[string]string{
		"PT( therapy)?": "Physical Therapy", // Regex for context
		"PT( time)?":    "Prothrombin Time", // Regex for context
		"PT/INR":        "Prothrombin Time/International Normalized Ratio",
		"MRI":           "Magnetic Resonance Imaging",
		"CT":            "Computed Tomography",
		"PET":           "Positron Emission Tomography",
		"AI":            "Artificial Intelligence",
		"NLP":           "Natural Language Processing",
		"ER":            "Emergency Room",
		"ED":            "Emergency Department",
		"DNA":           "Deoxyribonucleic Acid",
		"RNA":           "Ribonucleic Acid",
		"COVID-19":      "Coronavirus Disease 2019",
		"HIV":           "Human Immunodeficiency Virus",
		"AIDS":          "Acquired Immunodeficiency Syndrome",
		"FDA":           "Food and Drug Administration",
		"NIH":           "National Institutes of Health",
		"WHO":           "World Health Organization",
		"CDC":           "Centers for Disease Control and Prevention",
		"ECG":           "Electrocardiogram",
		"EEG":           "Electroencephalogram",
		"EMG":           "Electromyography",
		"ICU":           "Intensive Care Unit",
		"OR":            "Operating Room",
		"OT":            "Occupational Therapy",
		"Rx":            "Prescription",
		"Dx":            "Diagnosis",
		"Tx":            "Treatment",
		"Hx":            "History",
		"Sx":            "Symptoms",
		"RO":            "Rule Out",
		"SOB":           "Shortness of Breath",
		"CP":            "Chest Pain",
		"HA":            "Headache",
		"HTN":           "Hypertension",
		"DM":            "Diabetes Mellitus",
		"CAD":           "Coronary Artery Disease",
		"CHF":           "Congestive Heart Failure",
		"COPD":          "Chronic Obstructive Pulmonary Disease",
		"ARDS":          "Acute Respiratory Distress Syndrome",
		"DVT":           "Deep Vein Thrombosis",
		"PE":            "Pulmonary Embolism",
		"MI":            "Myocardial Infarction",
		"CVA":           "Cerebrovascular Accident",
		"TIA":           "Transient Ischemic Attack",
		"GBS":           "Guillain-Barré Syndrome",
		"MS":            "Multiple Sclerosis",
		"ALS":           "Amyotrophic Lateral Sclerosis",
		"PD":            "Parkinson's Disease",
		"AD":            "Alzheimer's Disease",
		"RA":            "Rheumatoid Arthritis",
		"SLE":           "Systemic Lupus Erythematosus",
		"IBD":           "Inflammatory Bowel Disease",
		"IBS":           "Irritable Bowel Syndrome",
		"GERD":          "Gastroesophageal Reflux Disease",
		"PUD":           "Peptic Ulcer Disease",
		"CKD":           "Chronic Kidney Disease",
		"ESRD":          "End Stage Renal Disease",
		"UTI":           "Urinary Tract Infection",
		"STI":           "Sexually Transmitted Infection",
		"PID":           "Pelvic Inflammatory Disease",
		"OCP":           "Oral Contraceptive Pill",
		"IUD":           "Intrauterine Device",
		"HRT":           "Hormone Replacement Therapy",
		"BRCA":          "Breast Cancer gene",
		"PSA":           "Prostate-Specific Antigen",
		"CEA":           "Carcinoembryonic Antigen",
		"AFP":           "Alpha-Fetoprotein",
		"CA":            "Cancer",
		"CA-125":        "Cancer Antigen 125",
		"CA-19-9":       "Cancer Antigen 19-9",
		"WBC":           "White Blood Cell",
		"RBC":           "Red Blood Cell",
		"HGB":           "Hemoglobin",
		"HCT":           "Hematocrit",
		"PLT":           "Platelet",
		"INR":           "International Normalized Ratio",
		"PTT":           "Partial Thromboplastin Time",
		"ALT":           "Alanine Aminotransferase",
		"AST":           "Aspartate Aminotransferase",
		"ALP":           "Alkaline Phosphatase",
		"GGT":           "Gamma-Glutamyl Transferase",
		"BUN":           "Blood Urea Nitrogen",
		"Cr":            "Creatinine",
		"Na":            "Sodium",
		"K":             "Potassium",
		"Cl":            "Chloride",
		"CO2":           "Carbon Dioxide",
		"Ca":            "Calcium",
		"Mg":            "Magnesium",
		"PO4":           "Phosphate",
		"LFT":           "Liver Function Test",
		"BMP":           "Basic Metabolic Panel",
		"CMP":           "Comprehensive Metabolic Panel",
		"CBC":           "Complete Blood Count",
		"ABG":           "Arterial Blood Gas",
		"VQ":            "Ventilation-Perfusion",
		"CPR":           "Cardiopulmonary Resuscitation",
		"ACLS":          "Advanced Cardiac Life Support",
		"PALS":          "Pediatric Advanced Life Support",
		"BLS":           "Basic Life Support",
		"CCU":           "Coronary Care Unit",
		"PICU":          "Pediatric Intensive Care Unit",
		"NICU":          "Neonatal Intensive Care Unit",
		"SICU":          "Surgical Intensive Care Unit",
		"MICU":          "Medical Intensive Care Unit",
		"ERCP":          "Endoscopic Retrograde Cholangiopancreatography",
		"EGD":           "Esophagogastroduodenoscopy",
		"COLON":         "Colonoscopy",
		"EUS":           "Endoscopic Ultrasound",
		"US":            "Ultrasound",
		"USG":           "Ultrasonography",
	}

	// Replace abbreviations
	for abbr, full := range abbreviations {
		re := regexp.MustCompile(`\b` + abbr + `\b`)
		text = re.ReplaceAllString(text, full)
	}

	return text
}

// EnhanceArticle extracts key concepts and detects medical terminology
func EnhanceArticle(article *models.MedicalArticle) *models.MedicalArticle {
	if article == nil {
		return nil
	}

	// Extract key medical concepts from title and abstract
	fullText := article.Title + " " + article.Abstract
	article.KeyConcepts = ExtractKeyConcepts(fullText)
	article.HasMedicalTerms = ContainsMedicalTerm(fullText)

	return article
}

// ValidateArticle checks if an article meets quality standards
func ValidateArticle(article models.MedicalArticle) bool {
	// Check absolutely required fields
	if article.ID == "" || article.Title == "" {
		return false
	}

	// Check if title is meaningful (not too short or placeholder)
	cleanTitle := strings.TrimSpace(article.Title)
	if len(cleanTitle) < 5 {
		return false
	}

	// Check for placeholder or invalid titles
	invalidTitlePatterns := []string{
		"undefined", "null", "none", "unknown",
		"[No title]", "No title", "Untitled",
	}
	for _, pattern := range invalidTitlePatterns {
		if strings.EqualFold(cleanTitle, pattern) {
			return false
		}
	}

	// Check if abstract is present and meaningful
	if article.Abstract == "" {
		// Allow articles without abstract but with good title
		if len(cleanTitle) < 15 {
			return false
		}
	} else if len(strings.TrimSpace(article.Abstract)) < 30 {
		// Abstract is too short to be useful
		return false
	}

	// Validate publication date - be more lenient with dates
	if !article.PublishedDate.IsZero() {
		// Shouldn't be too far in future
		if article.PublishedDate.After(time.Now().AddDate(2, 0, 0)) {
			return false
		}

		// Shouldn't be too old (PubMed started in 1960s, but be lenient)
		if article.PublishedDate.Before(time.Date(1960, 1, 1, 0, 0, 0, 0, time.UTC)) {
			return false
		}
	}

	// Additional quality checks
	if isLowQualityArticle(article) {
		return false
	}

	return true
}

func isLowQualityArticle(article models.MedicalArticle) bool {
	// Check for retraction notices, errata, etc.
	lowerTitle := strings.ToLower(article.Title)
	retractionIndicators := []string{
		"retraction", "retracted", "withdrawal", "withdrawn",
		"erratum", "corrigendum", "expression of concern",
		"notice of", "editorial note", "comment on",
	}

	for _, indicator := range retractionIndicators {
		if strings.Contains(lowerTitle, indicator) {
			return true
		}
	}

	// Check for very short non-informative content
	if len(article.Abstract) > 0 && len(article.Abstract) < 50 && len(article.Title) < 20 {
		return true
	}

	return false
}

// FormatAuthors converts author objects to a readable string
func FormatAuthors(authors []models.Author) string {
	if len(authors) == 0 {
		return "Unknown Author"
	}

	var builder strings.Builder
	for i, author := range authors {
		if i > 0 {
			if i == len(authors)-1 {
				builder.WriteString(" and ")
			} else {
				builder.WriteString(", ")
			}
		}

		if author.FullName != "" {
			builder.WriteString(author.FullName)
		} else if author.LastName != "" && author.ForeName != "" {
			builder.WriteString(fmt.Sprintf("%s %s", author.ForeName, author.LastName))
		} else if author.LastName != "" {
			builder.WriteString(author.LastName)
		} else {
			builder.WriteString("Unknown Author")
		}
	}

	return builder.String()
}

// ExtractKeyConcepts identifies important medical terms from text
func ExtractKeyConcepts(text string) []string {
	if text == "" {
		return nil
	}

	// Comprehensive medical terminology dictionary
	medicalTerms := map[string]bool{
		// Diseases and Conditions
		"diabetes": true, "cancer": true, "hypertension": true, "arthritis": true,
		"asthma": true, "migraine": true, "depression": true, "anxiety": true,
		"osteoporosis": true, "alzheimer": true, "parkinson": true,
		"epilepsy": true, "schizophrenia": true, "fibromyalgia": true, "lupus": true,
		"multiple sclerosis": true, "crohn": true, "colitis": true, "hepatitis": true,
		"hiv": true, "aids": true, "tuberculosis": true, "malaria": true,
		"pneumonia": true, "bronchitis": true, "emphysema": true, "copd": true,

		// Symptoms
		"pain": true, "fever": true, "fatigue": true, "nausea": true, "vomiting": true,
		"headache": true, "dizziness": true, "rash": true, "swelling": true,
		"inflammation": true, "bleeding": true, "shortness of breath": true,
		"chest pain": true, "palpitations": true, "numbness": true, "weakness": true,

		// Treatments and Procedures
		"surgery": true, "chemotherapy": true, "radiotherapy": true, "immunotherapy": true,
		"medication": true, "antibiotics": true, "antiviral": true, "antifungal": true,
		"vaccine": true, "transplant": true, "dialysis": true, "biopsy": true,
		"endoscopy": true, "colonoscopy": true, "mri": true, "ct scan": true,
		"x-ray": true, "ultrasound": true, "blood test": true, "genetic testing": true,

		// Body Systems and Anatomy
		"cardiac": true, "pulmonary": true, "neurological": true, "gastrointestinal": true,
		"renal": true, "hepatic": true, "endocrine": true, "musculoskeletal": true,
		"dermatological": true, "ophthalmological": true, "otolaryngological": true,
		"psychological": true, "immunological": true, "hematological": true,

		// Medical Specialties
		"oncology": true, "cardiology": true, "neurology": true, "psychiatry": true,
		"pediatrics": true, "geriatrics": true, "radiology": true,
		"pathology": true, "pharmacology": true, "epidemiology": true, "toxicology": true,
		"dermatology": true, "endocrinology": true, "gastroenterology": true,
		"nephrology": true, "pulmonology": true, "rheumatology": true, "urology": true,
	}

	words := strings.Fields(text)
	var concepts []string
	seen := make(map[string]bool)

	for _, word := range words {
		cleanWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}"))

		// Check for multi-word concepts first
		if len(cleanWord) > 3 && medicalTerms[cleanWord] && !seen[cleanWord] {
			concepts = append(concepts, word) // Preserve original case
			seen[cleanWord] = true
		}
	}

	return removeDuplicates(concepts)
}

// removeDuplicates removes duplicate strings from a slice
func removeDuplicates(slice []string) []string {
	seen := make(map[string]bool)
	result := []string{}

	for _, item := range slice {
		lowerItem := strings.ToLower(item)
		if !seen[lowerItem] {
			seen[lowerItem] = true
			result = append(result, item)
		}
	}

	return result
}

// IsValidDOI validates DOI format
func IsValidDOI(doi string) bool {
	if doi == "" {
		return false
	}
	return regexp.MustCompile(`^10\.\d{4,9}/[-._;()/:A-Z0-9]+$`).MatchString(doi)
}

// IsValidDate validates date format (YYYY-MM-DD)
func IsValidDate(dateStr string) bool {
	if dateStr == "" {
		return false
	}
	return regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`).MatchString(dateStr)
}

// TruncateText truncates text to specified length with ellipsis
func TruncateText(text string, maxLength int) string {
	if len(text) <= maxLength {
		return text
	}

	// Find a good breaking point (end of sentence or word)
	if maxLength > 3 {
		// Try to break at sentence end
		if pos := strings.LastIndex(text[:maxLength-3], ". "); pos != -1 && pos > maxLength/2 {
			return text[:pos+1] + "..."
		}
		// Try to break at word boundary
		if pos := strings.LastIndex(text[:maxLength-3], " "); pos != -1 && pos > maxLength/2 {
			return text[:pos] + "..."
		}
	}

	return text[:maxLength-3] + "..."
}

// ExtractYearFromDate extracts year from time.Time
func ExtractYearFromDate(date time.Time) int {
	if date.IsZero() {
		return 0
	}
	return date.Year()
}

// CalculateArticleAge calculates how old an article is in years
func CalculateArticleAge(publishedDate time.Time) int {
	if publishedDate.IsZero() {
		return 0
	}
	age := time.Since(publishedDate)
	return int(age.Hours() / 24 / 365.25)
}

// IsRecentArticle checks if article is from recent years
func IsRecentArticle(publishedDate time.Time, years int) bool {
	if publishedDate.IsZero() {
		return false
	}
	return time.Since(publishedDate) <= time.Duration(years)*365*24*time.Hour
}

// ContainsMedicalTerm checks if text contains medical terminology - NOW USED!
func ContainsMedicalTerm(text string) bool {
	if text == "" {
		return false
	}

	medicalIndicators := []string{
		// Medical conditions
		"patient", "treatment", "therapy", "diagnosis", "clinical", "medical",
		"disease", "symptom", "procedure", "surgery", "medication", "drug",
		"hospital", "clinic", "doctor", "physician", "nurse", "health",
		"illness", "disorder", "syndrome", "infection", "inflammatory",

		// Body systems
		"cardiac", "pulmonary", "neurological", "gastrointestinal", "renal",
		"hepatic", "endocrine", "musculoskeletal", "dermatological",

		// Medical procedures
		"operation", "transplant", "biopsy", "scan", "test", "imaging",
		"injection", "infusion", "transfusion", "dialysis",

		// Medications
		"antibiotic", "antiviral", "antifungal", "anti-inflammatory",
		"analgesic", "antidepressant", "antipsychotic", "vaccine",
	}

	lowerText := strings.ToLower(text)
	for _, term := range medicalIndicators {
		if strings.Contains(lowerText, term) {
			return true
		}
	}

	return false
}

// ExtractMedicalEntities extracts structured medical information - NEW!
func ExtractMedicalEntities(text string) map[string][]string {
	entities := map[string][]string{
		"conditions":  {},
		"symptoms":    {},
		"treatments":  {},
		"procedures":  {},
		"medications": {},
		"body_parts":  {},
	}

	// Simple pattern matching for entity extraction
	patterns := map[string][]string{
		"conditions": {
			`\b(diabetes|hypertension|arthritis|asthma|cancer|migraine|depression|anxiety)\b`,
			`\b(heart disease|lung disease|kidney disease|liver disease)\b`,
			`\b(alzheimer|parkinson|epilepsy|schizophrenia|fibromyalgia|lupus)\b`,
		},
		"symptoms": {
			`\b(pain|fever|fatigue|nausea|headache|dizziness|rash|swelling)\b`,
			`\b(shortness of breath|chest pain|palpitations|numbness|weakness)\b`,
		},
		"treatments": {
			`\b(surgery|chemotherapy|radiotherapy|immunotherapy|medication)\b`,
			`\b(physical therapy|occupational therapy|speech therapy)\b`,
		},
		"medications": {
			`\b(antibiotic|antiviral|antifungal|anti-inflammatory|analgesic)\b`,
			`\b(antidepressant|antipsychotic|vaccine|insulin|metformin)\b`,
		},
	}

	for entityType, regexPatterns := range patterns {
		for _, pattern := range regexPatterns {
			re := regexp.MustCompile(pattern)
			matches := re.FindAllString(text, -1)
			entities[entityType] = append(entities[entityType], matches...)
		}
		entities[entityType] = removeDuplicates(entities[entityType])
	}

	return entities
}

// CalculateMedicalRelevance scores how medical a text is - NEW!
func CalculateMedicalRelevance(text string) float64 {
	if text == "" {
		return 0.0
	}

	words := strings.Fields(text)
	if len(words) == 0 {
		return 0.0
	}

	medicalWords := 0
	medicalTerms := []string{
		"patient", "treatment", "diagnosis", "symptom", "disease",
		"therapy", "clinical", "medical", "hospital", "doctor",
		"medication", "surgery", "procedure", "test", "scan",
	}

	lowerText := strings.ToLower(text)
	for _, term := range medicalTerms {
		if strings.Contains(lowerText, term) {
			medicalWords++
		}
	}

	// Calculate ratio of medical terms to total words
	return float64(medicalWords) / float64(len(words))
}

// IsClinicalStudy checks if article appears to be a clinical study - NEW!
func IsClinicalStudy(article models.MedicalArticle) bool {
	// Check publication types
	clinicalTypes := []string{
		"Clinical Trial", "Randomized Controlled Trial", "Clinical Study",
		"Case Report", "Case Series", "Observational Study",
	}

	for _, pubType := range article.PublicationTypes {
		for _, clinicalType := range clinicalTypes {
			if strings.EqualFold(pubType, clinicalType) {
				return true
			}
		}
	}

	// Check abstract for clinical indicators
	abstract := strings.ToLower(article.Abstract)
	clinicalIndicators := []string{
		"clinical trial", "randomized", "controlled study", "patient cohort",
		"treatment group", "placebo", "double-blind", "follow-up",
	}

	for _, indicator := range clinicalIndicators {
		if strings.Contains(abstract, indicator) {
			return true
		}
	}

	return false
}

func ValidateArticleWithReason(article models.MedicalArticle) (bool, string) {
	if article.ID == "" {
		return false, "missing ID"
	}

	if article.Title == "" {
		return false, "missing title"
	}

	cleanTitle := strings.TrimSpace(article.Title)
	if len(cleanTitle) < 5 {
		return false, fmt.Sprintf("title too short: '%s'", cleanTitle)
	}

	// Check for placeholder titles
	invalidTitlePatterns := []string{"undefined", "null", "none", "unknown"}
	for _, pattern := range invalidTitlePatterns {
		if strings.EqualFold(cleanTitle, pattern) {
			return false, fmt.Sprintf("invalid title: '%s'", cleanTitle)
		}
	}

	if article.Abstract == "" && len(cleanTitle) < 15 {
		return false, "no abstract and title too short"
	}

	if len(strings.TrimSpace(article.Abstract)) < 30 && article.Abstract != "" {
		return false, "abstract too short"
	}

	if !article.PublishedDate.IsZero() {
		if article.PublishedDate.After(time.Now().AddDate(2, 0, 0)) {
			return false, "future date"
		}
		if article.PublishedDate.Before(time.Date(1960, 1, 1, 0, 0, 0, 0, time.UTC)) {
			return false, "date too old"
		}
	}

	if isLowQualityArticle(article) {
		return false, "low quality article"
	}

	return true, ""
}
