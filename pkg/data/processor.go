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

	words := strings.Fields(text)
	var concepts []string

	// Comprehensive medical terminology dictionary
	medicalTerms := map[string]bool{
		// General Medical Terms
		"patient": true, "treatment": true, "therapy": true, "diagnosis": true,
		"prognosis": true, "symptoms": true, "clinical": true, "medical": true,
		"health": true, "disease": true, "disorder": true, "syndrome": true,
		"condition": true, "illness": true, "infection": true, "inflammatory": true,
		"chronic": true, "acute": true, "severe": true, "mild": true, "moderate": true,

		// Body Systems
		"cardiac": true, "cardiovascular": true, "pulmonary": true, "respiratory": true,
		"neurological": true, "nervous": true, "gastrointestinal": true, "digestive": true,
		"hepatic": true, "renal": true, "urinary": true, "reproductive": true,
		"endocrine": true, "metabolic": true, "musculoskeletal": true, "hematological": true,
		"immunological": true, "dermatological": true, "ophthalmological": true,
		"otolaryngological": true, "psychiatric": true, "psychological": true,

		// Medical Specialties
		"oncology": true, "cardiology": true, "neurology": true, "psychiatry": true,
		"pediatrics": true, "geriatrics": true, "surgery": true, "radiology": true,
		"pathology": true, "pharmacology": true, "epidemiology": true, "toxicology": true,
		"anesthesiology": true, "dermatology": true, "endocrinology": true, "gastroenterology": true,
		"hematology": true, "nephrology": true, "pulmonology": true, "rheumatology": true,
		"urology": true, "ophthalmology": true, "otolaryngology": true, "orthopedics": true,

		// Treatments and Procedures
		"medication": true, "drug": true, "pharmaceutical": true,
		"procedure": true, "operation": true, "transplant": true, "transplantation": true,
		"biopsy": true, "resection": true, "excision": true, "incision": true,
		"drainage": true, "aspiration": true, "injection": true, "infusion": true,
		"transfusion": true, "dialysis": true, "ventilation": true, "resuscitation": true,
		"rehabilitation": true, "chemotherapy": true, "radiotherapy": true,
		"immunotherapy": true, "targeted": true, "biological": true, "vaccine": true,
		"antibiotic": true, "antiviral": true, "antifungal": true, "antiinflammatory": true,

		// Diagnostic Terms
		"screening": true, "detection": true, "monitoring": true,
		"assessment": true, "evaluation": true, "examination": true, "test": true,
		"assay": true, "biomarker": true, "marker": true, "indicator": true,
		"sign": true, "symptom": true, "finding": true, "result": true,
		"positive": true, "negative": true, "abnormal": true, "normal": true,
		"elevated": true, "reduced": true, "increased": true, "decreased": true,

		// Research Terms
		"study": true, "trial": true, "research": true, "investigation": true,
		"analysis": true, "observation": true,
		"experiment": true, "randomized": true, "controlled": true,
		"prospective": true, "retrospective": true, "cohort": true, "case": true,
		"control": true, "cross-sectional": true, "longitudinal": true, "meta-analysis": true,
		"systematic": true, "review": true, "literature": true, "evidence": true,
		"data": true, "results": true, "findings": true, "conclusion": true,
		"significance": true, "correlation": true, "association": true, "risk": true,
		"factor": true, "predictor": true, "outcome": true, "mortality": true,
		"morbidity": true, "survival": true, "recurrence": true, "progression": true,
		"response": true, "efficacy": true, "safety": true, "tolerability": true,
		"adverse": true, "event": true, "side": true, "effect": true,
		"complication": true, "toxicity": true, "interaction": true, "contraindication": true,

		// Statistical Terms
		"statistical": true, "method": true, "model": true,
		"regression": true, "value": true,
		"confidence": true, "interval": true, "odds": true, "ratio": true,
		"hazard": true, "adjusted": true, "multivariate": true,
		"univariate": true, "sensitivity": true, "specificity": true, "accuracy": true,
		"precision": true, "recall": true, "auc": true, "roc": true,
	}

	for _, word := range words {
		cleanWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}"))
		if len(cleanWord) > 3 && medicalTerms[cleanWord] {
			// Preserve original case for display
			concepts = append(concepts, word)
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

// ContainsMedicalTerm checks if text contains medical terminology
func ContainsMedicalTerm(text string) bool {
	medicalIndicators := []string{
		"patient", "treatment", "therapy", "diagnosis", "clinical", "medical",
		"disease", "symptom", "procedure", "surgery", "medication", "drug",
		"hospital", "clinic", "doctor", "physician", "nurse", "health",
	}

	lowerText := strings.ToLower(text)
	for _, term := range medicalIndicators {
		if strings.Contains(lowerText, term) {
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
