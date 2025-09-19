package models

type ProcessAudioRequest struct {
	AudioData []byte `json:"audio_data"`
	FileName  string `json:"file_name"`
}

type MedicalEntity struct {
	Entity     string  `json:"entity"`
	Text       string  `json:"text"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Confidence float64 `json:"confidence"`
}

type SpeakerSegment struct {
	Speaker string  `json:"speaker"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Text    string  `json:"text"`
}

type ProcessAudioResponse struct {
	Status               string           `json:"status"`
	Transcript           string           `json:"transcript"`
	Entities             []MedicalEntity  `json:"entities"` // Changed from []string
	SOAPNote             string           `json:"soap_note"`
	SpeakerSegments      []SpeakerSegment `json:"speaker_segments"` // Changed from string
	ModelUsed            string           `json:"model_used"`
	NLUModelUsed         string           `json:"nlu_model_used"`
	LLMModelUsed         string           `json:"llm_model_used"`
	DiarizationModelUsed string           `json:"diarization_model_used"`
	Error                string           `json:"error,omitempty"`
}

type EmbedRequest struct {
	Text string `json:"text"`
}

type EmbedResponse struct {
	Vector []float32 `json:"vector"`
	Model  string    `json:"model"`
	Dims   int       `json:"dims"`
}
