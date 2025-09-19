package models

type ProcessAudioRequest struct {
	AudioData []byte `json:"audio_data"`
	FileName  string `json:"file_name"`
}

type ProcessAudioResponse struct {
	Status     string `json:"status"`
	Transcript string `json:"transcript"`
	SOAPNote   string `json:"soap_note"`
	Error      string `json:"error,omitempty"`
}

type EmbedRequest struct {
	Text string `json:"text"`
}

type EmbedResponse struct {
	Vector []float32 `json:"vector"`
	Model  string    `json:"model"`
	Dims   int       `json:"dims"`
}
