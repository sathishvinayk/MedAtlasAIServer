curl -X POST http://localhost:8000/process-audio \
  -H "Content-Type: application/json" \
  -d "{
    \"audio_data\": \"$(base64 -i doctor_patient_conversation.mp3 | tr -d '\n')\",
    \"file_name\": \"doctor_patient_conversation.mp3\"
  }"