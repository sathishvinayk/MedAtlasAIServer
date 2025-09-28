#!/bin/bash

echo "=== Sending Audio to Server ==="

# Encode to base64
base64_data=$(base64 -w 0 doctor_patient_conversation.wav)

echo "Base64 length: ${#base64_data} characters"

# Create JSON manually but properly formatted
cat > request.json << EOF
{
    "audio_data": "$base64_data",
    "file_name": "doctor_patient_conversation.wav"
}
EOF

echo "Sending request to server..."
curl -X POST http://localhost:8080/process-audio \
  -H "Content-Type: application/json" \
  -d "@request.json" \
  -w "\nHTTP Status: %{http_code}\n"

rm -f request.json
echo "Done!"