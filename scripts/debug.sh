#!/bin/bash

echo "=== Debugging Audio Conversion ==="

# Check original file
echo "1. Original file info:"
file doctor_patient_conversation.wav
ls -la doctor_patient_conversation.wav
echo "Original file duration:"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 doctor_patient_conversation.wav || echo "Cannot get duration"

# Convert properly
echo -e "\n2. Converting to proper WAV..."
ffmpeg -i doctor_patient_conversation.wav -acodec pcm_s16le -ar 16000 -ac 1 -f wav converted.wav -y

echo -e "\n3. Converted file info:"
file converted.wav
ls -la converted.wav
echo "Converted file duration:"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 converted.wav || echo "Cannot get duration"

# Test base64 encoding
echo -e "\n4. Testing base64 encoding..."
base64_length=$(base64 -w 0 converted.wav | wc -c)
echo "Base64 length: $base64_length characters"

# Test if base64 is valid
echo -e "\n5. Validating base64..."
base64 -w 0 converted.wav > test.b64
base64 -d test.b64 > test_decoded.wav 2>/dev/null
if [ $? -eq 0 ] && [ -s test_decoded.wav ]; then
    echo "✓ Base64 encoding is valid"
    echo "Decoded file size: $(wc -c < test_decoded.wav) bytes"
else
    echo "✗ Base64 encoding failed"
fi

# Cleanup
rm -f test.b64 test_decoded.wav

echo -e "\n=== Debug Complete ==="