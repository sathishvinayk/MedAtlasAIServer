#!/bin/bash

# Setup script for Mistral AI medical chat app

echo "🚀 Setting up Medical Chat App with Mistral AI..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Pull Mistral model
echo "🔽 Downloading Mistral 7B Instruct model..."
ollama pull mistral:7b-instruct

# Start Ollama service
echo "🔄 Starting Ollama service..."
ollama serve &

# Wait for service to start
sleep 10

# Test the model
echo "🧪 Testing Mistral model..."
ollama run mistral:7b-instruct "Hello, are you ready for medical conversations?"

echo "✅ Setup complete! Run 'docker-compose -f docker-compose.mistral.yml up -d' to start the app."