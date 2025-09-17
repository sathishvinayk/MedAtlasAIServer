import requests
import json

def test_embedding():
    """Test the embedding service"""
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health")
        print(f"Health check: {health_response.status_code} - {health_response.json()}")
        
        # Test embedding endpoint
        embed_response = requests.post(
            "http://localhost:8000/embed",
            json={"text": "heart disease treatment options"},
            headers={"Content-Type": "application/json"}
        )
        
        if embed_response.status_code == 200:
            result = embed_response.json()
            print(f"Embedding successful! Vector length: {result['dims']}")
            print(f"Model: {result['model']}")
            print(f"First 5 values: {result['vector'][:5]}")
        else:
            print(f"Embedding failed: {embed_response.status_code} - {embed_response.text}")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_embedding()