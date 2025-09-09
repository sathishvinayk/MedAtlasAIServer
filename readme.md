# Medical Literature Search Engine

A high-performance, scalable backend for a semantic search engine specialized in medical and scientific literature. Built with Go, Qdrant vector database, and Python AI services, this system enables natural language search across medical research papers and abstracts.

## 🚀 Features

- **Semantic Search**: Find relevant medical papers using natural language queries
- **Real PubMed Integration**: Direct integration with NIH PubMed API for live medical literature
- **Advanced Filtering**: Filter by date range, journal, authors, and medical subject headings
- **Vector-Based Retrieval**: State-of-the-art sentence transformers for accurate similarity matching
- **High Performance**: Built with Go for exceptional concurrency and low latency
- **Scalable Architecture**: Microservices design with Docker containerization
- **RESTful API**: Clean JSON API for easy integration
- **Web Interface**: Simple responsive web UI for searching and browsing
- **Biomedical Optimized**: Pre-configured with models trained on scientific and medical text

## 🛠️ Technology Stack

### Backend Services
- **Go 1.21+**: Primary backend language for API and indexer
- **Gorilla Mux**: HTTP router and middleware
- **Qdrant Go Client**: gRPC client for vector database operations

### AI/ML Components
- **Python 3.11**: Embedding service runtime
- **FastAPI**: Modern Python web framework for embeddings
- **Sentence Transformers**: State-of-the-art sentence embeddings
- **Hugging Face Models**: Pre-trained biomedical language models

### Infrastructure
- **Docker**: Containerization for all services
- **Docker Compose**: Multi-container orchestration
- **Qdrant**: Vector database for similarity search

### Recommended Models
- `sentence-transformers/all-MiniLM-L6-v2` (Default, 384-dim)
- `pubMedBERT-base-embeddings` (Biomedical specialized)
- `SPECTER` (Scientific paper embeddings)
- `GTE-Large` (General text embeddings)

## 📦 Installation & Setup

### Prerequisites

- **Docker** and **Docker Compose**
- **Go 1.21+** (for local development)
- **Python 3.11+** (for local embedding service development)

### Quick Start with Docker

1. **Clone and setup the repository**
   ```bash
   git clone <repository-url>
   cd medical-search-backend

2. **Build and start all services**
    ```bash
    docker-compose up -d --build

3. **Verify services are running**
    ```bash
    docker-compose ps

4. **Check service health**
    ```bash
    curl http://localhost:8080/health
    curl http://localhost:8000/health

5. **Run data collection (optional - uses real PubMed API)**
    ```bash
    go run scripts/data_sources/pubmed_collector.go

6. **Index the data**
    ```bash
    go run cmd/indexer/main.go

## 🚀 Manual Setup (Development)

1. **Start dependencies**
    ```bash
    docker-compose up -d qdrant embedding-service

2. **Install Go dependencies**
    ```bash
    go mod download

3. **Run the API server**
    ```bash
    go run cmd/api/main.go


