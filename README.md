# RAG MVP - Air-Gapped RAG System

A Retrieval-Augmented Generation (RAG) system designed for air-gapped deployment on RHEL 8.10 with GPU support.

## Features

- **Document Ingestion**: Upload PDF, DOCX, TXT, and Markdown files
- **OCR Support**: Process scanned documents using Tesseract
- **Vector Search**: pgvector for efficient similarity search
- **BM25 Reranking**: Hybrid retrieval with BM25 scoring
- **GPU-Accelerated LLM**: Ollama with gpt-oss:20b model
- **Source Attribution**: See which documents and sections informed each answer
- **Chain of Thought**: View the model's reasoning process

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Podman Compose Stack                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Streamlit  │────│  RAG API    │────│  Ollama (GPU)       │  │
│  │  Frontend   │    │  (FastAPI)  │    │  - gpt-oss:20b      │  │
│  │  Port 8501  │    │  Port 8000  │    │  - nomic-embed-text │  │
│  └─────────────┘    └─────────────┘    │  Port 11434         │  │
│         │                  │           └─────────────────────┘  │
│         └──────────────────┼─────────────────────────────────┐  │
│                            │                                 │  │
│                    ┌───────▼───────┐                         │  │
│                    │  PostgreSQL   │                         │  │
│                    │  + pgvector   │                         │  │
│                    │  Port 5432    │                         │  │
│                    └───────────────┘                         │  │
└─────────────────────────────────────────────────────────────────┘
```

## System Requirements

### Target Environment (Air-Gapped)
- **OS**: RHEL 8.10
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **RAM**: 24GB minimum
- **Storage**: 50GB+ for images and models
- **Container Runtime**: Podman 4.x

### Development Environment
- Docker or Podman with GPU support
- Internet access for initial setup

## Software Versions

| Component | Version |
|-----------|---------|
| Python | 3.10.14 |
| Ollama | 0.1.23 |
| PostgreSQL | 15.10 |
| pgvector | 0.7.4 |
| FastAPI | 0.109.x |
| Streamlit | 1.31.x |

## Quick Start (Development)

### Using Docker

```bash
# Clone the repository
cd RAG_MVP

# Copy environment file
cp .env.example .env

# Start the stack (first run downloads models)
docker compose -f containers/docker-compose.dev.yml up -d

# Watch Ollama download models
docker logs -f rag-ollama-init

# Access the UI
open http://localhost:8501
```

### Using Podman

```bash
# Start with Podman
podman-compose -f containers/podman-compose.dev.yml up -d

# Check status
podman-compose -f containers/podman-compose.dev.yml ps
```

## Air-Gapped Deployment

### Phase 1: Preparation (Internet-Connected Machine)

```bash
# 1. Download all dependencies
./scripts/download_for_airgap.sh

# 2. Export container images
./scripts/export_images.sh
```

This creates:
- `offline_packages/images/rag_stack.tar.gz` - Container images with models
- `offline_packages/wheels/` - Python packages
- `offline_packages/rpms/` - Instructions for NVIDIA toolkit

### Phase 2: Transfer

Copy to the air-gapped machine:
- `offline_packages/images/rag_stack.tar.gz`
- `scripts/` directory
- `containers/` directory
- `.env.example`

### Phase 3: Deployment (Air-Gapped Machine)

```bash
# 1. Run setup script (installs Podman, configures GPU)
sudo ./scripts/setup_rhel.sh

# 2. Import container images
./scripts/import_images.sh

# 3. Copy and edit environment file
cp .env.example .env

# 4. Start the stack
podman-compose -f containers/podman-compose.yml up -d
```

### Systemd Service (Optional)

```bash
# Enable automatic start on boot
sudo systemctl enable --now rag-stack
```

## Usage

### Web Interface

1. Open http://localhost:8501
2. Upload documents using the sidebar
3. Wait for processing to complete (chunking + embedding)
4. Ask questions in the query box
5. View responses with source citations

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/documents` | GET | List all documents |
| `/api/documents/upload` | POST | Upload a document |
| `/api/documents/{id}` | DELETE | Delete a document |
| `/api/query` | POST | Execute RAG query |
| `/api/query/stream` | POST | Streaming RAG query |
| `/api/feedback` | POST | Submit user feedback |
| `/api/stats` | GET | System statistics |

### Example API Usage

```bash
# Upload a document
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@document.pdf"

# Query the system
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?"}'
```

## Configuration

Environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgresql://... | PostgreSQL connection string |
| `OLLAMA_HOST` | http://ollama:11434 | Ollama API endpoint |
| `EMBEDDING_MODEL` | nomic-embed-text | Embedding model name |
| `LLM_MODEL` | gpt-oss:20b | LLM model name |
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap tokens between chunks |
| `TOP_K_RETRIEVAL` | 20 | Chunks to retrieve |
| `TOP_K_RERANK` | 5 | Chunks after reranking |

## Model Swapping

Models can be swapped by changing environment variables. The system is not locked to specific models.

### Memory Budget

**System RAM (24GB):**
| Component | Allocation |
|-----------|------------|
| PostgreSQL | 4GB |
| Python/Streamlit/FastAPI | 4GB |
| Document processing (OCR) | 4GB |
| System/OS | 4GB |
| Buffer/headroom | 8GB |

**GPU VRAM (24GB):** Models run entirely on GPU.

### Compatible LLM Models

To swap models, edit `.env` and modify `LLM_MODEL` and/or `EMBEDDING_MODEL`, then restart.

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| gpt-oss:20b | ~10-12GB | Default, good quality |
| gemma2:27b | ~16-18GB | High quality, fits 24GB |
| gemma2:9b | ~6GB | Great quality for size |
| gemma2:2b | ~2GB | Fast, lightweight |
| llama2:13b | ~8GB | Balanced performance |
| llama2:7b | ~4GB | Faster, lower quality |
| mistral:7b | ~4GB | Good quality for size |
| mixtral:8x7b | ~26GB | Won't fit on 24GB |
| codellama:13b | ~8GB | Better for code docs |

### Compatible Embedding Models

| Model | VRAM | Dimensions | Notes |
|-------|------|------------|-------|
| nomic-embed-text | ~0.5GB | 768 | Default |
| mxbai-embed-large | ~0.7GB | 1024 | Higher quality |
| all-minilm | ~0.2GB | 384 | Fastest |

**Note:** If you change embedding dimensions, you must:
1. Update `init.sql` to match: `VECTOR(768)` → `VECTOR(1024)`
2. Re-ingest all documents (embeddings are dimension-specific)

### Air-Gap Model Selection

To include different models in your air-gap bundle, edit `scripts/download_for_airgap.sh`:

```bash
# Change these lines to your preferred models
ollama pull nomic-embed-text      # embedding model
ollama pull gpt-oss:20b           # LLM model
ollama pull mistral:7b            # additional model (optional)
```

## Project Structure

```
RAG_MVP/
├── app/
│   ├── main.py              # FastAPI backend
│   ├── streamlit_app.py     # Streamlit frontend
│   ├── rag/
│   │   ├── embeddings.py    # Ollama embedding service
│   │   ├── retriever.py     # pgvector retrieval
│   │   ├── ranker.py        # BM25 reranking
│   │   └── pipeline.py      # RAG orchestration
│   └── utils/
│       ├── document_loader.py  # PDF/DOCX/TXT parsing + OCR
│       └── chunker.py          # Token-based chunking
├── containers/
│   ├── Containerfile.app       # Python application image
│   ├── Containerfile.ollama    # Ollama with models
│   ├── docker-compose.dev.yml  # Docker development
│   ├── podman-compose.yml      # Podman production
│   ├── podman-compose.dev.yml  # Podman development
│   └── init.sql                # Database schema
├── scripts/
│   ├── download_for_airgap.sh  # Download dependencies
│   ├── export_images.sh        # Export container images
│   ├── import_images.sh        # Import on air-gapped machine
│   └── setup_rhel.sh           # RHEL 8.10 setup
├── requirements.txt
├── .env.example
└── README.md
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CDI configuration
nvidia-ctk cdi list

# Regenerate CDI spec
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### Ollama Connection Issues

```bash
# Check Ollama logs
podman logs rag-ollama

# Test Ollama API
curl http://localhost:11434/api/tags
```

### Database Connection Issues

```bash
# Check PostgreSQL logs
podman logs rag-postgres

# Test connection
psql -h localhost -U rag -d ragdb -c "SELECT 1;"
```

### Memory Issues

If the 20B model causes OOM:
1. Edit `.env` and change `LLM_MODEL` to a smaller model
2. Restart the stack

## License

[Your License Here]
