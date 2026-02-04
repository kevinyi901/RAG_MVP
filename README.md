# RAG MVP - Air-Gapped RAG System

A Retrieval-Augmented Generation (RAG) system designed for air-gapped deployment on RHEL 8.10 with GPU support.

## Features

- **Pre-loaded Documents**: System works with documents loaded during deployment
- **Vector Search**: pgvector for efficient similarity search
- **Semantic Ranking**: Pure vector similarity ranking based on embeddings
- **GPU-Accelerated LLM**: vLLM with gpt-oss-20b model
- **Source Attribution**: See which documents and sections informed each answer
- **Chain of Thought**: View the model's reasoning process

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Podman Compose Stack                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Streamlit  │────│  RAG API    │────│  vLLM (GPU)         │  │
│  │  Frontend   │    │  (FastAPI)  │    │  - gpt-oss-20b      │  │
│  │  Port 8501  │    │  Port 8001  │    │  Port 8000          │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                  │                                    │
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
- **GPU**: NVIDIA A10G (23GB), A40 (48GB), or RTX 3090/4090 (24GB)
- **NVIDIA Driver**: 570.x from CUDA repo (NOT elrepo) — supports CUDA 12.8
- **RAM**: 32GB minimum (64GB recommended)
- **Storage**: 100GB+ for full models, 50GB+ for quantized
- **Container Runtime**: Podman 4.x or Docker with NVIDIA Container Toolkit

### Development Environment (EC2 Spot Instance)
- RHEL 8.10 GPU instance (e.g., g5.xlarge with A10G)
- NVIDIA driver 570.x from CUDA repo
- Docker with NVIDIA Container Toolkit
- Internet access for initial setup

## Software Versions

| Component | Version |
|-----------|---------|
| Python | 3.12.11 |
| vLLM | latest (CUDA 12.9, with quantization support) |
| PostgreSQL | 16.8 |
| pgvector | 0.2.5 |
| FastAPI | 0.109.x |
| Streamlit | 1.51.0 |
| NVIDIA Driver | 570.x (from CUDA repo, supports CUDA 12.8) |
| CUDA (container) | 12.9 (bundled in vLLM image) |

## Quick Start (Development on EC2)

### 1. EC2 Instance Setup (RHEL 8.10 with GPU)

```bash
# Install NVIDIA driver 570 from CUDA repo (NOT elrepo — 580.x is incompatible with vLLM)
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf module install nvidia-driver:570 -y

# Install NVIDIA Container Toolkit
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit

# Install and configure Docker
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Reboot to load the driver
sudo reboot
```

### 2. Verify GPU Access

```bash
nvidia-smi                         # Should show driver 570.x, CUDA 12.8
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 3. Download Models and Start

```bash
cd RAG_MVP
pip install huggingface-hub
bash scripts/download_llm_models.sh

docker pull vllm/vllm-openai:latest
docker compose -f containers/docker-compose.dev.yml up -d

# vLLM takes a few minutes to load models — wait for healthy, then re-run:
docker compose -f containers/docker-compose.dev.yml up -d
```

### 4. Access the UI

Open `http://<ec2-public-ip>:8501` (requires security group inbound rule for port 8501).

### Using Podman

```bash
podman-compose -f containers/podman-compose.dev.yml up -d
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

This creates a single `rag-offline-package.tar.gz` containing:
- Container images (rag-app, rag-vllm, rag-postgres)
- LLM models (gpt-oss-20b, mistral-7b-awq)
- Embedding model (nomic-ai/nomic-embed-text-v1.5)
- Python wheels for offline pip install
- Compose files, init.sql, .env.example, import script

### Phase 2: Transfer

Copy `rag-offline-package.tar.gz` to the air-gapped machine.

### Phase 3: Deployment (Air-Gapped Machine)

```bash
# 1. Extract the package
mkdir -p RAG_MVP && cd RAG_MVP
tar xzf rag-offline-package.tar.gz

# 2. Run setup script (installs Podman, configures GPU)
sudo bash scripts/setup_rhel.sh

# 3. Import container images
bash scripts/import_images.sh images/rag_stack.tar.gz

# 4. Copy and edit environment file
cp .env.example .env
# Edit .env as needed (e.g., change default passwords)

# 5. Start the stack
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
2. View available documents in the sidebar
3. Ask questions in the query box
4. View responses with source citations from pre-loaded documents

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/documents` | GET | List all documents |
| `/api/query` | POST | Execute RAG query |
| `/api/query/stream` | POST | Streaming RAG query |
| `/api/chat/stream` | POST | Streaming chat with conversation history |
| `/api/feedback` | POST | Submit user feedback |
| `/api/stats` | GET | System statistics |

### Example API Usage

```bash
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
| `VLLM_HOST` | http://vllm:8000 | vLLM API endpoint |
| `EMBEDDING_MODEL` | nomic-ai/nomic-embed-text-v1.5 | Embedding model name |
| `EMBEDDING_MODEL_PATH` | /models/embedding | Local path to embedding model files |
| `LLM_MODEL` | gpt-oss-20b | LLM model name |
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
| Document processing | 4GB |
| System/OS | 4GB |
| Buffer/headroom | 8GB |

**GPU VRAM (24GB):** Models run entirely on GPU.

### Compatible LLM Models

To swap models, edit `.env` and modify `LLM_MODEL`, then restart. Models must be in HuggingFace format and placed in the vLLM model directory.

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| gpt-oss-20b | ~10-12GB | Default, good quality |
| gemma2:27b | ~16-18GB | High quality, fits 24GB |
| gemma2:9b | ~6GB | Great quality for size |
| gemma2:2b | ~2GB | Fast, lightweight |
| llama2:13b | ~8GB | Balanced performance |
| llama2:7b | ~4GB | Faster, lower quality |
| mistral-7b | ~4GB | Good quality for size |
| codellama:13b | ~8GB | Better for code docs |

### Compatible Embedding Models

Embeddings are generated locally via sentence-transformers (no GPU service required).

| Model | Dimensions | Notes |
|-------|------------|-------|
| nomic-ai/nomic-embed-text-v1.5 | 768 | **Default - Good baseline** |
| mxbai-embed-large | 1024 | Higher quality |
| all-MiniLM-L6-v2 | 384 | Fastest, smallest |

**Note:** If you change embedding dimensions, you must:
1. Update `init.sql` to match: `VECTOR(1024)` → `VECTOR(new_dimensions)`
2. Re-ingest all documents (embeddings are dimension-specific)

### Air-Gap Model Selection

To include different models in your air-gap bundle, edit `scripts/download_for_airgap.sh` and change the model download commands for both the LLM (HuggingFace model) and embedding model (sentence-transformers model).

## Project Structure

```
RAG_MVP/
├── app/
│   ├── main.py              # FastAPI backend
│   ├── streamlit_app.py     # Streamlit frontend
│   ├── rag/
│   │   ├── embeddings.py    # sentence-transformers embedding service
│   │   ├── retriever.py     # pgvector retrieval
│   │   ├── ranker.py        # BM25 reranking
│   │   └── pipeline.py      # RAG orchestration
│   └── utils/
│       ├── document_loader.py  # PDF/DOCX/TXT parsing
│       └── chunker.py          # Token-based chunking
├── containers/
│   ├── Containerfile.app       # Python application image
│   ├── Containerfile.vllm      # vLLM with model
│   ├── docker-compose.dev.yml  # Docker development
│   ├── podman-compose.yml      # Podman production
│   ├── podman-compose.dev.yml  # Podman development
│   └── init.sql                # Database schema
├── scripts/
│   ├── download_for_airgap.sh  # Download all dependencies for air-gap
│   ├── download_llm_models.sh  # Download LLM models from HuggingFace
│   ├── download_wheels.sh      # Download Python wheels for offline install
│   ├── export_images.sh        # Package everything into offline tar.gz
│   ├── import_images.sh        # Import container images on air-gapped machine
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

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check CDI configuration (Podman)
nvidia-ctk cdi list

# Regenerate CDI spec (Podman)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### CUDA Driver Mismatch (Error 803)

If you see `system has unsupported display driver / cuda driver combination`:
- **Cause**: Driver 580.x (from elrepo) is incompatible with vLLM's CUDA 12.9 runtime
- **Fix**: Install driver 570.x from the NVIDIA CUDA repo instead:
  ```bash
  sudo dnf remove nvidia-x11-drv nvidia-x11-drv-libs kmod-nvidia -y
  sudo dnf module install nvidia-driver:570 -y
  sudo reboot
  ```

### vLLM Connection Issues

```bash
# Check vLLM logs
docker logs rag-vllm --tail 50

# Test vLLM API
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
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
