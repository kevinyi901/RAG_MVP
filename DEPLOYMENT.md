# RAG MVP Deployment Guide

## Overview

This document provides deployment instructions and security information for the RAG MVP system. The system is designed for air-gapped deployment on RHEL 8.10 with GPU support.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Deployment Stack                           │
├──────────────────────────────────────────────────────────────┤
│  Streamlit UI       FastAPI Backend      vLLM LLM Server     │
│  (Port 8501)        (Port 8001)          (Port 8000)         │
│                                                               │
│  All Connected to PostgreSQL + pgvector for vector search    │
└──────────────────────────────────────────────────────────────┘
```

## System Requirements

### Minimum Production Requirements
- **OS**: RHEL 8.10+
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **CPU**: 8+ cores
- **RAM**: 32GB minimum
- **Storage**: 100GB+ (45GB models + 20GB database + buffers)
- **Network**: For initial model download (can be air-gapped after)

### Development Requirements
- Docker or Podman
- Python 3.12+
- Internet access (for initial model downloads)

## Model Download

### GPU Memory Requirements

| GPU | VRAM | Quantized 4-bit |
|-----|------|-----------------|
| A40 | 48GB | ✅ Both models + headroom |
| RTX 3090 | 24GB | ✅ Both models |
| RTX 4090 | 24GB | ✅ Both models |

### Recommended: 4-bit Quantized Models (AWQ)

**Advantages:**
- Native MXFP4 quantization applied at inference time by vLLM
- gpt-oss-20b: ~41GB on disk, but only ~16GB VRAM at runtime
- mistral-7b-AWQ: ~4GB on disk and VRAM
- Fits on RTX 3090/4090 (24GB) and A10G (23GB) with headroom

### 4-bit Quantized Models (AWQ)

| Model | Size | Download Time | Source |
|-------|------|---------------|--------|
| gpt-oss-20b | ~41GB (disk), ~16GB VRAM | 5-10 min | HuggingFace: openai/gpt-oss-20b |
| mistral-7b-AWQ | ~4GB | 2-3 min | HuggingFace: TheBloke/Mistral-7B-Instruct-v0.2-AWQ |

### Download Instructions

**Prerequisites:**
```bash
pip install huggingface-hub
```

**Download quantized models:**
```bash
mkdir -p models

# gpt-oss-20b (22B MoE, ~41GB on disk, ~16GB VRAM with MXFP4)
huggingface-cli download openai/gpt-oss-20b \
  --local-dir ./models/gpt-oss-20b \
  --local-dir-use-symlinks False

# mistral-7b-AWQ (4GB)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
  --local-dir ./models/mistral-7b-awq \
  --local-dir-use-symlinks False
```

**Or use the script:**
```bash
./scripts/download_llm_models.sh quantized
```

### Verify Model Downloads

After downloading, verify file integrity:

```bash
# List downloaded models
find ./models -type f -name "*.safetensors" -o -name "*.bin" | head -20

# Check size
du -sh ./models/gpt-oss-20b
du -sh ./models/mistral-7b-awq
```

## Development Deployment

### Using Docker Compose (Recommended)

```bash
# 1. Clone repository
cd RAG_MVP

# 2. Download models (see "Model Download" section above)
# Models should be in ./models/ directory

# 3. Start the stack
docker compose -f containers/docker-compose.dev.yml up -d

# 4. Check status
docker compose -f containers/docker-compose.dev.yml ps

# 5. Access services
# - Streamlit UI: http://localhost:8501
# - FastAPI Backend: http://localhost:8001
# - vLLM API: http://localhost:8000
# - PostgreSQL: localhost:5432
```

### Verify Deployment

```bash
# Check API health
curl http://localhost:8001/health

# Check vLLM health
curl http://localhost:8000/health

# View logs
docker compose -f containers/docker-compose.dev.yml logs -f api
```

### Shutdown

```bash
docker compose -f containers/docker-compose.dev.yml down

# Remove volumes (WARNING: deletes database)
docker compose -f containers/docker-compose.dev.yml down -v
```

## Air-Gapped Deployment (Production)

### Phase 1: Preparation (Internet-Connected Machine)

```bash
# 1. Download all models (see "Model Download" section)
./scripts/download_llm_models.sh quantized
# Models will be saved to ./models/gpt-oss-20b-awq and ./models/mistral-7b-awq

# 2. Download Python wheels for air-gapped install
./scripts/download_wheels.sh
# Wheels will be saved to ./wheels/ (~920MB)

# 3. Build the application container image
docker build -f containers/Containerfile.app -t localhost/rag-app:latest .

# 4. Pull and tag base images
docker pull pgvector/pgvector:pg16
docker tag pgvector/pgvector:pg16 localhost/rag-postgres:latest

docker pull vllm/vllm-openai:latest
docker tag vllm/vllm-openai:latest localhost/rag-vllm:latest

# 5. Create offline packages directory
mkdir -p offline_packages/{images,models}

# 6. Copy models
cp -r models/gpt-oss-20b offline_packages/models/
cp -r models/mistral-7b-awq offline_packages/models/

# 7. Export container images
docker save -o offline_packages/images/rag-postgres.tar localhost/rag-postgres:latest
docker save -o offline_packages/images/rag-vllm.tar localhost/rag-vllm:latest
docker save -o offline_packages/images/rag-app.tar localhost/rag-app:latest

# 8. Compress for transfer
cd offline_packages
tar czf ../rag-offline.tar.gz .
```

### Phase 2: Transfer to Air-Gapped Machine

Transfer the following to the air-gapped system:
- `rag-offline.tar.gz` (container images + models)
- `containers/` directory (compose files)
- `.env.example`

**Directory structure on air-gapped machine:**
```
RAG_MVP/
├── models/
│   ├── gpt-oss-20b/
│   └── mistral-7b-awq/
├── containers/
│   ├── docker-compose.dev.yml
│   ├── podman-compose.yml
│   └── init.sql
└── .env
```

### Phase 3: Deployment on Air-Gapped Machine

```bash
# 1. Extract offline packages
tar xzf rag-offline.tar.gz -C ./

# 2. Copy models to expected location
cp -r offline_packages/models/* ./models/

# 3. Load container images (images are tagged as localhost/*)
podman load -i offline_packages/images/rag-postgres.tar
podman load -i offline_packages/images/rag-vllm.tar
podman load -i offline_packages/images/rag-app.tar

# 4. Verify images are loaded
podman images | grep localhost

# 5. Copy environment file
cp .env.example .env
# Edit .env as needed

# 6. Start the stack
podman-compose -f containers/podman-compose.yml up -d

# 7. Verify deployment
podman-compose -f containers/podman-compose.yml ps
```

## Security Checklist

### Code Review
- [x] No hardcoded credentials in code
- [x] All secrets loaded from environment variables
- [x] Dependencies explicitly listed with exact versions
- [x] No unnecessary system packages (removed Java, tabula-py)
- [x] Semantic-only ranking (no BM25 keyword indexing)
- [x] Non-root container user (appuser)

### Model Security
- [ ] Verify HuggingFace model sources
- [ ] Check model file integrity (SHA256 checksums below)
- [ ] Scan for malicious content in model weights
- [ ] Document model licenses

### Deployment Security
- [ ] Change default PostgreSQL password
- [ ] Configure CORS properly for production
- [ ] Use HTTPS/SSL in production
- [ ] Enable container security policies
- [ ] Setup network isolation
- [ ] Enable audit logging

### Dependencies
- All ~110 Python packages are production dependencies
- No development-only packages in requirements.txt
- All transitive dependencies explicitly listed for air-gap compatibility
- Pre-downloaded wheels available in `wheels/` directory (~920MB)

## Model Integrity (SHA256)

After downloading models, compute checksums for security verification:

```bash
# For gpt-oss-20b
find ./models/gpt-oss-20b -type f -exec sha256sum {} \; > gpt-oss-20b.sha256

# For mistral-7b-awq
find ./models/mistral-7b-awq -type f -exec sha256sum {} \; > mistral-7b.sha256

# Verify (on air-gapped machine)
sha256sum -c gpt-oss-20b.sha256
sha256sum -c mistral-7b.sha256
```

## Environment Variables

Key environment variables for deployment:

```bash
# Database
DATABASE_URL=postgresql://rag:rag_password@postgres:5432/ragdb

# vLLM Configuration
VLLM_HOST=http://vllm:8000
LLM_MODEL=gpt-oss:20b
VLLM_MODEL_HOSTS=mistral:7b=http://vllm-mistral:8000

# Embedding Model
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_MODEL_PATH=/models/nomic-embed-text

# Document Processing
CHUNK_SIZE=500           # Token size for document chunks
CHUNK_OVERLAP=100        # Token overlap between chunks
TOP_K_RETRIEVAL=20       # Number of chunks to retrieve
TOP_K_RERANK=5           # Number of top results after ranking
```

## Troubleshooting

### Out of Memory
- Reduce `TOP_K_RETRIEVAL` in .env
- Ensure 32GB+ RAM available
- Check GPU VRAM (need 24GB+ for gpt-oss-20b)

### Models Not Found
- Verify models are in `./models/gpt-oss-20b` and `./models/mistral-7b-awq`
- Check compose volumes are mounted correctly (`../models:/models:ro`)
- Ensure model file permissions are readable
- All compose files use local images (`localhost/rag-*:latest`)

### Database Connection Failed
- Verify PostgreSQL container is running: `docker ps`
- Check DATABASE_URL in .env
- Ensure network connectivity between containers

### Embedding Model Download Fails
- Ensure internet access on first run
- Model will be cached in `~/.cache/huggingface/`
- For air-gapped, pre-download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1')"`

## Upgrade & Rollback

### Backup Database
```bash
# Backup PostgreSQL
docker exec rag-postgres pg_dump -U rag ragdb > backup.sql

# Restore from backup
docker exec -i rag-postgres psql -U rag ragdb < backup.sql
```

### Update Container Images
```bash
# For air-gapped: rebuild from updated wheels/source
docker build -f containers/Containerfile.app -t localhost/rag-app:latest .

# Restart services
docker compose -f containers/docker-compose.dev.yml up -d
```

## Support & Security Issues

For security issues, please:
1. Do not open public issues
2. Report to security team
3. Include model source verification
4. Include container image SHA256 hashes

---

**Last Updated**: 2026-02-02
**Version**: 1.1
**Python**: 3.12.11
**PostgreSQL**: 16.8
**Streamlit**: 1.51.0
**Status**: Ready for air-gapped production deployment
