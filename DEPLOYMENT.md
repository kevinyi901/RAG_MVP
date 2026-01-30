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
- **Storage**: 100GB+ (55GB models + 20GB database + buffers)
- **Network**: For initial model download (can be air-gapped after)

### Development Requirements
- Docker or Podman
- Python 3.10+
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
- Smaller downloads: ~14GB total vs 55GB full precision
- Smaller storage footprint
- Faster transfer for air-gapped deployment
- Still excellent inference quality
- ~10% speed penalty (40-80 tokens/sec vs 50-100)
- Fits easily on all GPUs

### 4-bit Quantized Models (AWQ)

| Model | Size | Download Time | Source |
|-------|------|---------------|--------|
| gpt-oss-20b-AWQ | ~10GB | 5-10 min | HuggingFace: TheBloke/gpt-oss-20B-AWQ |
| mistral-7b-AWQ | ~4GB | 2-3 min | HuggingFace: TheBloke/Mistral-7B-Instruct-v0.2-AWQ |

### Download Instructions

**Prerequisites:**
```bash
pip install huggingface-hub
```

**Download quantized models:**
```bash
mkdir -p models

# gpt-oss-20b-AWQ (10GB)
huggingface-cli download TheBloke/gpt-oss-20B-AWQ \
  --local-dir ./models/gpt-oss-20b-awq \
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
du -sh ./models/mistral-7b
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
mkdir -p models
# ... (download commands as above)

# 2. Create offline packages directory
mkdir -p offline_packages/{images,models,wheels}

# 3. Copy models
cp -r models/gpt-oss-20b offline_packages/models/
cp -r models/mistral-7b offline_packages/models/

# 4. Export container images
docker save -o offline_packages/images/rag-postgres.tar pgvector/pgvector:pg15
docker save -o offline_packages/images/rag-vllm.tar vllm/vllm-openai:latest
docker save -o offline_packages/images/rag-app.tar rag-app:latest

# Compress for transfer
cd offline_packages
tar czf ../rag-offline.tar.gz .
```

### Phase 2: Transfer to Air-Gapped Machine

Transfer the following to the air-gapped system:
- `rag-offline.tar.gz` (container images + models)
- `scripts/` directory
- `containers/` directory
- `app/` directory
- `requirements.txt`
- `.env.example`

### Phase 3: Deployment on Air-Gapped Machine

```bash
# 1. Extract offline packages
tar xzf rag-offline.tar.gz -C ./

# 2. Load container images
podman load -i offline_packages/images/rag-postgres.tar
podman load -i offline_packages/images/rag-vllm.tar
podman load -i offline_packages/images/rag-app.tar

# 3. Setup system (requires sudo)
sudo ./scripts/setup_rhel.sh

# 4. Copy environment file
cp .env.example .env
# Edit .env as needed

# 5. Start the stack
podman-compose -f containers/podman-compose.yml up -d

# 6. Verify deployment
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
- All 89 Python packages are production dependencies
- No development-only packages in requirements.txt
- All transitive dependencies explicitly listed for air-gap compatibility

## Model Integrity (SHA256)

After downloading models, compute checksums for security verification:

```bash
# For gpt-oss-20b
find ./models/gpt-oss-20b -type f -exec sha256sum {} \; > gpt-oss-20b.sha256

# For mistral-7b
find ./models/mistral-7b -type f -exec sha256sum {} \; > mistral-7b.sha256

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
- Verify models are in `./models/gpt-oss-20b` and `./models/mistral-7b`
- Check docker-compose volumes are mounted correctly
- Ensure model file permissions are readable

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
# Pull latest images
docker compose pull

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

**Last Updated**: 2026-01-30
**Version**: 1.0
**Python**: 3.10+
**Status**: Ready for production deployment
