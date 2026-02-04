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
- **GPU**: NVIDIA A10G (23GB), A40 (48GB), or RTX 3090/4090 (24GB)
- **NVIDIA Driver**: 570.x from CUDA repo (NOT elrepo — 580.x is incompatible)
- **CPU**: 8+ cores
- **RAM**: 32GB minimum
- **Storage**: 100GB+ (45GB models + 20GB Docker images + database + buffers)
- **Network**: For initial model download (can be air-gapped after)

### Development Requirements (EC2 Spot Instance)
- RHEL 8.10 GPU instance (e.g., g5.xlarge with A10G)
- NVIDIA driver 570.x from CUDA repo
- Docker with NVIDIA Container Toolkit
- Python 3.12+
- Internet access (for initial model downloads)
- AWS security group: inbound ports 22 (SSH), 8501 (Streamlit), 8001 (API), 8000 (vLLM)

## Model Download

### GPU Memory Requirements

| GPU | VRAM | gpt-oss-20b (70%) | mistral-7b-awq (25%) | Status |
|-----|------|-------------------|---------------------|--------|
| A10G | 23GB | ~16GB | ~5.7GB | Both models fit |
| A40 | 48GB | ~33GB | ~12GB | Both models + headroom |
| RTX 3090 | 24GB | ~16.8GB | ~6GB | Both models fit |
| RTX 4090 | 24GB | ~16.8GB | ~6GB | Both models fit |

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
bash scripts/download_llm_models.sh
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

## Development Deployment (EC2 Spot Instance)

### 1. EC2 Instance Setup

```bash
# Install NVIDIA driver 570 from CUDA repo (NOT elrepo — 580.x causes CUDA Error 803)
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

### 2. Verify GPU Access After Reboot

```bash
nvidia-smi                         # Should show driver 570.x, CUDA 12.8
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 3. Deploy the Stack

```bash
cd RAG_MVP

# Download models
pip install huggingface-hub
bash scripts/download_llm_models.sh

# Pull vLLM image and start
docker pull vllm/vllm-openai:latest
docker compose -f containers/docker-compose.dev.yml up -d

# vLLM takes a few minutes to load models into GPU memory.
# Wait for healthy, then re-run to start dependent services:
docker compose -f containers/docker-compose.dev.yml up -d

# Check status
docker compose -f containers/docker-compose.dev.yml ps

# Access services (requires EC2 security group inbound rules):
# - Streamlit UI: http://<ec2-public-ip>:8501
# - FastAPI Backend: http://<ec2-public-ip>:8001
# - vLLM API: http://<ec2-public-ip>:8000
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
bash scripts/download_llm_models.sh
# Models will be saved to ./models/gpt-oss-20b and ./models/mistral-7b-awq

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

### CUDA Error 803: Unsupported Display Driver / CUDA Driver Combination

This means the NVIDIA driver is incompatible with the CUDA runtime in the vLLM container.

- **Cause**: Driver 580.x (from elrepo) is too new for vLLM's CUDA 12.9 runtime
- **Fix**: Install driver 570.x from the NVIDIA CUDA repo:
  ```bash
  sudo dnf remove nvidia-x11-drv nvidia-x11-drv-libs kmod-nvidia -y
  sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
  sudo dnf module install nvidia-driver:570 -y
  sudo reboot
  ```
- **Verify**: `nvidia-smi` should show driver 570.x, CUDA 12.8

### Out of Memory
- Reduce `TOP_K_RETRIEVAL` in .env
- Ensure 32GB+ RAM available
- Check GPU VRAM usage: `nvidia-smi`
- Both models use 95% of GPU memory (70% + 25%) — this is expected

### Models Not Found
- Verify models are in `./models/gpt-oss-20b` and `./models/mistral-7b-awq`
- Check compose volumes are mounted correctly (`../models:/models:ro`)
- Ensure model file permissions are readable: `sudo chmod -R 755 models/`
- If models directory was created by Docker (root-owned): `sudo rm -rf models && mkdir models`

### vLLM Takes Too Long to Start
- Model loading into GPU memory can exceed the 120s health check start period
- Run `docker compose -f containers/docker-compose.dev.yml up -d` again after vLLM is healthy
- Monitor with: `docker logs -f rag-vllm`

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

**Last Updated**: 2026-02-04
**Version**: 1.2
**Python**: 3.12.11
**PostgreSQL**: 16.8
**Streamlit**: 1.51.0
**vLLM**: latest (CUDA 12.9)
**NVIDIA Driver**: 570.x (from CUDA repo)
**Status**: Ready for air-gapped production deployment
