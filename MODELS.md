# Model Download Quick Start

## For Air-Gapped Deployment: Download Models NOW

Since your deployment environment has no internet, download the models on this machine first.

### Quick Download

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Download models (~45GB total)
bash scripts/download_llm_models.sh

# Verify
du -sh models/
# Should show: ~45GB total
```

### What Gets Downloaded

```
models/
├── gpt-oss-20b/        (~41GB on disk, ~16GB VRAM with MXFP4)
│   ├── model-*.safetensors
│   ├── config.json
│   └── ...
└── mistral-7b-awq/     (~4GB)
    ├── model-*.safetensors
    ├── config.json
    └── ...
```

### For Development (EC2 with Internet)

```bash
# Pull vLLM image
docker pull vllm/vllm-openai:latest

# Start containers with models
docker compose -f containers/docker-compose.dev.yml up -d

# vLLM takes a few minutes to load — wait, then re-run up -d
docker compose -f containers/docker-compose.dev.yml up -d

# Check vLLM is running
curl http://localhost:8000/health

# Access UI (requires EC2 security group port 8501 open)
open http://<ec2-public-ip>:8501
```

### For Air-Gapped Deployment

```bash
# Download everything (LLM models, embedding model, wheels, container images)
bash scripts/download_for_airgap.sh

# Package into a single offline tar.gz
bash scripts/export_images.sh

# Transfer rag-offline-package.tar.gz to the air-gapped machine
```

See DEPLOYMENT.md for full instructions.

### Model Specs

**LLM Models (served by vLLM):**
- **gpt-oss-20b**: 22B MoE (3.6B active), ~41GB on disk, ~16GB VRAM (MXFP4 at inference)
- **mistral-7b-awq**: 4-bit quantized, 4GB, ~80 tokens/sec
- **Both fit on**: A10G (23GB), A40 (48GB), RTX 3090/4090 (24GB) — combined 0.70 + 0.25 GPU memory utilization

**Embedding Model (local, CPU-based via sentence-transformers):**
- **nomic-ai/nomic-embed-text-v1.5**: 768 dimensions, 137M params, ~550MB on disk
- Runs on CPU inside the API container (no GPU needed)
- Requires `trust_remote_code=True` and `einops` package
- Pre-downloaded to `models/embedding/nomic-embed-text-v1.5/` for air-gap

### Troubleshooting

**Download fails - "No space left"**
```bash
du -sh models/  # Check current size
df -h           # Check disk space
```

**Network timeout**
```bash
# Resume with increased timeout
huggingface-cli download openai/gpt-oss-20b \
  --local-dir ./models/gpt-oss-20b \
  --local-dir-use-symlinks False
```

**Model not loading in vLLM**
```bash
# Check models are readable
ls -lh models/gpt-oss-20b/model-*.safetensors
chmod -R 755 models/
```

---

**Next Step:** Run deployment commands in DEPLOYMENT.md
