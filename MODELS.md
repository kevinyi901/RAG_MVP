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

1. **Download models here:** `bash scripts/download_llm_models.sh`
2. **Commit code to git** (models in .gitignore, not committed)
3. **Transfer to air-gapped machine:**
   - Code directory (from git)
   - `models/` directory (local, ~45GB)
4. **Deploy:** See DEPLOYMENT.md

### Model Specs

- **gpt-oss-20b**: 22B MoE (3.6B active), ~41GB on disk, ~16GB VRAM (MXFP4 at inference)
- **mistral-7b-awq**: 4-bit quantized, 4GB, ~80 tokens/sec
- **Both fit on**: A10G (23GB), A40 (48GB), RTX 3090/4090 (24GB) — combined 0.70 + 0.25 GPU memory utilization

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
