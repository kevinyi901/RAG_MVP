# Model Download Quick Start

## For Air-Gapped Deployment: Download Models NOW

Since your deployment environment has no internet, download the quantized models on this machine first.

### Quick Download

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Download quantized models (~14GB total, ~15 min)
./scripts/download_llm_models.sh quantized

# Verify
du -sh models/
# Should show: ~14GB total
```

### What Gets Downloaded

```
models/
├── gpt-oss-20b-awq/    (~10GB)
│   ├── model-*.safetensors
│   ├── config.json
│   └── ...
└── mistral-7b-awq/     (~4GB)
    ├── model-*.safetensors
    ├── config.json
    └── ...
```

### For Development (Internet Available)

```bash
# Start containers with models
docker compose -f containers/docker-compose.dev.yml up -d

# Check vLLM is running
curl http://localhost:8000/health

# Access UI
open http://localhost:8501
```

### For Air-Gapped Deployment

1. **Download models here:** `./scripts/download_llm_models.sh quantized`
2. **Commit code to git** (models in .gitignore, not committed)
3. **Transfer to air-gapped machine:**
   - Code directory (from git)
   - `models/` directory (local, ~14GB)
4. **Deploy:** See DEPLOYMENT.md

### Model Specs

- **gpt-oss-20b-awq**: 4-bit quantized, 10GB, ~50 tokens/sec
- **mistral-7b-awq**: 4-bit quantized, 4GB, ~80 tokens/sec
- **Both fit on**: A40 (48GB) with headroom

### Troubleshooting

**Download fails - "No space left"**
```bash
du -sh models/  # Check current size
df -h           # Check disk space
```

**Network timeout**
```bash
# Resume with increased timeout
huggingface-cli download TheBloke/gpt-oss-20B-AWQ \
  --local-dir ./models/gpt-oss-20b-awq \
  --local-dir-use-symlinks False
```

**Model not loading in vLLM**
```bash
# Check models are readable
ls -lh models/gpt-oss-20b-awq/model-*.safetensors
chmod -R 755 models/
```

---

**Next Step:** Run deployment commands in DEPLOYMENT.md
