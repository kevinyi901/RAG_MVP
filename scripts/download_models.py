#!/usr/bin/env python3
"""Download LLM models from Hugging Face Hub"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface-hub not installed. Install with: pip install huggingface-hub")
    sys.exit(1)

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODE = sys.argv[1] if len(sys.argv) > 1 else "quantized"

if MODE not in ["full", "quantized", "both"]:
    print(f"Usage: {sys.argv[0]} [full|quantized|both]")
    print("  full      - Download full precision models (~55GB)")
    print("  quantized - Download 4-bit quantized models (~14GB)")
    print("  both      - Download both versions (~69GB)")
    sys.exit(1)

def download_full():
    print("\n========================================")
    print("Downloading Full Precision Models")
    print("========================================")
    
    print("Downloading gpt-oss-20b (13GB - BF16)...")
    snapshot_download("openai/gpt-oss-20b", 
                     local_dir=str(MODELS_DIR / "gpt-oss-20b"),
                     cache_dir=None)
    print("✓ gpt-oss-20b (BF16)")

def download_quantized():
    print("\n========================================")
    print("Downloading 4-bit Quantized Models (AWQ)")
    print("========================================")
    
    print("Downloading gpt-oss-20b AWQ (6GB)...")
    snapshot_download("TheBloke/gpt-oss-20B-AWQ",
                     local_dir=str(MODELS_DIR / "gpt-oss-20b-awq"),
                     cache_dir=None)
    print("✓ gpt-oss-20b AWQ (4-bit)")
    
    print("Downloading Mistral 7B Instruct v0.2 AWQ (5GB)...")
    snapshot_download("TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                     local_dir=str(MODELS_DIR / "mistral-7b-awq"),
                     cache_dir=None)
    print("✓ Mistral 7B Instruct v0.2 AWQ")

if __name__ == "__main__":
    try:
        if MODE in ["full", "both"]:
            download_full()
        if MODE in ["quantized", "both"]:
            download_quantized()
        
        print("\n✓ All downloads complete!")
        print(f"✓ Models saved to: {MODELS_DIR}")
        print("\nNEXT STEPS:")
        print("  1. Update docker-compose files with your model paths")
        print("  2. Start deployment: docker-compose -f containers/docker-compose.dev.yml up")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}", file=sys.stderr)
        sys.exit(1)
