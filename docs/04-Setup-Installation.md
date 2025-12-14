# 📐 Setup & Installation

## Complete Hardware & Software Configuration Guide

This guide covers everything you need to set up and install Local LLMs on your hardware, from hardware selection to production-ready deployment.

## Table of Contents

- [System Requirements](#system-requirements)
- [Hardware Setup](#hardware-setup)
- [Software Installation](#software-installation)
- [Environment Configuration](#environment-configuration)
- [Verification & Testing](#verification--testing)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **CPU**: 4-core processor (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 8GB minimum (16GB recommended for optimal performance)
- **Storage**: 50GB SSD (fast I/O is critical)
- **GPU**: Optional but recommended (NVIDIA, AMD, or Apple Metal)
- **Internet**: Stable connection for model downloads

### Recommended Specifications

- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 32GB+ (more for larger models)
- **Storage**: 500GB+ NVMe SSD
- **GPU**: NVIDIA RTX 3080/4080+ or Apple Silicon M1/M2
- **Power**: 500W+ PSU for GPU-equipped systems

## Hardware Setup

### Step 1: Verify Your Hardware

```bash
# Check CPU information
cat /proc/cpuinfo

# Check RAM
free -h

# Check Storage
df -h

# Check GPU (NVIDIA)
nvidia-smi

# Check GPU (AMD)
rocm-smi
```

### Step 2: Enable Hardware Acceleration

#### NVIDIA GPU Setup

```bash
# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install nvidia-driver-535

# Verify installation
nvidia-smi

# Install CUDA Toolkit
wget https://developer.nvidia.com/cuda-12-3-0-download-archive
sudo sh cuda_12.3.0_linux.run
```

#### AMD GPU Setup

```bash
# Install ROCm
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install rocm-dkms
```

#### Apple Silicon Setup

```bash
# Metal is built-in with macOS
# No additional setup required
# Verify Metal support
system_profiler SPDisplaysDataType
```

## Software Installation

### Step 1: Install Python

```bash
# Install Python 3.10+
sudo apt-get install python3.10 python3.10-venv

# Create virtual environment
python3.10 -m venv llm-env
source llm-env/bin/activate
```

### Step 2: Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch (CPU only)
pip install torch torchvision torchaudio

# Install Core libraries
pip install transformers accelerate bitsandbytes
```

### Step 3: Install Ollama

```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull mistral
```

### Step 4: Install Development Tools

```bash
# Install git
sudo apt-get install git

# Clone your repository
git clone <your-repo-url>
cd <your-project>

# Install project dependencies
pip install -r requirements.txt
```

## Environment Configuration

### Create Environment Variables

```bash
# Create .env file
cat > .env << EOF
# Model Configuration
MODEL_NAME=mistral
MODEL_PATH=/models
CONTEXT_LENGTH=4096

# Hardware Configuration
USE_GPU=true
GPU_MEMORY_FRACTION=0.9
MAX_TOKENS=2048

# Server Configuration
API_PORT=8000
API_HOST=0.0.0.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=/logs/llm.log
EOF
```

### Python Configuration Example

```python
import os
from dotenv import load_dotenv

load_dotenv()

config = {
    'model_name': os.getenv('MODEL_NAME', 'mistral'),
    'model_path': os.getenv('MODEL_PATH', '/models'),
    'use_gpu': os.getenv('USE_GPU', 'true').lower() == 'true',
    'gpu_memory_fraction': float(os.getenv('GPU_MEMORY_FRACTION', '0.9')),
    'api_port': int(os.getenv('API_PORT', '8000')),
    'api_host': os.getenv('API_HOST', '0.0.0.0'),
}
```

## Verification & Testing

### Test Python Installation

```python
import sys
import torch
import transformers

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
```

### Test Ollama Installation

```bash
# Test Ollama API
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### Performance Benchmarking

```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Benchmark generation speed
start = time.time()
outputs = model.generate(
    **tokenizer("Hello, how are you?", return_tensors="pt"),
    max_length=100
)
end = time.time()

print(f"Generation time: {end - start:.2f}s")
print(f"Tokens/second: {len(outputs[0]) / (end - start):.2f}")
```

## Troubleshooting

### Common Issues

#### Issue: Out of Memory (OOM)

**Solution**:
- Reduce context length
- Use quantized models (8-bit, 4-bit)
- Enable CPU offloading
- Increase swap memory

#### Issue: GPU Not Detected

**Solution**:
```bash
# Reinstall GPU drivers
sudo apt-get purge nvidia*
sudo apt-get autoremove
# Reinstall following official guidelines
```

#### Issue: Slow Generation Speed

**Solution**:
- Enable GPU acceleration
- Use Flash Attention v2
- Increase batch size
- Use smaller models

#### Issue: Port Already in Use

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Change port in configuration
```

### Getting Help

- Check official documentation
- Search GitHub issues
- Ask community forums
- Review logs for specific errors

---

**Last Updated**: December 2024
**Status**: Active Development
