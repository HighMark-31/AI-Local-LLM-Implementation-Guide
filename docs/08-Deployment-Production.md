# 📄 Deployment & Production

## Getting Your LLM to Production

This guide covers deploying Local LLMs to production environments with focus on reliability, scalability, and cost-efficiency. Learn containerization, API deployment, monitoring, and optimization strategies.

## Table of Contents

- [Deployment Strategies](#deployment-strategies)
- [Containerization](#containerization)
- [API Servers](#api-servers)
- [Load Balancing](#load-balancing)
- [Monitoring & Logging](#monitoring--logging)
- [Optimization & Scaling](#optimization--scaling)

## Deployment Strategies

### Single Server

Best for: Development, small teams, low latency requirements

```bash
# Direct deployment
git clone <repo>
cd project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py --host 0.0.0.0 --port 8000
```

### Containerized (Docker)

Best for: Microservices, scaling, consistency across environments

```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py", "--host", "0.0.0.0"]
```

### Kubernetes

Best for: Enterprise, high availability, auto-scaling

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: llm-api
        image: llm-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
          limits:
            memory: "32Gi"
            cpu: "8"
```

## Containerization

### Docker Best Practices

```dockerfile
# Multi-stage build for smaller final image
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "app.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  llm-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=mistral-7b
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## API Servers

### FastAPI + Uvicorn

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="LLM API", version="1.0.0")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=request.max_tokens,
            temperature=request.temperature
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

## Load Balancing

### Nginx Configuration

```nginx
upstream llm_backend {
    least_conn;
    server localhost:8001 weight=3;
    server localhost:8002 weight=3;
    server localhost:8003 weight=1;
}

server {
    listen 80;
    server_name api.example.com;

    location /generate {
        proxy_pass http://llm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
    }

    location /health {
        access_log off;
        proxy_pass http://llm_backend;
    }
}
```

## Monitoring & Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest
import time

request_count = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['method', 'status']
)

request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds'
)

@app.post("/generate")
async def generate(request: GenerateRequest):
    start = time.time()
    try:
        # ... generation logic ...
        request_count.labels(method='generate', status='success').inc()
    except Exception as e:
        request_count.labels(method='generate', status='error').inc()
    finally:
        request_duration.observe(time.time() - start)

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest())
```

### Logging

```python
import logging
import json
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# JSON logging for structured logs
jsonFormater = jsonlogger.JsonFormatter()
logHandler = logging.StreamHandler()
logHandler.setFormatter(jsonFormater)
logger.addHandler(logHandler)

logger.info("Request", extra={
    "prompt": request.prompt[:100],
    "tokens": request.max_tokens,
    "duration": elapsed_time
})
```

## Optimization & Scaling

### Model Quantization for Deployment

```python
from transformers import AutoModelForCausalLM
import torch

# 4-bit quantization for deployment
model = AutoModelForCausalLM.from_pretrained(
    "mistral-7b",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    device_map="auto"
)

# Save for inference
model.save_pretrained("./mistral-7b-q4")
```

### Caching Strategies

```python
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379)

@app.post("/generate")
async def generate(request: GenerateRequest):
    # Check cache
    cache_key = f"llm:{request.prompt}:{request.temperature}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Generate and cache
    result = generate_response(request)
    redis_client.setex(cache_key, 3600, json.dumps(result))
    return result
```

### Horizontal Scaling

```bash
# Launch multiple instances
for i in {1..4}; do
    PORT=$((8000 + i)) python app.py --port $PORT &
done

# Use load balancer (nginx) to distribute requests
```

---

**Last Updated**: December 2024
**Status**: Active Development
