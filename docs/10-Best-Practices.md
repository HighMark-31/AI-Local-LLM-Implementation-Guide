# 💯 Best Practices

## Production-Ready LLM Implementation Guide

Essential practices and patterns for building robust, scalable, and maintainable Local LLM systems. Learn from real-world experience to avoid common pitfalls.

## Table of Contents

- [Code Quality](#code-quality)
- [Performance Optimization](#performance-optimization)
- [Security & Privacy](#security--privacy)
- [Monitoring & Observability](#monitoring--observability)
- [Error Handling](#error-handling)
- [Documentation](#documentation)
- [Testing](#testing)

## Code Quality

### Use Type Hints

```python
from typing import Optional, List, Dict
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7
) -> str:
    """Generate text from prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=max_length, temperature=temperature)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Follow PEP 8

```python
# Good
max_token_count = 2048
model_name = "mistral-7b"

# Bad
maxTokenCount = 2048
modelName = "mistral-7b"
```

### Use Meaningful Names

```python
# Good
def calculate_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate perplexity score."""
    pass

# Bad
def calc_ppl(l: torch.Tensor, t: torch.Tensor) -> float:
    pass
```

## Performance Optimization

### Memory Management

```python
import gc
import torch

# Clear cache regularly
torch.cuda.empty_cache()
gc.collect()

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(input_ids)
```

### Batch Processing

```python
def batch_generate(prompts: List[str], batch_size: int = 4) -> List[str]:
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        # Process batch
        batch_results = [generate(p) for p in batch]
        results.extend(batch_results)
    return results
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_embedding(text: str) -> List[float]:
    """Get cached embeddings."""
    return embedding_model.encode(text)
```

## Security & Privacy

### Input Validation

```python
import re

def validate_prompt(prompt: str, max_length: int = 10000) -> bool:
    """Validate user input."""
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be string")
    
    if len(prompt) > max_length:
        raise ValueError(f"Prompt too long: {len(prompt)} > {max_length}")
    
    # Check for injection attempts
    if re.search(r'<script|javascript:', prompt, re.IGNORECASE):
        raise ValueError("Invalid characters in prompt")
    
    return True
```

### Rate Limiting

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> bool:
        now = datetime.now()
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if now - req_time < self.window
        ]
        
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        self.requests[user_id].append(now)
        return True
```

### Sanitization

```python
import html

def sanitize_output(text: str) -> str:
    """Remove potentially harmful content."""
    # Escape HTML
    text = html.escape(text)
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    return text
```

## Monitoring & Observability

### Structured Logging

```python
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def log_generation(prompt: str, response: str, latency: float, status: str):
    """Log with structured format."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event': 'generation',
        'prompt_length': len(prompt),
        'response_length': len(response),
        'latency_ms': latency * 1000,
        'status': status
    }
    logger.info(json.dumps(log_entry))
```

### Metrics Tracking

```python
from prometheus_client import Counter, Histogram

generation_time = Histogram('llm_generation_seconds', 'Time to generate')
generation_errors = Counter('llm_generation_errors', 'Generation errors')

def track_generation(func):
    def wrapper(*args, **kwargs):
        try:
            with generation_time.time():
                return func(*args, **kwargs)
        except Exception as e:
            generation_errors.inc()
            raise
    return wrapper
```

## Error Handling

### Graceful Degradation

```python
def generate_with_fallback(prompt: str, max_retries: int = 3) -> str:
    """Generate with automatic retry and fallback."""
    for attempt in range(max_retries):
        try:
            return generate(prompt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt < max_retries - 1:
                continue
        except Exception as e:
            logger.error(f"Generation failed: {e}")
    
    # Fallback response
    return "I apologize, but I encountered an error processing your request."
```

## Documentation

### API Documentation

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Local LLM API",
    description="Complete guide to Local LLM implementation",
    version="1.0.0"
)

class GenerateRequest(BaseModel):
    """Request body for text generation.
    
    Attributes:
        prompt: The input text to continue
        max_tokens: Maximum tokens to generate (1-2048)
        temperature: Sampling temperature (0.0-2.0)
    """
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    """Generate text based on prompt.
    
    Returns:
        Generated text response
    """
    pass
```

## Testing

### Unit Tests

```python
import pytest

def test_prompt_validation():
    """Test input validation."""
    # Valid prompt
    assert validate_prompt("Hello world") == True
    
    # Too long
    with pytest.raises(ValueError):
        validate_prompt("x" * 20000)
    
    # Invalid characters
    with pytest.raises(ValueError):
        validate_prompt("<script>alert('xss')</script>")
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete generation pipeline."""
    prompt = "What is machine learning?"
    response = generate(prompt, max_length=100)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert len(response) <= 500
```

## Key Takeaways

1. **Always validate inputs**: Never trust user input
2. **Monitor everything**: Logging and metrics are crucial
3. **Plan for failure**: Implement graceful degradation
4. **Document thoroughly**: Make it easy for others to understand your code
5. **Test rigorously**: Unit and integration tests prevent issues
6. **Optimize continuously**: Profile and optimize bottlenecks
7. **Security first**: Treat security as a core feature, not an afterthought

---

**Last Updated**: December 2024
**Status**: Active Development
