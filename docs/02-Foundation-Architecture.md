# 📚 Foundation & Architecture

## LLM Core Concepts

### Transformer Architecture

Modern LLMs are built on the **Transformer** architecture, which uses:

- **Attention Mechanisms**: Allows the model to focus on relevant parts of input
- **Multi-Head Attention**: Processes information from multiple representation subspaces
- **Feed-Forward Networks**: Provides non-linear transformations
- **Layer Normalization**: Stabilizes training process
- **Positional Encoding**: Captures word order information

### Scaling Laws

Key findings in LLM development:

```
Performance ∝ (Model Size)^a * (Data Size)^b * (Compute)^c

Where a, b, c ≈ 0.07 (empirically derived)
```

Implications:
- Larger models perform better
- More training data improves results
- Compute efficiency critical for local deployment

## Local vs Cloud LLMs

| Aspect | Local LLM | Cloud API |
|--------|-----------|----------|
| **Privacy** | 🟢 Full | 🛖 Limited |
| **Cost** | 💰 One-time | 💰 Per-request |
| **Control** | 🎯 Complete | 🚫 Restricted |
| **Customization** | 🔧 Full | 🚫 Limited |
| **Setup Effort** | 😹 High | 😀 Low |

## Model Architecture Overview

### Encoder-Only (BERT-style)
- Good for: Classification, sentiment analysis
- Example: BERT, RoBERTa
- Trade-off: Can't generate text well

### Decoder-Only (GPT-style)
- Good for: Text generation, conversation
- Example: GPT-2, Llama, Mistral
- Trade-off: Slower inference

### Encoder-Decoder (T5-style)
- Good for: Translation, summarization
- Example: T5, BART
- Trade-off: More complex architecture

## Quantization Fundamentals

**What is Quantization?**

Reducing precision of model weights from 32-bit float to 8-bit or 4-bit integer.

**Benefits**:
- 4x-8x smaller model size
- 2x-4x faster inference
- Reduced VRAM requirements

**Types**:
- **INT8**: 8-bit integer (minimal quality loss)
- **INT4**: 4-bit integer (noticeable quality loss, best compression)
- **GGUF**: GPU-friendly quantization format

## Inference Optimization

### Batching
Processing multiple inputs simultaneously to maximize GPU utilization.

### KV-Cache
Stores computed key-value pairs to avoid recomputation.

### Token Streaming
Returns tokens as they're generated (faster perceived latency).

---

**Next**: [Tools & Frameworks](03-Tools-Frameworks.md)
