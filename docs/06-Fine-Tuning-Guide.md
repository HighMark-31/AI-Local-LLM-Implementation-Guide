# 🔧 Fine-Tuning Guide

## Adapting Pre-trained Models to Your Specific Domain

Fine-tuning allows you to adapt pre-trained models to your specific use cases, domain vocabulary, and style. This guide covers techniques from simple supervised fine-tuning to advanced methods like LoRA and QLoRA.

## Table of Contents

- [Fine-Tuning Fundamentals](#fine-tuning-fundamentals)
- [Data Preparation](#data-preparation)
- [Fine-Tuning Methods](#fine-tuning-methods)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [LoRA & QLoRA](#lora--qlora)
- [Training Configuration](#training-configuration)
- [Evaluation & Validation](#evaluation--validation)

## Fine-Tuning Fundamentals

### When to Fine-Tune

- **Domain Adaptation**: Specialized vocabulary (medical, legal, technical)
- **Style Matching**: Specific tone or format requirements
- **Task Specialization**: Domain-specific tasks not well-covered by base models
- **Data Privacy**: Keep sensitive data local
- **Cost Optimization**: Reduce inference latency for specialized tasks

### Methods Comparison

| Method | Memory | Speed | Quality | Use Case |
|--------|--------|-------|---------|----------|
| Full SFT | High | Slow | Excellent | Large budgets |
| LoRA | Low | Fast | Very Good | Most cases |
| QLoRA | Very Low | Fast | Good | Limited resources |
| Instruction Tuning | High | Slow | Excellent | Task-specific |

## Data Preparation

### Data Format

```json
{
  "instruction": "Translate English to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

### Data Quality Guidelines

- **Size**: 1K-10K examples for LoRA, 10K+ for full SFT
- **Quality**: Clear, correct, representative examples
- **Diversity**: Cover all important variations
- **Balance**: Equal representation of categories

### Data Preparation Script

```python
import json
from datasets import Dataset, DatasetDict

# Load data
with open('training_data.jsonl') as f:
    data = [json.loads(line) for line in f]

# Create dataset
dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.1)

# Save
dataset.save_to_disk('prepared_data')
```

## Fine-Tuning Methods

### 1. Supervised Fine-Tuning (SFT)

Train on instruction-output pairs:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "mistral-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset('json', data_files='train.jsonl')

def tokenize_function(examples):
    outputs = tokenizer(
        examples['text'],
        max_length=2048,
        truncation=True,
    )
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)

trainer.train()
```

## LoRA & QLoRA

### LoRA (Low-Rank Adaptation)

Train only small adapter weights:

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: trainable params: 4194304 || all params: 3733296384 || trainable%: 0.11

trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

### QLoRA (Quantized LoRA)

Combines 4-bit quantization with LoRA:

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
```

## Training Configuration

### Hyperparameters

```python
training_args = TrainingArguments(
    # Learning
    learning_rate=2e-5,          # Start lower for fine-tuning
    num_train_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    
    # Batch size
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    
    # Optimization
    optim="adamw_8bit",
    max_grad_norm=1.0,
    
    # Logging & saving
    logging_steps=10,
    save_steps=100,
    eval_steps=50,
    save_strategy="steps",
    evaluation_strategy="steps",
    
    # Output
    output_dir="./results",
    report_to=["tensorboard"],
)
```

## Evaluation & Validation

### Evaluation Metrics

```python
from evaluate import load
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
)
```

### Testing Fine-Tuned Model

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./results/checkpoint-1000",
    device_map="auto"
)

inputs = tokenizer("Your test prompt", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Inference Optimization

```python
model.eval()
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
    )
```

## Best Practices

1. **Start with LoRA**: More efficient than full fine-tuning
2. **Use gradient checkpointing**: Reduces memory usage
3. **Monitor validation loss**: Prevent overfitting
4. **Data quality > quantity**: Focus on data quality
5. **Regular evaluation**: Test on held-out data
6. **Version control**: Track training runs and configs

---

**Last Updated**: December 2024
**Status**: Active Development
