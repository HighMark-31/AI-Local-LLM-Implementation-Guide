# 🔍 RAG Implementation

## Retrieval-Augmented Generation for Enhanced Context and Knowledge

Retrieval-Augmented Generation (RAG) combines document retrieval with language models to provide accurate, contextual responses based on your specific knowledge base. This guide covers implementation from basics to production-ready systems.

## Table of Contents

- [RAG Fundamentals](#rag-fundamentals)
- [System Architecture](#system-architecture)
- [Vector Embeddings](#vector-embeddings)
- [Document Retrieval](#document-retrieval)
- [Implementation](#implementation)
- [Advanced Patterns](#advanced-patterns)
- [Evaluation & Optimization](#evaluation--optimization)

## RAG Fundamentals

### Why RAG?

- **Reduced Hallucinations**: Ground responses in actual documents
- **Up-to-date Knowledge**: Use current documents without retraining
- **Transparent Sources**: Show document sources for citations
- **Domain Specific**: Leverage proprietary knowledge bases
- **Cost Effective**: No fine-tuning required

### RAG Pipeline

```
User Query
    |
    v
[Embedding Model]
    |
    v
[Vector Search]
    |
    v
[Retrieved Documents] + [Original Query]
    |
    v
[LLM]
    |
    v
[Final Response]
```

## System Architecture

### Components

1. **Document Processor**: Ingests and chunks documents
2. **Embedding Model**: Converts text to vectors
3. **Vector Database**: Stores and retrieves embeddings
4. **Retriever**: Finds relevant documents
5. **Language Model**: Generates responses
6. **Ranker**: Scores document relevance

## Vector Embeddings

### Popular Embedding Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | General purpose |
| all-mpnet-base-v2 | 768 | Fast | Excellent | Best quality |
| BGE-base-en-v1.5 | 768 | Fast | Excellent | Dense retrieval |
| bge-large-en | 1024 | Medium | Outstanding | Complex queries |

### Implementation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Embed documents
documents = [
    "Python is a programming language",
    "RAG improves LLM responses",
    "Vector databases store embeddings"
]

embeddings = model.encode(documents)
print(f"Shape: {embeddings.shape}")  # (3, 768)

# Embed query
query = "What is RAG?"
query_embedding = model.encode(query)

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], embeddings)[0]
print(f"Similarities: {similarities}")
```

## Document Retrieval

### Vector Database Options

```python
# Using FAISS (lightweight)
import faiss

index = faiss.IndexFlatL2(768)
index.add(embeddings.astype('float32'))

# Search
k = 3
distances, indices = index.search(query_embedding.reshape(1, -1), k)
print(f"Top {k} documents: {indices}")

# Using Chroma (vector db)
from chromadb import Client

client = Client()
collection = client.create_collection(name="documents")

# Add documents
collection.add(
    ids=["1", "2", "3"],
    embeddings=embeddings.tolist(),
    documents=documents
)

# Query
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)
```

## Implementation

### Complete RAG Pipeline

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load and process documents
loader = PDFLoader("document.pdf")
documents = loader.load()

# 2. Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)

# 4. Store in vector database
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# 5. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. Setup LLM
model_name = "mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 7. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8. Query
result = qa_chain({"query": "What is the main topic?"})
print(f"Answer: {result['result']}")
print(f"Sources: {result['source_documents']}")
```

## Advanced Patterns

### 1. Multi-stage Retrieval

```python
# BM25 (sparse) + Vector (dense) retrieval
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi([doc.split() for doc in documents])
vector_results = vectorstore.search(query, k=5)
bm25_results = bm25.get_top_n(query.split(), documents, n=5)

# Combine and rerank
all_results = set(vector_results + bm25_results)
```

### 2. Query Expansion

```python
# Generate multiple queries from original
query_expansion_prompt = f"Generate 3 variations of: {query}"
expanded_queries = model.generate(query_expansion_prompt)

# Retrieve with all variations
all_docs = []
for expanded_query in expanded_queries:
    docs = retriever.get_relevant_documents(expanded_query)
    all_docs.extend(docs)
```

### 3. Hierarchical Retrieval

```python
# First retrieve summaries, then full documents
summary_index = create_index_from_summaries(documents)
relevant_summaries = summary_index.search(query)

# Then retrieve full documents
full_docs = [doc for summary in relevant_summaries 
             for doc in get_full_documents(summary)]
```

## Evaluation & Optimization

### Evaluation Metrics

```python
from ragas.metrics import (
    context_relevance,
    faithfulness,
    answer_relevancy
)

# Evaluate RAG system
scores = ragas_score(
    dataset,
    llm=model,
    embeddings=embeddings,
    metrics=[context_relevance, faithfulness, answer_relevancy]
)

print(f"Context Relevance: {scores['context_relevance']}")
print(f"Faithfulness: {scores['faithfulness']}")
print(f"Answer Relevancy: {scores['answer_relevancy']}")
```

### Optimization Tips

1. **Chunk Size**: Balance between context and irrelevance (500-2000 tokens)
2. **Overlap**: Use 10-20% overlap for context continuity
3. **Embedding Model**: Choose based on domain and quality requirements
4. **Retrieval Count**: Start with 3-5, increase if needed
5. **Reranking**: Use cross-encoders to rerank retrieved documents
6. **Caching**: Cache embeddings and retrieved documents

### Performance Optimization

```python
# Use GPU for embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2",
    model_kwargs={"device": "cuda"}
)

# Use batch processing
embedding_batch = model.encode(documents, batch_size=32)

# Use approximate nearest neighbor search
index = faiss.IndexIVFFlat(
    faiss.IndexFlatL2(768), 768, 100
)
```

---

**Last Updated**: December 2025
**Status**: Active Development
