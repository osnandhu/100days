# RAG Financial Advisor Agent - Deep Dive

## What Does This Agent Do?

The RAG agent answers: **"What should I do about [financial question]?"**

Instead of relying purely on the LLM's general knowledge, it retrieves
specific financial advice from a local knowledge base and uses that to
generate grounded, personalised recommendations.

---

## Core Concepts

### What is RAG (Retrieval-Augmented Generation)?

RAG is a pattern that combines **search** with **generation**:

```
User Question
     ↓
[1. RETRIEVE] → Search a knowledge base for relevant documents
     ↓
[2. AUGMENT]  → Add those documents to the LLM prompt as context
     ↓
[3. GENERATE] → LLM produces an answer grounded in the documents
```

**Why not just ask the LLM directly?**
- LLMs hallucinate (confidently make things up)
- LLMs have stale training data (can't know about 2025 CPF changes)
- LLMs give generic advice (not Singapore-specific)
- RAG gives you traceable sources (you know WHICH document informed the answer)

### What are Embeddings?

Embeddings convert text into numerical vectors (lists of numbers).
Similar texts produce similar vectors — "budget tips" and "saving money"
would be close together in vector space, while "quantum physics" would be far.

```
"budget tips"     → [0.23, -0.15, 0.87, ...]  (1536 numbers)
"saving money"    → [0.21, -0.18, 0.85, ...]  (very close!)
"quantum physics" → [-0.45, 0.92, 0.11, ...]  (very different)
```

### What is FAISS?

Facebook AI Similarity Search. A library that stores embedding vectors
and finds the nearest neighbours blazingly fast.

Think of it as a specialized database:
- Regular DB: `SELECT * FROM docs WHERE title = 'budgeting'`
- FAISS: `Find the 3 documents whose embeddings are closest to this query embedding`

The regular DB needs exact keyword matches. FAISS finds semantically
similar content even if different words are used.

### What is LiteLLM?

A Python library that provides a unified interface to 100+ LLM providers:
- OpenAI, Anthropic, Google, Mistral, Cohere, Ollama, etc.
- Same code, swap providers by changing one string

```python
# OpenAI
litellm.completion(model="openai/gpt-4o-mini", messages=[...])

# Anthropic (same code!)
litellm.completion(model="anthropic/claude-sonnet-4-20250514", messages=[...])

# Local Ollama (still same code!)
litellm.completion(model="ollama/llama3", messages=[...])
```

---

## How the Code Works (Step by Step)

### Step 1: Build the Knowledge Base

At startup, we load financial documents from `data/financial_knowledge.json`:

```python
documents = [
    {"title": "The 50/30/20 Budget Rule", "content": "Allocate 50%..."},
    {"title": "Emergency Fund Basics", "content": "An emergency fund..."},
    ...
]
```

Each document is a self-contained piece of financial advice. There are
10 documents covering budgeting, savings, inflation, debt, and spending.

### Step 2: Create Embeddings

Each document gets converted to a vector using LiteLLM:

```python
texts = ["50/30/20 Budget Rule: Allocate 50%...", ...]
response = litellm.embedding(model="text-embedding-ada-002", input=texts)
vectors = [item["embedding"] for item in response.data]
```

Each vector has 1536 dimensions (for ada-002). That's 1536 numbers that
capture the "meaning" of the text.

### Step 3: Build FAISS Index

```python
index = faiss.IndexFlatL2(dimension=1536)
index.add(numpy_array_of_vectors)
```

`IndexFlatL2` = exact search using L2 (Euclidean) distance. For our 10
documents this is instant. For millions of documents, you'd use
`IndexIVFFlat` (approximate but faster).

### Step 4: Handle a Query

When a user asks "How do I save with high inflation?":

```python
# 1. Embed the query
query_vector = litellm.embedding(model="...", input=["How do I save..."])

# 2. Search FAISS for nearest documents
distances, indices = index.search(query_vector, k=3)  # top 3

# 3. Build RAG prompt
prompt = f"""
KNOWLEDGE BASE:
[Inflation Impact on Savings]: If inflation is 4%...
[Emergency Fund Basics]: An emergency fund should...
[Investing During High Inflation]: During high inflation...

QUESTION: How do I save with high inflation?
"""

# 4. Generate answer
response = litellm.completion(model="openai/gpt-4o-mini", messages=[...])
```

The LLM now answers based on OUR documents, not its general training data.

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| FAISS (local) over Pinecone/Weaviate (cloud) | No external service dependency; works offline |
| 10 curated documents | Small, high-quality knowledge base > thousands of scraped articles |
| Top-3 retrieval | More than 3 adds noise; fewer might miss relevant context |
| Temperature 0.3 | Low creativity = more factual, less hallucination |
| Fallback to random vectors | If embedding API fails, app still works (degraded but not crashed) |

---

## Common Pitfalls

1. **Embedding model mismatch** — If you build the index with OpenAI embeddings
   but query with Ollama embeddings, similarity search won't work. Always use
   the same model for indexing AND querying.

2. **Document chunking** — Our documents are short (1 paragraph). Long documents
   should be split into chunks of 200-500 tokens. Otherwise the embedding
   averages over too much content and loses specificity.

3. **Stale knowledge base** — If CPF rules change in 2026 and we don't update
   the JSON, the agent gives outdated advice. Build a process for regular updates.

4. **Over-reliance on top-k** — If the question is completely unrelated to our
   knowledge base, FAISS still returns the "least distant" documents, which
   might be irrelevant. The system prompt tells the LLM to say so honestly.

---

## Practice Exercises (Weekend)

### Exercise 1: Add More Documents
Add 5 more documents to `financial_knowledge.json` about topics you care
about (e.g., HDB buying, investment-linked policies, robo-advisors).
Rebuild the index and test if they get retrieved for relevant queries.

### Exercise 2: Try Different Embedding Models
Swap `text-embedding-ada-002` for `text-embedding-3-small` or an Ollama
embedding model. Compare retrieval quality — do the same documents get
retrieved for the same queries?

### Exercise 3: Measure Retrieval Quality
Create 10 test questions where you know which document SHOULD be retrieved.
Run them all and compute accuracy: how often is the correct document in
the top-3? This is your retrieval recall metric.

### Exercise 4: Add a Feedback Loop
When the user says the advice was helpful, boost that document's relevance.
Implement a simple re-ranking: if a document has been marked helpful 5 times
for similar queries, prioritise it in results.
