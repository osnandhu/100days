# Observability with Langfuse

## What is Langfuse?

Langfuse is an **open-source LLM observability platform**. It answers:
- Which agent made which LLM call?
- How long did each tool take?
- How many tokens did each query cost?
- Where is the bottleneck in my pipeline?

Think of it as **Chrome DevTools for your AI agents**.

---

## How It Integrates (Zero Code Changes)

```
User Query
    ↓
FastAPI → Crew Manager → CrewAI Agent → LiteLLM → OpenAI/Anthropic
                                            ↓
                                      Langfuse Callback
                                            ↓
                                    ┌─────────────────┐
                                    │ Langfuse Cloud   │
                                    │ or Self-Hosted   │
                                    │                  │
                                    │ Traces           │
                                    │ Generations      │
                                    │ Costs            │
                                    │ Latency          │
                                    └─────────────────┘
```

LiteLLM has a **native Langfuse callback**. When you set the env vars,
every `litellm.completion()` and `litellm.embedding()` call automatically
sends telemetry. CrewAI uses LiteLLM under the hood, so all agent calls
are captured without touching agent code.

---

## Setup (5 minutes)

### Step 1: Create a Langfuse Account

Go to **https://cloud.langfuse.com** → Sign up (free tier is generous).

### Step 2: Get API Keys

Dashboard → Settings → API Keys → Create New

You'll get:
- `LANGFUSE_PUBLIC_KEY` (starts with `pk-lf-`)
- `LANGFUSE_SECRET_KEY` (starts with `sk-lf-`)

### Step 3: Set Environment Variables

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

### Step 4: Run the App

```bash
uvicorn main_api:app --reload
```

That's it. Open Langfuse dashboard and you'll see traces appear.

---

## What You See in the Dashboard

### Traces View
Each user query creates a **trace**:
```
Trace: "Can I save $1000/month with rising food inflation?"
├── Span: Orchestrator routing (2ms)
├── Generation: Inflation Agent LLM call
│   ├── Input: "Fetch CPI data and forecast..."
│   ├── Output: "Singapore inflation predicted at 2.6%..."
│   ├── Tokens: 1,247 (in: 892, out: 355)
│   ├── Latency: 3.2s
│   └── Cost: $0.0018
├── Generation: RAG Agent - Embedding call
│   ├── Model: text-embedding-ada-002
│   ├── Tokens: 48
│   └── Cost: $0.000005
└── Generation: RAG Agent LLM call
    ├── Input: "Based on these documents..."
    ├── Output: "Here are strategies..."
    ├── Tokens: 1,890
    ├── Latency: 4.1s
    └── Cost: $0.0025
```

### Metrics You Can Track
- **Latency per agent**: Which agent is slowest?
- **Token usage per query**: How much does each question cost?
- **Error rate**: Which agent fails most often?
- **Cost over time**: Daily/weekly spend trending

---

## Self-Hosting (Optional)

If you don't want data on Langfuse's cloud:

```bash
# Docker Compose (Langfuse + PostgreSQL)
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Then set `LANGFUSE_HOST=http://localhost:3000`.

---

## Why Langfuse Over Alternatives

| Feature | Langfuse | LangSmith | AgentOps |
|---------|----------|-----------|----------|
| Open source | Yes | No | No |
| Self-hostable | Yes | No | No |
| Free tier | Generous | Limited | Limited |
| LiteLLM native | Yes | Plugin | Plugin |
| CrewAI support | Via LiteLLM | Direct | Direct |
| Cost tracking | Yes | Yes | Yes |

Langfuse wins for our stack because LiteLLM integration is zero-config.

---

## Presentation Talking Points

When demoing observability:
1. "Every LLM call is traced automatically — no code changes needed"
2. "I can see exactly which agent costs the most per query"
3. "If the inflation agent is slow, I can see it's the Prophet tool, not the LLM"
4. "Open-source, so we can self-host for production data privacy"
