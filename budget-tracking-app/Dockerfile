# ---------------------------------------------------------------------------
# Multi-stage Dockerfile for the Smart Budget Tracker
# ---------------------------------------------------------------------------
# WHY MULTI-STAGE:
#   Stage 1 (builder): installs all dependencies including build tools
#   Stage 2 (runtime): copies only what's needed to run
#   Result: smaller final image (no compilers, no pip cache)
#
# HOW TO USE:
#   docker build -t smart-budget-app .
#   docker run -p 8000:8000 -e LLM_API_KEY=sk-... smart-budget-app
#
# Then visit http://localhost:8000/docs for the API explorer.
# ---------------------------------------------------------------------------

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for Prophet (needs C compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# FastAPI default port
EXPOSE 8000

# Environment variables (override at runtime with -e)
ENV DATABASE_URI=sqlite:///budget.db \
    SQL_ECHO=false \
    LLM_MODEL=openai/gpt-4o-mini \
    LLM_API_KEY="" \
    EMBEDDING_MODEL=text-embedding-ada-002

# Start the API server
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
