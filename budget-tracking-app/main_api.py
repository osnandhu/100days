"""
FastAPI Application
===================
REST API that exposes the multi-agent system over HTTP.

ENDPOINTS:
  POST /agent/analyze   - Full analysis (orchestrator routes to all relevant agents)
  POST /agent/forecast  - Inflation forecast only
  POST /agent/anomalies - Spending anomaly detection only
  POST /agent/advice    - RAG financial advice only
  GET  /health          - Health check

WHY FASTAPI:
  - Automatic OpenAPI/Swagger docs at /docs (free API documentation)
  - Pydantic models for request validation (catches bad input before your code runs)
  - Async support (can handle concurrent requests efficiently)
  - Type hints everywhere = your IDE helps you write correct code

HOW TO RUN:
  uvicorn main_api:app --reload --port 8000
  Then visit http://localhost:8000/docs for the interactive API explorer.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path so imports work
sys.path.insert(0, ".")

from database import init_db
from services.agents.crew_manager import (
    initialize,
    run_full_analysis,
    run_inflation_forecast,
    run_rag_advice,
    run_spending_analysis,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: runs once at startup and shutdown
# ---------------------------------------------------------------------------
# FastAPI's lifespan replaces the old @app.on_event("startup") pattern.
# Everything before `yield` runs at startup; everything after runs at shutdown.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initialising database and knowledge base...")
    init_db()
    initialize()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Smart Budget Tracker - Multi-Agent API",
    description=(
        "AI-powered budget analysis using CrewAI agents for inflation forecasting, "
        "spending anomaly detection, and personalised financial advice."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------
# Pydantic models define the shape of request bodies and responses.
# FastAPI uses these to:
#   1. Validate incoming JSON automatically (returns 422 if invalid)
#   2. Generate OpenAPI docs with example values
#   3. Provide IDE autocompletion
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language question about your finances",
        examples=["Can I save $1000/month with rising food inflation?"],
    )
    user_id: int = Field(default=1, description="User ID for transaction lookup")
    user_context: str = Field(
        default="",
        description="Additional context about your financial situation",
        examples=["I earn $5000/month and spend $500 on groceries"],
    )


class ForecastRequest(BaseModel):
    context: str = Field(
        default="",
        description="Additional context for the forecast",
        examples=["Focus on food and transport inflation"],
    )


class AnomalyRequest(BaseModel):
    user_id: int = Field(..., description="User ID to analyze spending for")
    context: str = Field(
        default="",
        description="Additional context",
        examples=["I recently moved house so some spending may look unusual"],
    )


class AdviceRequest(BaseModel):
    question: str = Field(
        ...,
        description="Your financial question",
        examples=["How can I build an emergency fund on a $4000 salary?"],
    )
    user_context: str = Field(
        default="",
        description="Your financial situation",
        examples=["Single, 28, renting in Singapore, no debt"],
    )


class AgentResponse(BaseModel):
    status: str
    result: str
    agents_used: list = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Simple health check — use this to verify the API is running."""
    return {"status": "healthy", "service": "smart-budget-agent-api"}


@app.post("/agent/analyze", response_model=AgentResponse)
def full_analysis(request: AnalyzeRequest):
    """
    Full multi-agent analysis.

    The orchestrator analyzes your query, decides which specialist agents
    to call (inflation, spending, advice), runs them, and synthesises
    their outputs into a comprehensive response.
    """
    try:
        response = run_full_analysis(
            query=request.query,
            user_id=request.user_id,
            user_context=request.user_context,
        )
        return AgentResponse(
            status="success",
            result=response.synthesised_response,
            agents_used=response.agents_used,
        )
    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/forecast", response_model=AgentResponse)
def inflation_forecast(request: ForecastRequest):
    """
    Inflation forecast only.

    Fetches Singapore CPI data from the World Bank API and uses
    Prophet to forecast 6 months ahead with confidence intervals.
    """
    try:
        result = run_inflation_forecast(context=request.context)
        return AgentResponse(
            status="success",
            result=result,
            agents_used=["inflation"],
        )
    except Exception as e:
        logger.error(f"Inflation forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/anomalies", response_model=AgentResponse)
def spending_anomalies(request: AnomalyRequest):
    """
    Spending anomaly detection only.

    Fetches recent transactions from the database and runs
    Z-score + Isolation Forest analysis to detect unusual spending.
    """
    try:
        result = run_spending_analysis(
            user_id=request.user_id,
            context=request.context,
        )
        return AgentResponse(
            status="success",
            result=result,
            agents_used=["spending"],
        )
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/advice", response_model=AgentResponse)
def financial_advice(request: AdviceRequest):
    """
    RAG-powered financial advice.

    Searches the financial knowledge base using FAISS, retrieves
    relevant documents, and generates personalised advice via LiteLLM.
    """
    try:
        result = run_rag_advice(
            question=request.question,
            user_context=request.user_context,
        )
        return AgentResponse(
            status="success",
            result=result,
            agents_used=["rag"],
        )
    except Exception as e:
        logger.error(f"RAG advice failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
