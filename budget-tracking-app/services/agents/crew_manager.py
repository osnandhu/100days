"""
Crew Manager
=============
The glue layer that creates CrewAI Crews and runs them.

CREWAI CONCEPTS USED HERE:

  Agent   - A specialist with a role, goal, backstory, and tools.
            Think of it as a team member with a specific job description.

  Task    - A specific piece of work assigned to an agent.
            Has a description (what to do) and expected_output (what success looks like).

  Crew    - A team of agents working together on tasks.
            Runs tasks either sequentially or in a managed hierarchy.

  Process - How the crew coordinates:
            SEQUENTIAL = tasks run one after another, output flows forward
            HIERARCHICAL = a manager agent delegates to specialists

  LLM     - CrewAI wraps LiteLLM so you can swap providers without code changes.

WHY SEQUENTIAL PROCESS:
  Simpler to debug. Each agent's output feeds into the next agent's context.
  For our use case (analyze → detect → advise), sequential makes sense
  because the advisor benefits from knowing the inflation forecast and
  spending anomalies.

  Hierarchical is better when agents are truly independent and you want
  a manager to decide dynamically. We keep it sequential for clarity.
"""

import json
import logging
import os
from typing import Optional

try:
    from crewai import Crew, LLM, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

from services.agents.inflation_agent import create_inflation_agent, create_inflation_task
from services.agents.orchestrator import (
    AgentResult,
    AgentType,
    OrchestratorResponse,
    classify_query,
    synthesise_results,
)
from services.agents.rag_agent import create_rag_agent, create_rag_task, initialize_knowledge_base
from services.agents.spending_agent import create_spending_agent, create_spending_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Langfuse Observability
# ---------------------------------------------------------------------------
# Langfuse traces every LLM call, tool invocation, and agent step.
# Set these env vars to enable:
#   LANGFUSE_PUBLIC_KEY=pk-lf-...
#   LANGFUSE_SECRET_KEY=sk-lf-...
#   LANGFUSE_HOST=https://cloud.langfuse.com  (or self-hosted URL)
#
# HOW IT WORKS:
#   LiteLLM has a native Langfuse callback. When enabled, every call
#   to litellm.completion() or litellm.embedding() automatically sends
#   a trace to Langfuse — no code changes in the agent files needed.
#   CrewAI uses LiteLLM under the hood, so all agent LLM calls are
#   captured automatically.
#
# WHAT YOU SEE IN THE LANGFUSE DASHBOARD:
#   - Each user query → a "trace" with a unique ID
#   - Each agent's LLM call → a "generation" (input, output, tokens, latency)
#   - Each tool call → a "span" showing what the agent invoked
#   - Cost tracking per query (which agent costs the most?)
#   - Latency breakdown (which tool is the bottleneck?)
# ---------------------------------------------------------------------------

LANGFUSE_ENABLED = False


def _init_langfuse():
    """
    Initialize Langfuse tracing via LiteLLM callbacks.

    This is the zero-code-change approach: we tell LiteLLM to send
    telemetry to Langfuse, and since CrewAI uses LiteLLM internally,
    all agent calls are traced automatically.
    """
    global LANGFUSE_ENABLED

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.info("Langfuse not configured (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)")
        return

    try:
        import litellm
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]

        os.environ.setdefault("LANGFUSE_HOST", host)

        LANGFUSE_ENABLED = True
        logger.info(f"Langfuse tracing enabled → {host}")
    except ImportError:
        logger.warning("litellm not installed, Langfuse tracing disabled")


def _get_llm() -> LLM:
    """
    Create a CrewAI LLM instance using LiteLLM under the hood.

    LiteLLM format: "provider/model" (e.g., "openai/gpt-4o-mini").
    Set LLM_MODEL and LLM_API_KEY as environment variables to switch
    providers without changing code.

    Examples:
      LLM_MODEL=openai/gpt-4o-mini     LLM_API_KEY=sk-...
      LLM_MODEL=anthropic/claude-sonnet-4-20250514  LLM_API_KEY=sk-ant-...
      LLM_MODEL=ollama/llama3           (no API key needed for local)
    """
    model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    api_key = os.getenv("LLM_API_KEY", "")

    return LLM(model=model, api_key=api_key) if api_key else LLM(model=model)


# ---------------------------------------------------------------------------
# Individual Agent Runners
# ---------------------------------------------------------------------------
# These run a single agent for the dedicated API endpoints
# (POST /agent/forecast, /agent/anomalies, /agent/advice).
# ---------------------------------------------------------------------------


def run_inflation_forecast(context: str = "") -> str:
    """Run only the inflation agent. Used by POST /agent/forecast."""
    llm = _get_llm()
    agent = create_inflation_agent(llm)
    task = create_inflation_task(agent, context)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return result.raw


def run_spending_analysis(user_id: int, context: str = "") -> str:
    """Run only the spending agent. Used by POST /agent/anomalies."""
    llm = _get_llm()
    agent = create_spending_agent(llm)
    task = create_spending_task(agent, user_id, context)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return result.raw


def run_rag_advice(question: str, user_context: str = "") -> str:
    """Run only the RAG agent. Used by POST /agent/advice."""
    llm = _get_llm()
    agent = create_rag_agent(llm)
    task = create_rag_task(agent, question, user_context)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return result.raw


# ---------------------------------------------------------------------------
# Full Multi-Agent Analysis
# ---------------------------------------------------------------------------
# This is the main entry point: POST /agent/analyze
# It uses the orchestrator to classify the query, runs the relevant agents,
# and synthesises their outputs.
# ---------------------------------------------------------------------------


def run_full_analysis(
    query: str,
    user_id: int = 1,
    user_context: str = "",
) -> OrchestratorResponse:
    """
    Full multi-agent analysis pipeline.

    FLOW:
    1. Orchestrator classifies the query → decides which agents to call
    2. For each selected agent:
       a. Create the agent with tools
       b. Create a task with the user's query as context
       c. Run it as a mini Crew
       d. Collect the result
    3. Orchestrator synthesises all results into one response

    WHY SEPARATE CREWS (instead of one big Crew):
    Running each agent as its own single-agent Crew gives us:
    - Isolation: one agent's failure doesn't crash others
    - Flexibility: we only run agents the query actually needs
    - Clarity: each agent's output is clearly attributed
    """
    # Step 1: Route the query
    routing = classify_query(query)
    logger.info(f"Routing decision: {routing.reasoning}")

    llm = _get_llm()
    agent_results = []

    # Step 2: Run each selected agent
    for agent_type in routing.agents_to_call:
        try:
            if agent_type == AgentType.INFLATION:
                agent = create_inflation_agent(llm)
                task = create_inflation_task(agent, query)
            elif agent_type == AgentType.SPENDING:
                agent = create_spending_agent(llm)
                task = create_spending_task(agent, user_id, query)
            elif agent_type == AgentType.RAG:
                agent = create_rag_agent(llm)
                task = create_rag_task(agent, query, user_context)
            else:
                continue

            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            agent_results.append(
                AgentResult(
                    agent_type=agent_type,
                    raw_output=result.raw,
                    success=True,
                )
            )

        except Exception as e:
            logger.error(f"Agent {agent_type.value} failed: {e}")
            agent_results.append(
                AgentResult(
                    agent_type=agent_type,
                    raw_output="",
                    success=False,
                    error=str(e),
                )
            )

    # Step 3: Synthesise results
    return synthesise_results(query, agent_results)


def initialize() -> None:
    """
    Call once at application startup.
    Sets up Langfuse tracing and pre-builds the FAISS index.
    """
    _init_langfuse()

    try:
        initialize_knowledge_base()
        logger.info("Knowledge base initialised successfully")
    except Exception as e:
        logger.warning(f"Knowledge base init failed (non-fatal): {e}")
