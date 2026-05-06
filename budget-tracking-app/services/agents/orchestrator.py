"""
Orchestrator Agent
==================
Routes user queries to the appropriate specialist agents and synthesises
their responses into a single coherent answer.

HOW ROUTING WORKS:
  The orchestrator doesn't use an LLM to decide routing (too slow, too
  expensive for a simple decision). Instead, it uses keyword matching:
    - "inflation" / "forecast" / "prices" → Inflation Agent
    - "spending" / "anomaly" / "unusual"  → Spending Agent
    - "advice" / "save" / "budget" / "recommend" → RAG Agent
    - Ambiguous queries → run ALL agents for comprehensive analysis

  This is intentionally simple. In production, you might use a lightweight
  classifier or the LLM itself. But for learning: start simple, add
  complexity only when simple breaks.

SYNTHESIS:
  After agents return their results, the orchestrator combines them into
  a unified response. If multiple agents ran, it weaves their findings
  together (e.g., inflation forecast + spending data + savings advice).
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Which specialist agent to invoke."""
    INFLATION = "inflation"
    SPENDING = "spending"
    RAG = "rag"


# ---------------------------------------------------------------------------
# Query Classifier
# ---------------------------------------------------------------------------
# Simple keyword-based routing. Each keyword maps to one or more agents.
# If a query matches multiple agents, all are called.
# ---------------------------------------------------------------------------

ROUTING_RULES = {
    AgentType.INFLATION: [
        "inflation", "cpi", "forecast", "prices", "price rise",
        "cost of living", "economic", "predict",
    ],
    AgentType.SPENDING: [
        "spending", "anomaly", "unusual", "transaction", "expense",
        "overspend", "category", "pattern", "detect",
    ],
    AgentType.RAG: [
        "advice", "save", "saving", "budget", "recommend", "tip",
        "strategy", "how to", "should i", "can i", "help me",
        "plan", "invest", "debt", "emergency fund",
    ],
}


@dataclass
class RoutingDecision:
    """Result of the query classifier."""
    agents_to_call: List[AgentType]
    reasoning: str
    original_query: str


@dataclass
class AgentResult:
    """Output from a single agent execution."""
    agent_type: AgentType
    raw_output: str
    success: bool
    error: str = ""


@dataclass
class OrchestratorResponse:
    """Final synthesised response from all agents."""
    query: str
    agents_used: List[str]
    individual_results: List[dict] = field(default_factory=list)
    synthesised_response: str = ""


def classify_query(query: str) -> RoutingDecision:
    """
    Determine which agents should handle a query.

    The algorithm: check the query against keyword lists for each agent.
    If no keywords match, default to RAG (it's the most general-purpose).
    If the query is complex (matches 2+ agents), call all matched agents.
    """
    query_lower = query.lower()
    matched_agents = []
    match_reasons = []

    for agent_type, keywords in ROUTING_RULES.items():
        matching_keywords = [kw for kw in keywords if kw in query_lower]
        if matching_keywords:
            matched_agents.append(agent_type)
            match_reasons.append(
                f"{agent_type.value}: matched [{', '.join(matching_keywords)}]"
            )

    # Default to RAG if nothing matches — it can handle open-ended questions
    if not matched_agents:
        matched_agents = [AgentType.RAG]
        match_reasons = ["No specific keywords matched → defaulting to RAG advisor"]

    reasoning = "; ".join(match_reasons)
    logger.info(f"Query routed to: {[a.value for a in matched_agents]} ({reasoning})")

    return RoutingDecision(
        agents_to_call=matched_agents,
        reasoning=reasoning,
        original_query=query,
    )


def synthesise_results(
    query: str,
    results: List[AgentResult],
) -> OrchestratorResponse:
    """
    Combine outputs from multiple agents into a coherent response.

    WHY NOT USE AN LLM FOR SYNTHESIS:
    We could, but it adds latency and cost. For a learning project, string
    formatting is transparent and debuggable. In production, you'd use an
    LLM here for more natural language.
    """
    response = OrchestratorResponse(
        query=query,
        agents_used=[r.agent_type.value for r in results],
    )

    sections = []

    for result in results:
        if not result.success:
            sections.append(
                f"**{result.agent_type.value.title()} Analysis**: "
                f"Unable to complete — {result.error}"
            )
            response.individual_results.append(
                {"agent": result.agent_type.value, "status": "error", "error": result.error}
            )
            continue

        response.individual_results.append(
            {"agent": result.agent_type.value, "status": "success", "output": result.raw_output}
        )

        if result.agent_type == AgentType.INFLATION:
            sections.append(f"**Inflation Forecast**:\n{result.raw_output}")
        elif result.agent_type == AgentType.SPENDING:
            sections.append(f"**Spending Analysis**:\n{result.raw_output}")
        elif result.agent_type == AgentType.RAG:
            sections.append(f"**Financial Advice**:\n{result.raw_output}")

    response.synthesised_response = "\n\n---\n\n".join(sections)

    if len(results) > 1:
        response.synthesised_response += (
            "\n\n---\n\n**Summary**: The above analysis combines insights from "
            f"{len(results)} specialist agents ({', '.join(response.agents_used)}) "
            "to give you a comprehensive picture."
        )

    return response
