"""
Tests for the Orchestrator routing and synthesis logic.

The orchestrator is pure Python (no LLM calls), so every path is testable
without API keys or mocks. These tests verify the keyword routing works
correctly for various query types.
"""

import sys

import pytest

sys.path.insert(0, ".")

from services.agents.orchestrator import (
    AgentResult,
    AgentType,
    OrchestratorResponse,
    classify_query,
    synthesise_results,
)


class TestQueryClassification:
    """Tests for the keyword-based query router."""

    def test_inflation_keywords(self):
        """Inflation-related queries should route to the inflation agent."""
        queries = [
            "What is the inflation forecast?",
            "How will CPI change?",
            "Are prices going up?",
        ]
        for query in queries:
            result = classify_query(query)
            assert AgentType.INFLATION in result.agents_to_call, f"Failed for: {query}"

    def test_spending_keywords(self):
        """Spending queries should route to the spending agent."""
        queries = [
            "Detect anomalies in my spending",
            "Is my transaction history unusual?",
            "Show me my expenses",
        ]
        for query in queries:
            result = classify_query(query)
            assert AgentType.SPENDING in result.agents_to_call, f"Failed for: {query}"

    def test_rag_keywords(self):
        """Advice queries should route to the RAG agent."""
        queries = [
            "How should I budget my salary?",
            "Give me saving tips",
            "Can I invest with $500?",
            "Recommend a strategy",
        ]
        for query in queries:
            result = classify_query(query)
            assert AgentType.RAG in result.agents_to_call, f"Failed for: {query}"

    def test_multi_agent_routing(self):
        """Complex queries should trigger multiple agents."""
        query = "Can I save $1000/month with rising inflation?"
        result = classify_query(query)

        assert AgentType.INFLATION in result.agents_to_call
        assert AgentType.RAG in result.agents_to_call

    def test_default_to_rag(self):
        """Unrecognised queries should default to RAG."""
        result = classify_query("hello there")
        assert AgentType.RAG in result.agents_to_call
        assert len(result.agents_to_call) == 1

    def test_preserves_original_query(self):
        result = classify_query("test query 123")
        assert result.original_query == "test query 123"

    def test_case_insensitive(self):
        """Routing should work regardless of case."""
        result = classify_query("INFLATION FORECAST PLEASE")
        assert AgentType.INFLATION in result.agents_to_call

    def test_reasoning_is_populated(self):
        """The routing decision should explain itself."""
        result = classify_query("inflation forecast")
        assert len(result.reasoning) > 0
        assert "inflation" in result.reasoning.lower()


class TestResultSynthesis:
    """Tests for combining multiple agent outputs."""

    def test_single_result_synthesis(self):
        results = [
            AgentResult(
                agent_type=AgentType.INFLATION,
                raw_output="Inflation is predicted at 3.5%",
                success=True,
            )
        ]

        response = synthesise_results("test query", results)
        assert "Inflation Forecast" in response.synthesised_response
        assert "3.5%" in response.synthesised_response
        assert response.agents_used == ["inflation"]

    def test_multi_result_synthesis(self):
        results = [
            AgentResult(agent_type=AgentType.INFLATION, raw_output="Inflation rising", success=True),
            AgentResult(agent_type=AgentType.SPENDING, raw_output="No anomalies found", success=True),
            AgentResult(agent_type=AgentType.RAG, raw_output="Consider cutting subscriptions", success=True),
        ]

        response = synthesise_results("complex query", results)
        assert len(response.agents_used) == 3
        assert "Summary" in response.synthesised_response

    def test_failed_agent_handling(self):
        """Failed agents should show error message, not crash."""
        results = [
            AgentResult(agent_type=AgentType.INFLATION, raw_output="", success=False, error="API down"),
            AgentResult(agent_type=AgentType.RAG, raw_output="Here is advice", success=True),
        ]

        response = synthesise_results("test", results)
        assert "Unable to complete" in response.synthesised_response
        assert "API down" in response.synthesised_response
        assert "Here is advice" in response.synthesised_response

    def test_all_agents_failed(self):
        results = [
            AgentResult(agent_type=AgentType.INFLATION, raw_output="", success=False, error="timeout"),
        ]

        response = synthesise_results("test", results)
        assert "Unable to complete" in response.synthesised_response

    def test_individual_results_tracked(self):
        results = [
            AgentResult(agent_type=AgentType.RAG, raw_output="advice here", success=True),
        ]

        response = synthesise_results("test", results)
        assert len(response.individual_results) == 1
        assert response.individual_results[0]["status"] == "success"
