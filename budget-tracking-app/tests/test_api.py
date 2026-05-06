"""
Tests for the FastAPI endpoints.

Uses httpx.AsyncClient with FastAPI's TestClient pattern. These tests
verify request/response shapes without running real agent logic.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, ".")

# Mock the startup initialization so tests don't need a real DB
with patch("main_api.init_db"), patch("main_api.initialize"):
    from main_api import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestAnalyzeEndpoint:
    @patch("main_api.run_full_analysis")
    def test_analyze_success(self, mock_analysis):
        mock_analysis.return_value = MagicMock(
            synthesised_response="Analysis complete",
            agents_used=["inflation", "rag"],
        )

        response = client.post(
            "/agent/analyze",
            json={"query": "Can I save with inflation rising?", "user_id": 1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["agents_used"]) == 2

    def test_analyze_missing_query(self):
        """Should return 422 when required field is missing."""
        response = client.post("/agent/analyze", json={})
        assert response.status_code == 422


class TestForecastEndpoint:
    @patch("main_api.run_inflation_forecast")
    def test_forecast_success(self, mock_forecast):
        mock_forecast.return_value = "Inflation predicted at 3.5%"

        response = client.post("/agent/forecast", json={"context": ""})

        assert response.status_code == 200
        assert response.json()["agents_used"] == ["inflation"]


class TestAnomaliesEndpoint:
    @patch("main_api.run_spending_analysis")
    def test_anomalies_success(self, mock_analysis):
        mock_analysis.return_value = "No anomalies found"

        response = client.post("/agent/anomalies", json={"user_id": 1})

        assert response.status_code == 200
        assert response.json()["agents_used"] == ["spending"]

    def test_anomalies_missing_user_id(self):
        response = client.post("/agent/anomalies", json={})
        assert response.status_code == 422


class TestAdviceEndpoint:
    @patch("main_api.run_rag_advice")
    def test_advice_success(self, mock_advice):
        mock_advice.return_value = "Consider the 50/30/20 rule"

        response = client.post(
            "/agent/advice",
            json={"question": "How to budget?"},
        )

        assert response.status_code == 200
        assert response.json()["agents_used"] == ["rag"]

    def test_advice_missing_question(self):
        response = client.post("/agent/advice", json={})
        assert response.status_code == 422
