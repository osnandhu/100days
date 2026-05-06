"""
Tests for the Inflation Agent tools.

TESTING STRATEGY:
  We test the tools directly (not via CrewAI) so tests are fast and
  don't need an LLM API key. The tools are pure Python functions that
  take input and return JSON — easy to test in isolation.

  Integration tests (tool + agent + LLM) belong in a separate suite
  and need API credentials. We skip those here.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from services.agents.inflation_agent import (
    ProphetForecastTool,
    WorldBankCPITool,
    create_inflation_agent,
)


class TestWorldBankCPITool:
    """Tests for the CPI data fetcher tool."""

    def setup_method(self):
        self.tool = WorldBankCPITool()

    @patch("services.agents.inflation_agent.requests.get")
    def test_successful_fetch(self, mock_get):
        """Tool should parse World Bank API response correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"page": 1, "pages": 1, "total": 2},
            [
                {"date": "2024", "value": 2.40, "indicator": {}},
                {"date": "2023", "value": 4.82, "indicator": {}},
            ],
        ]
        mock_get.return_value = mock_response

        result = json.loads(self.tool._run(start_year=2023, end_year=2024))

        assert len(result) == 2
        assert result[0]["year"] == 2023
        assert result[0]["cpi_inflation"] == 4.82
        assert result[1]["year"] == 2024

    @patch("services.agents.inflation_agent.requests.get")
    def test_handles_null_values(self, mock_get):
        """Tool should skip entries where value is None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"page": 1},
            [
                {"date": "2024", "value": 2.40, "indicator": {}},
                {"date": "2023", "value": None, "indicator": {}},
            ],
        ]
        mock_get.return_value = mock_response

        result = json.loads(self.tool._run())
        assert len(result) == 1
        assert result[0]["year"] == 2024

    @patch("services.agents.inflation_agent.requests.get")
    def test_handles_api_error(self, mock_get):
        """Tool should return error JSON on API failure, not crash."""
        mock_get.side_effect = Exception("Connection timeout")

        result = json.loads(self.tool._run())
        assert "error" in result

    @patch("services.agents.inflation_agent.requests.get")
    def test_handles_empty_response(self, mock_get):
        """Tool should handle empty data gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"page": 1}, None]
        mock_get.return_value = mock_response

        result = json.loads(self.tool._run())
        assert "error" in result


class TestProphetForecastTool:
    """Tests for the Prophet forecasting tool."""

    def setup_method(self):
        self.tool = ProphetForecastTool()

    def test_forecast_produces_correct_number_of_predictions(self, sample_cpi_data):
        """Forecast should return exactly the requested number of months."""
        data_json = json.dumps(sample_cpi_data)
        result = json.loads(self.tool._run(historical_data=data_json, forecast_months=6))

        assert "predictions" in result
        assert len(result["predictions"]) == 6

    def test_forecast_includes_confidence_intervals(self, sample_cpi_data):
        """Each prediction should have lower and upper bounds."""
        data_json = json.dumps(sample_cpi_data)
        result = json.loads(self.tool._run(historical_data=data_json, forecast_months=3))

        for pred in result["predictions"]:
            assert "predicted_inflation" in pred
            assert "lower_bound" in pred
            assert "upper_bound" in pred
            assert pred["lower_bound"] <= pred["predicted_inflation"]
            assert pred["predicted_inflation"] <= pred["upper_bound"]

    def test_forecast_handles_invalid_json(self):
        """Tool should return error on bad input, not crash."""
        result = json.loads(self.tool._run(historical_data="not json"))
        assert "error" in result

    def test_forecast_passes_through_upstream_errors(self):
        """If input contains an error from CPI tool, pass it through."""
        error_json = json.dumps({"error": "API was down"})
        result = json.loads(self.tool._run(historical_data=error_json))
        assert result["error"] == "API was down"


class TestAgentCreation:
    """Tests that agent factory functions produce valid agents."""

    def test_create_inflation_agent(self):
        mock_llm = MagicMock()
        agent = create_inflation_agent(mock_llm)

        assert agent.role == "Inflation Forecasting Specialist"
        assert len(agent.tools) == 2
        assert agent.allow_delegation is False
