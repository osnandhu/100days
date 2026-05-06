"""
Inflation Analyst Agent
=======================
Fetches historical CPI data from the World Bank API and forecasts
future inflation using Facebook Prophet.

WHY THIS APPROACH:
- World Bank API is free, reliable, no auth needed
- Prophet handles yearly seasonality and missing data gracefully
- Confidence intervals come built-in (no extra stats work)

TRADE-OFF:
  World Bank provides ANNUAL CPI data. We interpolate to monthly for
  Prophet, which works well for trend direction but overstates precision.
  In production, use Singapore DOS monthly CPI data instead.
"""

import json
import logging
from typing import Type

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field

# Heavy deps: deferred so tools work standalone for demos
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from crewai import Agent, Task
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    BaseTool = object

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: World Bank CPI Fetcher
# ---------------------------------------------------------------------------
# CrewAI tools need an input schema (Pydantic model) and a _run method.
# The schema tells the LLM what arguments the tool accepts.
# ---------------------------------------------------------------------------


class FetchCPIInput(BaseModel):
    start_year: int = Field(default=2015, description="Start year for CPI data range")
    end_year: int = Field(default=2024, description="End year for CPI data range")


class WorldBankCPITool(BaseTool):
    """
    Calls the World Bank API to get Singapore's annual CPI inflation rate.

    HOW THE API WORKS:
    - Endpoint: api.worldbank.org/v2/country/{ISO}/indicator/{INDICATOR}
    - SGP = Singapore's ISO country code
    - FP.CPI.TOTL.ZG = "Inflation, consumer prices (annual %)"
    - Returns JSON with two elements: [metadata, data_array]
    - Each data entry has {"date": "2023", "value": 4.82, ...}
    """

    name: str = "fetch_singapore_cpi"
    description: str = (
        "Fetches Singapore annual CPI inflation rates from the World Bank API. "
        "Returns JSON array of {year, cpi_inflation} records sorted by year."
    )
    args_schema: Type[BaseModel] = FetchCPIInput

    def _run(self, start_year: int = 2015, end_year: int = 2024) -> str:
        url = "https://api.worldbank.org/v2/country/SGP/indicator/FP.CPI.TOTL.ZG"
        params = {
            "date": f"{start_year}:{end_year}",
            "format": "json",
            "per_page": 100,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"World Bank API failed ({e}), using cached CPI data")
            return json.dumps(_get_fallback_cpi(start_year, end_year))

        data = response.json()

        # World Bank returns [metadata_dict, data_list]. If data_list is None,
        # no records matched our query.
        if len(data) < 2 or data[1] is None:
            return json.dumps(_get_fallback_cpi(start_year, end_year))

        records = []
        for entry in data[1]:
            if entry["value"] is not None:
                records.append(
                    {
                        "year": int(entry["date"]),
                        "cpi_inflation": round(entry["value"], 2),
                    }
                )

        records.sort(key=lambda x: x["year"])
        logger.info(f"Fetched {len(records)} years of CPI data from World Bank")
        return json.dumps(records)


# Real Singapore CPI data (source: World Bank, cached for offline demos)
_SINGAPORE_CPI_CACHE = {
    2015: -0.52, 2016: -0.53, 2017: 0.58, 2018: 0.44, 2019: 0.57,
    2020: -0.18, 2021: 2.30, 2022: 6.12, 2023: 4.82, 2024: 2.40,
}


def _get_fallback_cpi(start_year: int, end_year: int) -> list:
    """Return cached CPI data when the API is unavailable."""
    return [
        {"year": y, "cpi_inflation": v}
        for y, v in sorted(_SINGAPORE_CPI_CACHE.items())
        if start_year <= y <= end_year
    ]


# ---------------------------------------------------------------------------
# Tool 2: Prophet Forecaster
# ---------------------------------------------------------------------------
# Prophet needs a DataFrame with columns 'ds' (date) and 'y' (value).
# Since our CPI data is annual, we expand each year into 12 monthly points
# so Prophet has enough data points to model trends.
# ---------------------------------------------------------------------------


class ForecastInput(BaseModel):
    historical_data: str = Field(
        description="JSON string of CPI data: [{year, cpi_inflation}, ...]"
    )
    forecast_months: int = Field(
        default=6, description="Number of months to forecast ahead"
    )


class ProphetForecastTool(BaseTool):
    """
    Fits a Prophet model on historical CPI data and forecasts N months ahead.

    WHY PROPHET OVER ARIMA:
    - Prophet auto-detects yearly seasonality
    - Handles missing data without manual imputation
    - Built-in uncertainty intervals (interval_width parameter)
    - Much simpler API than statsmodels ARIMA/SARIMAX

    TRADE-OFF:
      Prophet is a heavy dependency (~100MB). For a simpler project,
      you could use exponential smoothing from statsmodels instead.
    """

    name: str = "forecast_inflation"
    description: str = (
        "Takes historical CPI data JSON and forecasts inflation for the next N months "
        "using Prophet. Returns predictions with 95% confidence intervals."
    )
    args_schema: Type[BaseModel] = ForecastInput

    def _run(self, historical_data: str, forecast_months: int = 6) -> str:
        if not PROPHET_AVAILABLE:
            # Simple linear trend fallback when Prophet isn't installed
            return self._simple_forecast(historical_data, forecast_months)

        try:
            records = json.loads(historical_data)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON input for forecasting"})

        if isinstance(records, dict) and "error" in records:
            return historical_data

        df = pd.DataFrame(records)

        # Expand annual data → monthly by repeating each year's value 12 times.
        # This gives Prophet enough data points to fit a trend.
        # Each month within a year gets the same CPI value (simplification).
        monthly_dates = []
        monthly_values = []
        for _, row in df.iterrows():
            for month in range(1, 13):
                monthly_dates.append(
                    pd.Timestamp(year=int(row["year"]), month=month, day=1)
                )
                # Add small noise so Prophet doesn't see flat segments
                noise = np.random.normal(0, 0.05)
                monthly_values.append(row["cpi_inflation"] + noise)

        prophet_df = pd.DataFrame({"ds": monthly_dates, "y": monthly_values})

        # Configure Prophet: only yearly seasonality makes sense for CPI
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,  # 95% confidence interval
        )

        # Prophet prints fitting logs by default; suppress them
        model.fit(prophet_df)

        # Generate future dates: 'MS' = month start frequency
        future = model.make_future_dataframe(periods=forecast_months, freq="MS")
        forecast = model.predict(future)

        # Extract only the forecasted period (last N rows)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(
            forecast_months
        )

        predictions = []
        for _, row in result.iterrows():
            predictions.append(
                {
                    "date": row["ds"].strftime("%Y-%m"),
                    "predicted_inflation": round(row["yhat"], 2),
                    "lower_bound": round(row["yhat_lower"], 2),
                    "upper_bound": round(row["yhat_upper"], 2),
                }
            )

        return json.dumps(
            {
                "forecast_months": forecast_months,
                "predictions": predictions,
                "confidence_level": "95%",
                "data_source": "World Bank CPI (annual, interpolated to monthly)",
            }
        )

    def _simple_forecast(self, historical_data: str, forecast_months: int) -> str:
        """Linear trend fallback when Prophet is not installed."""
        try:
            records = json.loads(historical_data)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON input"})

        if isinstance(records, dict) and "error" in records:
            return historical_data

        df = pd.DataFrame(records)
        values = df["cpi_inflation"].values

        # Simple linear regression for trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        last_val = values[-1]
        std = np.std(values)

        predictions = []
        for i in range(1, forecast_months + 1):
            pred = last_val + slope * (i / 12)
            predictions.append(
                {
                    "date": f"2025-{i:02d}",
                    "predicted_inflation": round(pred, 2),
                    "lower_bound": round(pred - 1.96 * std, 2),
                    "upper_bound": round(pred + 1.96 * std, 2),
                }
            )

        return json.dumps(
            {
                "forecast_months": forecast_months,
                "predictions": predictions,
                "confidence_level": "95% (linear trend fallback - install prophet for better results)",
                "data_source": "World Bank CPI (cached)",
            }
        )


# ---------------------------------------------------------------------------
# Agent + Task Factory Functions
# ---------------------------------------------------------------------------
# These create the CrewAI Agent and Task objects. Kept as functions so
# crew_manager.py can call them with different LLM configurations.
# ---------------------------------------------------------------------------


def create_inflation_agent(llm):
    if not CREWAI_AVAILABLE:
        raise ImportError("crewai is required to create agents. pip install crewai")
    return Agent(
        role="Inflation Forecasting Specialist",
        goal=(
            "Analyze historical CPI data for Singapore and produce accurate "
            "6-month inflation forecasts with confidence intervals."
        ),
        backstory=(
            "You are a senior economist specializing in Southeast Asian markets. "
            "You have 15 years of experience in inflation analysis and monetary "
            "policy research. You always ground analysis in data and provide "
            "clear, actionable insights for personal finance decisions."
        ),
        tools=[WorldBankCPITool(), ProphetForecastTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_inflation_task(agent, context: str = ""):
    if not CREWAI_AVAILABLE:
        raise ImportError("crewai is required to create tasks. pip install crewai")
    return Task(
        description=(
            "Fetch Singapore CPI inflation data from 2015 to 2024 using the "
            "World Bank API tool. Then use the Prophet forecasting tool to predict "
            "inflation for the next 6 months. Summarise the historical trend and "
            f"forecast results clearly. User context: {context or 'General forecast requested.'}"
        ),
        expected_output=(
            "A structured analysis containing:\n"
            "1. Historical inflation trend summary (2015-2024)\n"
            "2. Six-month forecast with predicted rates and confidence intervals\n"
            "3. Key insight: is inflation rising, falling, or stable?\n"
            "4. What this means for household budgets"
        ),
        agent=agent,
    )
