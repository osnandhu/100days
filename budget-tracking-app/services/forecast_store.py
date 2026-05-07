"""
Forecast Store
==============
Persists inflation forecast results and spending anomalies to the database.

WHY STORE FORECASTS:
1. Avoid re-running expensive Prophet fits for the same time period
2. Track prediction accuracy over time (compare forecast vs actual)
3. Show forecast history in dashboards
4. Audit trail: when was a forecast made, what model was used?
"""

import json
from datetime import datetime
from typing import List

from database import get_db_session
from models import InflationForecast, SpendingAnomaly


def save_inflation_forecast(forecast_json: str) -> List[InflationForecast]:
    """
    Parse forecast JSON from the inflation agent and save each prediction.
    Returns the saved ORM objects.
    """
    data = json.loads(forecast_json)

    if "error" in data:
        return []

    saved = []
    with get_db_session() as db:
        for pred in data.get("predictions", []):
            record = InflationForecast(
                forecast_date=pred["date"],
                predicted_inflation=pred["predicted_inflation"],
                lower_bound=pred["lower_bound"],
                upper_bound=pred["upper_bound"],
                confidence_level=data.get("confidence_level", "95%"),
                data_source=data.get("data_source", "unknown"),
                model_used="prophet" if "linear" not in data.get("confidence_level", "") else "linear_trend",
            )
            db.add(record)
            saved.append(record)

        db.flush()
        for r in saved:
            _ = r.id

    return saved


def get_latest_forecast() -> List[dict]:
    """Retrieve the most recent set of forecast predictions (last 6 records)."""
    with get_db_session() as db:
        records = (
            db.query(InflationForecast)
            .order_by(InflationForecast.id.desc())
            .limit(6)
            .all()
        )
        records.reverse()

        return [
            {
                "date": r.forecast_date,
                "predicted_inflation": r.predicted_inflation,
                "lower_bound": r.lower_bound,
                "upper_bound": r.upper_bound,
                "model": r.model_used,
                "generated_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]


def save_spending_anomalies(user_id: int, anomalies_json: str) -> int:
    """
    Save detected anomalies to the database.
    Returns the number of anomalies saved.
    """
    data = json.loads(anomalies_json)
    anomalies = data.get("anomalies", [])

    if not anomalies:
        return 0

    count = 0
    with get_db_session() as db:
        for a in anomalies:
            record = SpendingAnomaly(
                user_id=user_id,
                transaction_id=a["transaction_id"],
                severity=a["severity"],
                detection_method=a.get("method", "unknown"),
                z_score=a.get("z_score"),
            )
            db.add(record)
            count += 1

    return count


def get_anomaly_history(user_id: int) -> List[dict]:
    """Retrieve all past anomalies for a user."""
    with get_db_session() as db:
        records = (
            db.query(SpendingAnomaly)
            .filter(SpendingAnomaly.user_id == user_id)
            .order_by(SpendingAnomaly.created_at.desc())
            .all()
        )

        return [
            {
                "transaction_id": r.transaction_id,
                "severity": r.severity,
                "method": r.detection_method,
                "z_score": r.z_score,
                "detected_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
