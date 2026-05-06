"""
Spending Analyst Agent
======================
Detects unusual spending patterns using two complementary methods:

1. Z-SCORE (per-category):
   Measures how many standard deviations a transaction is from the mean
   for its category. Simple, interpretable, works well for single-variable
   outliers within known groups.

2. ISOLATION FOREST (cross-category):
   An unsupervised ML algorithm that isolates anomalies by randomly
   splitting the feature space. Points that are easy to isolate (few
   splits needed) are anomalies. Good for multivariate patterns the
   Z-score would miss (e.g., normal amount but unusual combination of
   category + time-of-month).

SEVERITY LEVELS:
  LOW      - Z-score 2-3 (mildly unusual, ~5% of transactions)
  MEDIUM   - Z-score 3-4 OR Isolation Forest flagged alone
  HIGH     - BOTH methods flag the same transaction
  CRITICAL - Z-score > 4 AND Isolation Forest flagged (extremely rare)
"""

import json
import logging
import sys
from datetime import datetime
from typing import Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.ensemble import IsolationForest

# Add parent paths so we can import project modules
sys.path.insert(0, ".")

# CrewAI imports are deferred — tools work standalone for testing/demos.
# Only needed when running as part of a Crew.
try:
    from crewai import Agent, Task
    from crewai.tools import BaseTool

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    BaseTool = object  # Fallback so class definitions still work
from database import get_db_session
from models import Transaction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: Fetch Transactions from Database
# ---------------------------------------------------------------------------
# This tool bridges our existing SQLAlchemy models with the CrewAI agent.
# The agent calls this tool to get raw transaction data, then the analysis
# tool processes it.
# ---------------------------------------------------------------------------


class FetchTransactionsInput(BaseModel):
    user_id: int = Field(description="The user ID to fetch transactions for")
    months: int = Field(default=3, description="How many months of history to fetch")


class FetchTransactionsTool(BaseTool):
    """
    Queries the SQLAlchemy Transaction table for a specific user.

    WHY A DEDICATED TOOL (instead of raw SQL):
    - Uses our existing ORM models = single source of truth
    - Session management via context manager = no connection leaks
    - Returns JSON so the LLM can reason about the data
    """

    name: str = "fetch_user_transactions"
    description: str = (
        "Fetches recent transactions for a user from the database. "
        "Returns JSON array of {id, description, amount, category, date} records."
    )
    args_schema: Type[BaseModel] = FetchTransactionsInput

    def _run(self, user_id: int, months: int = 3) -> str:
        with get_db_session() as db:
            cutoff = datetime.now() - pd.DateOffset(months=months)
            txns = (
                db.query(Transaction)
                .filter(Transaction.user_id == user_id)
                .filter(Transaction.date >= cutoff)
                .order_by(Transaction.date.desc())
                .all()
            )

            records = []
            for t in txns:
                records.append(
                    {
                        "id": t.id,
                        "description": t.description,
                        "amount": float(t.amount),
                        "category": t.category or "Uncategorised",
                        "date": t.date.strftime("%Y-%m-%d") if t.date else None,
                    }
                )

        if not records:
            return json.dumps({"message": "No transactions found", "anomalies": []})

        logger.info(f"Fetched {len(records)} transactions for user {user_id}")
        return json.dumps(records)


# ---------------------------------------------------------------------------
# Tool 2: Anomaly Detection Engine
# ---------------------------------------------------------------------------
# Takes transaction JSON and runs both Z-score and Isolation Forest.
# Returns a list of anomalies with severity ratings.
# ---------------------------------------------------------------------------


class DetectAnomaliesInput(BaseModel):
    transactions_json: str = Field(
        description="JSON string of transactions from fetch_user_transactions"
    )


class AnomalyDetectionTool(BaseTool):
    """
    Runs dual anomaly detection on transaction data.

    ALGORITHM DETAILS:

    Z-Score (per category):
      z = (x - mean) / std_dev
      If a transaction's amount is 3 std devs above the category mean,
      it's flagged. We group by category so a $200 grocery bill is
      compared against OTHER grocery bills, not against rent.

    Isolation Forest (global):
      Builds an ensemble of random trees. Each tree randomly picks a
      feature and a split point. Anomalies are isolated in fewer splits
      because they're far from normal data. contamination=0.1 means
      we expect ~10% of transactions to be unusual.
    """

    name: str = "detect_spending_anomalies"
    description: str = (
        "Analyzes transaction data for anomalies using Z-score and Isolation Forest. "
        "Returns list of flagged transactions with severity (LOW/MEDIUM/HIGH/CRITICAL)."
    )
    args_schema: Type[BaseModel] = DetectAnomaliesInput

    def _run(self, transactions_json: str) -> str:
        try:
            transactions = json.loads(transactions_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON input"})

        if isinstance(transactions, dict):
            return json.dumps(transactions)

        if len(transactions) < 3:
            return json.dumps(
                {"message": "Too few transactions for meaningful analysis", "anomalies": []}
            )

        df = pd.DataFrame(transactions)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        anomalies = []

        # --- PASS 1: Z-Score per category ---
        # Group transactions by category, compute z-score within each group.
        # This catches "that one grocery bill that was 4x your usual".
        z_flags = set()
        for category, group in df.groupby("category"):
            if len(group) < 2:
                continue

            amounts = group["amount"].values
            z_scores = np.abs(stats.zscore(amounts))

            for idx, z in zip(group.index, z_scores):
                if z > 2:
                    z_flags.add(idx)
                    severity = _z_score_severity(z)
                    anomalies.append(
                        {
                            "transaction_id": int(df.loc[idx, "id"]),
                            "description": df.loc[idx, "description"],
                            "amount": float(df.loc[idx, "amount"]),
                            "category": category,
                            "date": df.loc[idx, "date"],
                            "z_score": round(float(z), 2),
                            "method": "z_score",
                            "severity": severity,
                        }
                    )

        # --- PASS 2: Isolation Forest (global) ---
        # Uses amount + category encoded as numbers. Catches patterns
        # that z-score misses: e.g., normal amount but unusual category
        # for that time of month.
        iso_flags = set()
        if len(df) >= 5:
            features = df[["amount"]].copy()
            features["category_code"] = pd.Categorical(df["category"]).codes

            iso_forest = IsolationForest(
                contamination=0.1,  # Expect ~10% anomalies
                random_state=42,    # Reproducible results
                n_estimators=100,
            )
            predictions = iso_forest.fit_predict(features)

            # -1 = anomaly, 1 = normal
            for idx, pred in enumerate(predictions):
                if pred == -1:
                    iso_flags.add(idx)
                    if idx not in z_flags:
                        anomalies.append(
                            {
                                "transaction_id": int(df.loc[idx, "id"]),
                                "description": df.loc[idx, "description"],
                                "amount": float(df.loc[idx, "amount"]),
                                "category": df.loc[idx, "category"],
                                "date": df.loc[idx, "date"],
                                "method": "isolation_forest",
                                "severity": "MEDIUM",
                            }
                        )

        # --- PASS 3: Upgrade severity when both methods agree ---
        for anomaly in anomalies:
            idx_match = df[df["id"] == anomaly["transaction_id"]].index
            if len(idx_match) > 0:
                idx = idx_match[0]
                if idx in z_flags and idx in iso_flags:
                    z = anomaly.get("z_score", 0)
                    anomaly["severity"] = "CRITICAL" if z > 4 else "HIGH"
                    anomaly["method"] = "both_methods"

        anomalies.sort(
            key=lambda x: {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(
                x["severity"], 0
            ),
            reverse=True,
        )

        return json.dumps(
            {
                "total_transactions": len(df),
                "anomalies_found": len(anomalies),
                "anomalies": anomalies,
            }
        )


def _z_score_severity(z: float) -> str:
    """Maps a z-score to a human-readable severity level."""
    if z > 4:
        return "CRITICAL"
    elif z > 3:
        return "HIGH"
    elif z > 2:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Agent + Task Factory
# ---------------------------------------------------------------------------


def create_spending_agent(llm):
    if not CREWAI_AVAILABLE:
        raise ImportError("crewai is required to create agents. pip install crewai")
    return Agent(
        role="Spending Anomaly Detection Specialist",
        goal=(
            "Analyze a user's transaction history to detect unusual spending "
            "patterns and flag potential issues before they become problems."
        ),
        backstory=(
            "You are a forensic accountant turned personal finance analyst. "
            "You specialise in pattern recognition in financial data. "
            "You explain anomalies in plain language, not jargon, and always "
            "suggest concrete next steps."
        ),
        tools=[FetchTransactionsTool(), AnomalyDetectionTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_spending_task(agent, user_id: int, context: str = ""):
    if not CREWAI_AVAILABLE:
        raise ImportError("crewai is required to create tasks. pip install crewai")
    return Task(
        description=(
            f"Fetch the last 3 months of transactions for user {user_id} from the "
            f"database. Then run anomaly detection on the results. For each anomaly "
            f"found, explain WHY it's unusual and what the user should check. "
            f"User context: {context or 'General spending review requested.'}"
        ),
        expected_output=(
            "A spending analysis report containing:\n"
            "1. Summary: total transactions analyzed, anomalies found\n"
            "2. For each anomaly: description, amount, category, severity, "
            "and a plain-English explanation of why it's flagged\n"
            "3. Top 3 actionable recommendations"
        ),
        agent=agent,
    )
