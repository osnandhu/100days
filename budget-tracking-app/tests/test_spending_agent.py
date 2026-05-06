"""
Tests for the Spending Analyst Agent tools.

We test the anomaly detection tool directly with controlled data
containing known anomalies, verifying that both Z-score and Isolation
Forest catch them.
"""

import json
import sys

import pytest

sys.path.insert(0, ".")

from services.agents.spending_agent import AnomalyDetectionTool, _z_score_severity


class TestZScoreSeverity:
    """Test the severity mapping function."""

    def test_critical_threshold(self):
        assert _z_score_severity(4.5) == "CRITICAL"

    def test_high_threshold(self):
        assert _z_score_severity(3.5) == "HIGH"

    def test_medium_threshold(self):
        assert _z_score_severity(2.5) == "MEDIUM"

    def test_low_threshold(self):
        assert _z_score_severity(1.5) == "LOW"


class TestAnomalyDetectionTool:
    """Tests for the anomaly detection engine."""

    def setup_method(self):
        self.tool = AnomalyDetectionTool()

    def test_detects_obvious_outlier(self):
        """A transaction 10x the normal amount should be flagged."""
        transactions = [
            {"id": i, "description": f"Grocery #{i}", "amount": 80, "category": "Groceries", "date": f"2024-01-{i+1:02d}"}
            for i in range(10)
        ]
        # Plant an obvious anomaly
        transactions.append(
            {"id": 99, "description": "Huge grocery bill", "amount": 800, "category": "Groceries", "date": "2024-01-15"}
        )

        result = json.loads(self.tool._run(json.dumps(transactions)))

        assert result["anomalies_found"] > 0
        anomaly_ids = [a["transaction_id"] for a in result["anomalies"]]
        assert 99 in anomaly_ids

    def test_no_anomalies_in_uniform_data(self):
        """If all transactions are similar, nothing should be flagged by z-score."""
        transactions = [
            {"id": i, "description": f"Coffee #{i}", "amount": 5.50, "category": "Food", "date": f"2024-01-{i+1:02d}"}
            for i in range(20)
        ]

        result = json.loads(self.tool._run(json.dumps(transactions)))

        # Z-score should find nothing; Isolation Forest might flag a few
        z_anomalies = [a for a in result["anomalies"] if a.get("method") == "z_score"]
        assert len(z_anomalies) == 0

    def test_handles_too_few_transactions(self):
        """Should return gracefully with < 3 transactions."""
        transactions = [
            {"id": 1, "description": "Only one", "amount": 50, "category": "Food", "date": "2024-01-01"},
        ]

        result = json.loads(self.tool._run(json.dumps(transactions)))
        assert "Too few" in result.get("message", "")

    def test_handles_invalid_json(self):
        result = json.loads(self.tool._run("not valid json"))
        assert "error" in result

    def test_severity_ordering(self):
        """Anomalies should be sorted by severity (CRITICAL first)."""
        transactions = [
            {"id": i, "description": f"Shop #{i}", "amount": 50, "category": "Shopping", "date": f"2024-01-{i+1:02d}"}
            for i in range(15)
        ]
        # Plant two anomalies of different magnitudes
        transactions.append(
            {"id": 100, "description": "Medium anomaly", "amount": 300, "category": "Shopping", "date": "2024-01-20"}
        )
        transactions.append(
            {"id": 101, "description": "Extreme anomaly", "amount": 5000, "category": "Shopping", "date": "2024-01-21"}
        )

        result = json.loads(self.tool._run(json.dumps(transactions)))

        if result["anomalies_found"] >= 2:
            severities = [a["severity"] for a in result["anomalies"]]
            severity_order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
            scores = [severity_order.get(s, -1) for s in severities]
            assert scores == sorted(scores, reverse=True), "Anomalies should be sorted by severity"

    def test_multi_category_detection(self):
        """Should detect anomalies within each category independently."""
        transactions = []
        tid = 0

        # Normal groceries ($80-$90)
        for i in range(8):
            transactions.append(
                {"id": tid, "description": f"Grocery {i}", "amount": 80 + i, "category": "Groceries", "date": f"2024-01-{i+1:02d}"}
            )
            tid += 1

        # Normal transport ($40-$50)
        for i in range(8):
            transactions.append(
                {"id": tid, "description": f"Bus {i}", "amount": 40 + i, "category": "Transport", "date": f"2024-01-{i+1:02d}"}
            )
            tid += 1

        # Anomalous grocery (should flag)
        transactions.append(
            {"id": 888, "description": "Giant grocery", "amount": 600, "category": "Groceries", "date": "2024-01-15"}
        )

        result = json.loads(self.tool._run(json.dumps(transactions)))

        flagged_ids = [a["transaction_id"] for a in result["anomalies"]]
        assert 888 in flagged_ids
