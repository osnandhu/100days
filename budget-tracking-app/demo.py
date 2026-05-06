#!/usr/bin/env python3
"""
=================================================================
  SMART BUDGET TRACKER - MULTI-AGENT SYSTEM DEMO
=================================================================
  Run this to demonstrate all agents working together.
  NO API keys needed. NO CrewAI needed. Runs fully offline.

  Usage:
    python demo.py
    python demo.py --module 1    (run specific module only)
=================================================================
"""

import json
import sys
import time
from datetime import datetime, timedelta

# ── Ensure project root is in path ──
sys.path.insert(0, ".")


def banner(text):
    width = 60
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")


def section(text):
    print(f"\n  ── {text} ──\n")


# ───────────────────────────────────────────────────────────────
# MODULE 1: Database Foundation
# ───────────────────────────────────────────────────────────────
def demo_module_1():
    banner("MODULE 1: Database + ORM Models")

    from database import init_db, get_db_session
    from models import User, Budget, Transaction

    # Create tables
    init_db()
    print("  [OK] Database tables created (SQLite)")

    # Check if user exists, create if not
    with get_db_session() as db:
        user = db.query(User).filter_by(name="Nanditha").first()
        if not user:
            user = User(name="Nanditha", email="nanditha@budget.app")
            db.add(user)
            db.flush()
            budget = Budget(
                user_id=user.id,
                monthly_salary=5000,
                total_expenditure=3500,
            )
            db.add(budget)
            db.flush()
            print("  [OK] Created user: Nanditha")
        else:
            _ = user.budget
            print(f"  [OK] Found existing user: {user.name} (id={user.id})")

        budget = user.budget
        print(f"\n  Financial Summary:")
        print(f"    Monthly Salary:  ${float(budget.monthly_salary):>8,.2f}")
        print(f"    Total Spending:  ${float(budget.total_expenditure):>8,.2f}")
        print(f"    Savings:         ${budget.savings:>8,.2f}")
        print(f"    Spending %:      {budget.spending_percentage:>7.1f}%")
        print(f"    Savings %:       {budget.savings_percentage:>7.1f}%")

    # Seed transactions if empty
    with get_db_session() as db:
        count = db.query(Transaction).filter_by(user_id=1).count()
        if count == 0:
            base = datetime.now() - timedelta(days=30)
            txns = [
                ("NTUC FairPrice", 85, "Groceries", 0),
                ("Grab ride to work", 15, "Transport", 1),
                ("Starbucks", 8, "Food", 2),
                ("Netflix subscription", 16, "Entertainment", 3),
                ("SP Group electricity", 120, "Utilities", 4),
                ("NTUC FairPrice", 92, "Groceries", 7),
                ("Grab ride", 12, "Transport", 8),
                ("Kopitiam lunch", 6, "Food", 9),
                ("NTUC FairPrice", 78, "Groceries", 14),
                ("Shell petrol", 70, "Fuel", 15),
                ("Grab ride", 18, "Transport", 16),
                ("Kopitiam lunch", 7, "Food", 17),
                ("Netflix subscription", 16, "Entertainment", 18),
                ("NTUC FairPrice", 88, "Groceries", 21),
                ("Grab ride", 14, "Transport", 22),
                ("Kopitiam lunch", 5, "Food", 23),
                # PLANTED ANOMALIES:
                ("NTUC FairPrice BULK BUY", 500, "Groceries", 25),
                ("Concert tickets (suspicious)", 350, "Entertainment", 27),
            ]
            for desc, amt, cat, days in txns:
                db.add(Transaction(
                    user_id=1, description=desc, amount=amt,
                    category=cat, date=base + timedelta(days=days),
                ))
            print(f"\n  [OK] Seeded {len(txns)} transactions (2 planted anomalies)")
        else:
            print(f"\n  [OK] Found {count} existing transactions")

    print("\n  WHAT YOU LEARNED:")
    print("  - SQLAlchemy ORM maps Python classes → database tables")
    print("  - Budget.savings is a @property (computed, not stored)")
    print("  - Context manager (with get_db_session()) auto-commits/rollbacks")


# ───────────────────────────────────────────────────────────────
# MODULE 2: Spending Anomaly Detection
# ───────────────────────────────────────────────────────────────
def demo_module_2():
    banner("MODULE 2: Spending Anomaly Detection")
    print("  Methods: Z-Score (statistical) + Isolation Forest (ML)")
    print()

    from services.agents.spending_agent import (
        FetchTransactionsTool,
        AnomalyDetectionTool,
    )

    section("Step 1: Fetch transactions from database")
    fetch_tool = FetchTransactionsTool()
    txn_json = fetch_tool._run(user_id=1, months=3)
    txns = json.loads(txn_json)
    print(f"  Fetched {len(txns)} transactions")
    print(f"  {'Amount':>10}  {'Category':<15} Description")
    print(f"  {'------':>10}  {'--------':<15} -----------")
    for t in txns[:6]:
        print(f"  ${t['amount']:>8.2f}  {t['category']:<15} {t['description']}")
    if len(txns) > 6:
        print(f"  ... and {len(txns)-6} more")

    section("Step 2: Run dual anomaly detection")
    detect_tool = AnomalyDetectionTool()
    result_json = detect_tool._run(transactions_json=txn_json)
    result = json.loads(result_json)

    print(f"  Analyzed: {result['total_transactions']} transactions")
    print(f"  Anomalies: {result['anomalies_found']} detected")
    print()

    if result["anomalies"]:
        print(f"  {'Severity':>10}  {'Amount':>8}  {'Category':<15} {'Method':<18} Description")
        print(f"  {'--------':>10}  {'------':>8}  {'--------':<15} {'------':<18} -----------")
        for a in result["anomalies"]:
            method = a["method"]
            if "z_score" in a:
                method += f" (z={a['z_score']})"
            print(f"  {a['severity']:>10}  ${a['amount']:>7.2f}  {a['category']:<15} {method:<18} {a['description']}")

    print("\n  WHAT YOU LEARNED:")
    print("  - Z-score detects outliers WITHIN each category")
    print("  - Isolation Forest detects outliers ACROSS all features")
    print("  - When both agree → higher severity (HIGH/CRITICAL)")


# ───────────────────────────────────────────────────────────────
# MODULE 3: Orchestrator (Query Routing)
# ───────────────────────────────────────────────────────────────
def demo_module_3():
    banner("MODULE 3: Orchestrator - Query Routing")
    print("  Routes user questions to the right specialist agent(s)")
    print()

    from services.agents.orchestrator import classify_query

    queries = [
        "What is the inflation forecast for Singapore?",
        "Are there any unusual transactions in my account?",
        "How should I budget my $5000 salary?",
        "Can I save $1000/month with rising food inflation?",
        "Hello, what can you do?",
    ]

    print(f"  {'Query':<55} {'Routed To':<25}")
    print(f"  {'-----':<55} {'---------':<25}")

    for q in queries:
        result = classify_query(q)
        agents = " + ".join(a.value for a in result.agents_to_call)
        # Truncate long queries
        display_q = q if len(q) <= 53 else q[:50] + "..."
        print(f"  {display_q:<55} {agents:<25}")

    print("\n  KEY INSIGHT: Complex queries route to MULTIPLE agents.")
    print('  "save + inflation" → triggers both RAG + Inflation agents.')


# ───────────────────────────────────────────────────────────────
# MODULE 4: Inflation Forecasting
# ───────────────────────────────────────────────────────────────
def demo_module_4():
    banner("MODULE 4: Inflation Forecasting")
    print("  Data source: World Bank API (Singapore CPI)")
    print("  Model: Prophet time series (or linear fallback)")
    print()

    from services.agents.inflation_agent import (
        WorldBankCPITool,
        ProphetForecastTool,
    )

    section("Step 1: Fetch historical CPI data")
    cpi_tool = WorldBankCPITool()
    cpi_json = cpi_tool._run(start_year=2015, end_year=2024)
    data = json.loads(cpi_json)

    print(f"  Year   CPI %    Trend")
    print(f"  ----   -----    -----")
    for r in data:
        val = r["cpi_inflation"]
        bar_len = max(0, int(abs(val) * 4))
        bar = "#" * bar_len
        sign = "+" if val > 0 else ""
        indicator = "  " + bar if val >= 0 else "  " + bar + " (deflation)"
        print(f"  {r['year']}  {sign}{val:>5.2f}%  {indicator}")

    section("Step 2: Forecast next 6 months")
    forecast_tool = ProphetForecastTool()
    forecast_json = forecast_tool._run(historical_data=cpi_json, forecast_months=6)
    forecast = json.loads(forecast_json)

    print(f"  Method: {forecast['confidence_level']}")
    print()
    print(f"  {'Month':<10}  {'Predicted':>10}  {'Range':>20}")
    print(f"  {'-----':<10}  {'---------':>10}  {'-----':>20}")
    for p in forecast["predictions"]:
        rng = f"[{p['lower_bound']:.2f}% – {p['upper_bound']:.2f}%]"
        print(f"  {p['date']:<10}  {p['predicted_inflation']:>9.2f}%  {rng:>20}")

    print("\n  WHAT YOU LEARNED:")
    print("  - CPI inflation = how fast prices rise year-over-year")
    print("  - Post-COVID spike (6.12% in 2022) now cooling down")
    print("  - Confidence intervals widen = more uncertainty further out")


# ───────────────────────────────────────────────────────────────
# MODULE 5: RAG Financial Advisor
# ───────────────────────────────────────────────────────────────
def demo_module_5():
    banner("MODULE 5: RAG Financial Advisor")
    print("  FAISS vector search + financial knowledge base")
    print()

    from services.agents.rag_agent import (
        FinancialAdvisorTool,
        initialize_knowledge_base,
    )

    section("Step 1: Build FAISS index from knowledge base")
    kb = initialize_knowledge_base()
    print(f"  Documents loaded: {len(kb.documents)}")
    print(f"  FAISS index: {kb.index.ntotal} vectors, dim={kb.dimension}")

    section("Step 2: Semantic search (find relevant docs)")
    test_queries = [
        "How do I deal with rising inflation?",
        "What is a good budget rule?",
        "How to pay off debt faster?",
    ]
    for q in test_queries:
        results = kb.search(q, top_k=2)
        print(f"  Q: \"{q}\"")
        for r in results:
            print(f"     → [{r['category']}] {r['title']} (score: {r['relevance_score']})")
        print()

    section("Step 3: Full RAG pipeline (retrieve + generate)")
    rag_tool = FinancialAdvisorTool()
    result = json.loads(rag_tool._run(
        question="How can I save $1000/month when food prices are rising?",
        user_context="I earn $5000/month, single, renting in Singapore",
    ))

    print(f"  Question: How can I save $1000/month when food prices are rising?")
    print(f"  Sources retrieved: {result['sources']}")
    print(f"\n  Advice:")
    for line in result["advice"].split("\n")[:8]:
        print(f"    {line}")

    print("\n  WHAT YOU LEARNED:")
    print("  - Embeddings turn text → numbers (vectors)")
    print("  - FAISS finds semantically similar documents fast")
    print("  - RAG = Retrieve docs + Augment prompt + Generate answer")


# ───────────────────────────────────────────────────────────────
# MODULE 6: Full Pipeline - The Orchestrator Flow
# ───────────────────────────────────────────────────────────────
def demo_module_6():
    banner("MODULE 6: Full Multi-Agent Pipeline")
    print("  Simulating: 'Can I save $1000/month with rising food inflation?'")
    print()

    from services.agents.orchestrator import classify_query, synthesise_results, AgentResult, AgentType
    from services.agents.inflation_agent import WorldBankCPITool, ProphetForecastTool
    from services.agents.spending_agent import FetchTransactionsTool, AnomalyDetectionTool
    from services.agents.rag_agent import FinancialAdvisorTool, initialize_knowledge_base

    query = "Can I save $1000/month with rising food inflation?"

    # Step 1: Route
    section("Step 1: Orchestrator routes the query")
    routing = classify_query(query)
    agents = [a.value for a in routing.agents_to_call]
    print(f"  Query: \"{query}\"")
    print(f"  Routing decision: {agents}")
    print(f"  Reason: {routing.reasoning}")

    results = []

    # Step 2: Run Inflation Agent
    if any(a.value == "inflation" for a in routing.agents_to_call):
        section("Step 2: Inflation Agent analyzes price trends")
        cpi_tool = WorldBankCPITool()
        forecast_tool = ProphetForecastTool()
        cpi_data = cpi_tool._run(start_year=2020, end_year=2024)
        forecast = json.loads(forecast_tool._run(historical_data=cpi_data, forecast_months=6))

        avg_pred = sum(p["predicted_inflation"] for p in forecast["predictions"]) / len(forecast["predictions"])
        inflation_summary = f"Singapore inflation forecast: {avg_pred:.2f}% avg over next 6 months. Recent trend: cooling from 6.12% (2022) peak."
        print(f"  Result: {inflation_summary}")
        results.append(AgentResult(agent_type=AgentType.INFLATION, raw_output=inflation_summary, success=True))

    # Step 3: Run RAG Agent
    if any(a.value == "rag" for a in routing.agents_to_call):
        section("Step 3: RAG Agent retrieves relevant advice")
        initialize_knowledge_base()
        rag_tool = FinancialAdvisorTool()
        rag_result = json.loads(rag_tool._run(
            question=query,
            user_context="Earns $5000/month, spends $500 on groceries",
        ))
        print(f"  Sources: {rag_result['sources']}")
        advice_preview = rag_result["advice"][:200]
        results.append(AgentResult(agent_type=AgentType.RAG, raw_output=rag_result["advice"], success=True))

    # Step 4: Synthesize
    section("Step 4: Orchestrator synthesizes all results")
    final = synthesise_results(query, results)
    print(f"  Agents used: {final.agents_used}")
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  FINAL SYNTHESISED RESPONSE                         │")
    print(f"  └─────────────────────────────────────────────────────┘")
    for line in final.synthesised_response.split("\n"):
        print(f"  {line}")


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smart Budget Tracker Demo")
    parser.add_argument("--module", type=int, help="Run specific module (1-6)")
    args = parser.parse_args()

    modules = {
        1: ("Database + Models", demo_module_1),
        2: ("Spending Anomaly Detection", demo_module_2),
        3: ("Orchestrator Routing", demo_module_3),
        4: ("Inflation Forecasting", demo_module_4),
        5: ("RAG Financial Advisor", demo_module_5),
        6: ("Full Multi-Agent Pipeline", demo_module_6),
    }

    if args.module:
        if args.module in modules:
            name, func = modules[args.module]
            func()
        else:
            print(f"Module {args.module} not found. Choose 1-6.")
        return

    print("""
  ╔═══════════════════════════════════════════════════════╗
  ║   SMART BUDGET TRACKER - MULTI-AGENT AI SYSTEM       ║
  ║   Built with: CrewAI + FAISS + Prophet + SQLAlchemy   ║
  ╚═══════════════════════════════════════════════════════╝
    """)

    for num, (name, func) in modules.items():
        func()
        if num < len(modules):
            print(f"\n  {'─'*50}")
            print(f"  Press Enter for next module ({num+1}/{len(modules)})...")
            try:
                input()
            except EOFError:
                pass

    banner("DEMO COMPLETE")
    print("  All 6 modules demonstrated successfully.")
    print("  No API keys or external services required.")
    print()
    print("  To run with full CrewAI + LLM integration:")
    print("    pip install crewai crewai-tools litellm prophet")
    print("    export LLM_API_KEY=sk-...")
    print("    uvicorn main_api:app --reload")
    print()


if __name__ == "__main__":
    main()
