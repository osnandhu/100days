"""
Multi-Agent System for Smart Budget Tracking
=============================================
This package contains four AI agents that work together via CrewAI:

1. inflation_agent  - Forecasts inflation using World Bank data + Prophet
2. spending_agent   - Detects spending anomalies using Isolation Forest + Z-score
3. rag_agent        - Gives personalised financial advice using FAISS + LiteLLM
4. orchestrator     - Coordinates the above agents based on user queries

Entry point: crew_manager.py creates and runs the CrewAI Crew.
"""
