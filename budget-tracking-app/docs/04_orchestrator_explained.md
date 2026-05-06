# Orchestrator Agent - Deep Dive

## What Does This Component Do?

The orchestrator answers: **"Which agents should handle this query?"**

It's the router and synthesiser — it classifies user queries, dispatches
them to the right specialist agents, and combines the results.

---

## Core Concepts

### What is an Orchestrator Pattern?

In a multi-agent system, you need something to coordinate. There are three
common patterns:

**1. Sequential Pipeline** (simplest)
```
Query → Agent A → Agent B → Agent C → Response
```
Every query goes through every agent. Simple but wasteful — why run
inflation analysis for a question about debt payoff?

**2. Router/Dispatcher** (what we use)
```
Query → Classifier → Decide which agents → Run selected → Synthesise
```
Only runs agents that are relevant. More efficient. Trade-off: the
classifier needs to be smart enough to route correctly.

**3. Hierarchical Manager** (most complex)
```
Query → Manager Agent → Delegates to specialists → Manager synthesises
```
The manager itself is an LLM that decides everything. Most flexible but
slowest and most expensive (extra LLM call for every decision).

### Why We Chose the Router Pattern

- **Fast**: keyword matching is instant (no LLM call for routing)
- **Predictable**: you can trace exactly why a query went to an agent
- **Debuggable**: when routing is wrong, you fix a keyword list, not
  fine-tune a prompt
- **Cheap**: no extra LLM tokens for routing decisions

The trade-off: it's less flexible than an LLM-based router. A question
like "My rent went up, what should I do?" needs both inflation analysis
and budgeting advice, but might only match the RAG agent based on keywords.
We mitigate this by having overlapping keyword lists.

---

## How the Code Works (Step by Step)

### Step 1: Query Classification

```python
ROUTING_RULES = {
    AgentType.INFLATION: ["inflation", "cpi", "forecast", "prices", ...],
    AgentType.SPENDING:  ["spending", "anomaly", "unusual", ...],
    AgentType.RAG:       ["advice", "save", "budget", "recommend", ...],
}

def classify_query(query: str) -> RoutingDecision:
    query_lower = query.lower()
    matched_agents = []
    for agent_type, keywords in ROUTING_RULES.items():
        if any(kw in query_lower for kw in keywords):
            matched_agents.append(agent_type)
```

Example: "Can I save $1000/month with rising food inflation?"
- "save" → matches RAG
- "inflation" → matches INFLATION
- Result: both agents are called

### Step 2: Agent Execution (in crew_manager.py)

Each matched agent runs as its own single-agent Crew:

```python
for agent_type in routing.agents_to_call:
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    agent_results.append(result)
```

**Why separate Crews?** One agent failing doesn't crash the others. If the
World Bank API is down, the inflation agent fails gracefully while spending
analysis and RAG advice still work.

### Step 3: Synthesis

```python
def synthesise_results(query, results):
    sections = []
    for result in results:
        sections.append(f"**{result.agent_type.title()}**: {result.output}")
    return "\n---\n".join(sections)
```

Currently simple string concatenation. In production, you'd feed all
results into an LLM with instructions like "combine these findings into
a coherent paragraph."

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Keyword routing (not LLM routing) | Speed, cost, debuggability |
| Default to RAG on no match | RAG is most general-purpose |
| Separate Crews per agent | Fault isolation |
| String synthesis (not LLM synthesis) | Transparent, debuggable, no extra cost |
| Dataclass results (not dicts) | Type safety, IDE autocomplete, self-documenting |

---

## The Routing Rules in Detail

```
User says "inflation"     → Inflation Agent
User says "anomaly"       → Spending Agent
User says "how to save"   → RAG Agent
User says "save inflation" → Both RAG + Inflation
User says "hello there"   → Default to RAG (no keywords matched)
```

The keyword lists are intentionally overlapping. "Prices" maps to inflation
because price changes are inflation. "Help me" maps to RAG because it
implies an advice question.

---

## Common Pitfalls

1. **Keyword coverage gaps** — A user might ask "why is my electricity
   bill so high?" — this should hit spending analysis but "electricity"
   isn't in our keyword list. Fix: expand keywords over time based on
   real user queries.

2. **Running all agents is slow** — Each agent involves an LLM call
   (CrewAI uses the LLM to interpret tool results). If 3 agents run,
   that's at minimum 3 LLM calls. For latency-sensitive apps, consider
   running agents in parallel using asyncio.

3. **Synthesis losing nuance** — Simple string joining loses the connection
   between agent outputs. "Inflation is rising" + "groceries spending up"
   should lead to "your grocery costs are rising partly due to inflation"
   but simple concatenation doesn't make that leap.

---

## Practice Exercises (Weekend)

### Exercise 1: Add an LLM-Based Router
Replace `classify_query` with an LLM call that reads the query and outputs
which agents to call. Compare routing accuracy with the keyword approach
on 20 test queries.

### Exercise 2: Add Parallel Execution
Modify `run_full_analysis` in crew_manager.py to run agents concurrently
using `asyncio.gather()`. Measure the time difference vs sequential.

### Exercise 3: Build a Feedback Loop
After each query, ask the user "Was this helpful? Were the right agents
used?" Log the feedback. After collecting 50+ data points, analyze which
routing decisions were wrong and update the keyword lists.

### Exercise 4: Add a New Agent
Create a `SavingsGoalAgent` that calculates how long it will take to reach
a savings target given current income, spending, and inflation. Integrate
it into the orchestrator routing rules.
