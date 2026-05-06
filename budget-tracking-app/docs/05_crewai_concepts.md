# CrewAI Concepts - The Big Picture

## What is CrewAI?

CrewAI is a framework for building **multi-agent AI systems**. Instead of
one LLM doing everything, you create specialised agents that work together
like a team.

Think of it like a consulting firm:
- The **Agents** are team members with different expertise
- The **Tasks** are work assignments on the project
- The **Crew** is the team working together
- The **Process** defines how the team coordinates

---

## The Four Building Blocks

### 1. Agent

An agent is defined by:
- **Role**: job title ("Inflation Forecasting Specialist")
- **Goal**: what they're trying to achieve
- **Backstory**: personality and expertise (surprisingly important for
  output quality — the LLM "acts" more convincingly with good backstory)
- **Tools**: functions the agent can call
- **LLM**: which language model powers the agent

```python
agent = Agent(
    role="Inflation Forecasting Specialist",
    goal="Forecast inflation with confidence intervals",
    backstory="You are a senior economist...",
    tools=[WorldBankCPITool(), ProphetForecastTool()],
    llm=LLM(model="openai/gpt-4o-mini"),
)
```

**How the agent uses tools**: CrewAI formats the tools as function
descriptions and passes them to the LLM. The LLM decides which tool to
call, with what arguments. CrewAI executes the tool and feeds the result
back to the LLM. This loop continues until the agent has enough
information to produce a final answer.

### 2. Task

A task tells an agent what to do:
- **Description**: the work to be done (be specific!)
- **Expected output**: what "done" looks like
- **Agent**: who is responsible

```python
task = Task(
    description="Fetch CPI data 2015-2024 and forecast 6 months ahead",
    expected_output="Historical trend + forecast with confidence intervals",
    agent=inflation_agent,
)
```

**Tip**: The more specific the `expected_output`, the better the result.
"A report" is vague. "A numbered list with date, predicted rate, and
confidence interval for each month" is precise.

### 3. Crew

A crew is a group of agents working on tasks:

```python
crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
```

`kickoff()` starts the crew's work and blocks until all tasks complete.

### 4. Process

How the crew coordinates:

**Sequential** (default):
```
Task 1 → Task 2 → Task 3
```
Each task runs after the previous one finishes. Output from task 1 is
available as context for task 2. Use when later tasks depend on earlier ones.

**Hierarchical**:
```
Manager Agent → delegates → Specialist Agents → Manager synthesises
```
A manager agent decides which specialist to call and when. More flexible
but requires an additional LLM call for every management decision.

**We use sequential** because it's simpler and our pipeline naturally
flows: forecast → detect anomalies → generate advice.

---

## Tools: How Agents Interact with the World

Tools are functions that agents can call. CrewAI has two ways to create them:

### Method 1: @tool decorator (simple)

```python
from crewai.tools import tool

@tool("calculator")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))
```

### Method 2: BaseTool class (more control)

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class CalcInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluate a mathematical expression"
    args_schema: Type[BaseModel] = CalcInput

    def _run(self, expression: str) -> str:
        return str(eval(expression))
```

**We use BaseTool** because:
- Pydantic schemas give clear documentation for each parameter
- The LLM gets better descriptions of what the tool expects
- Easier to test (instantiate class, call _run directly)

---

## LLM Configuration with LiteLLM

CrewAI uses LiteLLM under the hood, so you can use any provider:

```python
from crewai import LLM

# OpenAI
llm = LLM(model="openai/gpt-4o-mini", api_key="sk-...")

# Anthropic
llm = LLM(model="anthropic/claude-sonnet-4-20250514", api_key="sk-ant-...")

# Local (Ollama)
llm = LLM(model="ollama/llama3")  # No API key needed

# Google
llm = LLM(model="gemini/gemini-pro", api_key="...")
```

We set this via environment variables so you never hardcode API keys:
```bash
export LLM_MODEL=openai/gpt-4o-mini
export LLM_API_KEY=sk-...
```

---

## How Our System Maps to CrewAI

```
┌───────────────────────────────────────────────────┐
│                  crew_manager.py                  │
│  Creates LLM → Creates Agents → Creates Tasks     │
│  → Builds Crew → crew.kickoff()                   │
└──────────────┬────────────────────────────────────┘
               │
  ┌────────────┼────────────┐
  │            │            │
  ▼            ▼            ▼
Agent 1      Agent 2      Agent 3
(Inflation)  (Spending)   (RAG)
  │            │            │
  ▼            ▼            ▼
Tools:       Tools:       Tools:
WorldBank    DB Query     FAISS Search
Prophet      IsoForest    LiteLLM Gen
```

Each agent is a self-contained specialist. The crew_manager.py is the
"team lead" that assembles the right team for each query.

---

## Common Patterns in CrewAI

### Pattern 1: Research → Analysis → Report

```python
researcher = Agent(role="Researcher", tools=[SearchTool()])
analyst = Agent(role="Analyst", tools=[ChartTool()])
writer = Agent(role="Report Writer", tools=[])

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
)
```

### Pattern 2: Parallel Specialists (our approach)

```python
# Run relevant agents independently, combine results
for agent_type in needed_agents:
    mini_crew = Crew(agents=[agent], tasks=[task])
    results.append(mini_crew.kickoff())
synthesise(results)
```

### Pattern 3: Manager-Delegate

```python
crew = Crew(
    agents=[specialist1, specialist2, specialist3],
    tasks=[open_ended_task],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4o"),  # manager needs strong model
)
```

---

## Practice Exercises (Weekend)

### Exercise 1: Build a Minimal Crew
Create a 2-agent crew from scratch (not using our budget code):
- Agent 1: "Joke Writer" with a tool that fetches random words
- Agent 2: "Joke Reviewer" that rates the joke 1-10
Run it and watch the agents interact. This helps you understand the
CrewAI execution flow in isolation.

### Exercise 2: Switch to Hierarchical Process
Change our system from sequential to hierarchical. Add a manager agent
that decides which specialists to call. Compare output quality and
latency with our current keyword-routing approach.

### Exercise 3: Add Memory
CrewAI supports agent memory (remembering past interactions). Enable it:
```python
agent = Agent(..., memory=True)
```
Ask the same agent the same question twice. Does it reference its
previous answer? When would memory be useful vs wasteful?

### Exercise 4: Build a CrewAI Pipeline
CrewAI has a Pipeline feature for chaining multiple crews. Build a
pipeline: Crew 1 (data collection) → Crew 2 (analysis) → Crew 3 (report).
This is useful when you need different team compositions for each phase.
