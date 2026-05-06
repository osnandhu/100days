"""
RAG Financial Advisor Agent
============================
Retrieves relevant financial knowledge from a FAISS vector store, then
generates personalised advice using LiteLLM.

RAG = Retrieval-Augmented Generation. Instead of relying purely on the
LLM's training data, we:
  1. RETRIEVE relevant documents from our own knowledge base
  2. AUGMENT the LLM prompt with those documents as context
  3. GENERATE an answer grounded in our specific financial content

WHY RAG OVER FINE-TUNING:
- No model training needed (cheaper, faster)
- Knowledge base is updatable without retraining
- Sources are traceable (we know which doc informed the answer)
- Works with any LLM via LiteLLM

WHY FAISS:
- Facebook AI Similarity Search - blazing fast vector lookups
- Works entirely locally (no external vector DB service needed)
- Scales to millions of vectors on a single machine
- faiss-cpu is pip-installable, no GPU required
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Type

import faiss
import numpy as np
from pydantic import BaseModel, Field

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    from crewai import Agent, Task
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    BaseTool = object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent.parent / "data" / "financial_knowledge.json"
FAISS_INDEX_PATH = Path(__file__).parent.parent.parent / "data" / "faiss_index.bin"


# ---------------------------------------------------------------------------
# Knowledge Base Manager
# ---------------------------------------------------------------------------
# Handles loading documents, creating embeddings, and building the FAISS index.
# This is separate from the tools so it can be reused and tested independently.
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """
    Manages the financial knowledge base and FAISS vector index.

    HOW IT WORKS:
    1. Load documents from a JSON file (each doc has title + content)
    2. Generate embedding vectors for each document using LiteLLM
    3. Store vectors in a FAISS index for fast similarity search
    4. On query: embed the query → search FAISS → return top-k docs
    """

    def __init__(self):
        self.documents: List[dict] = []
        self.index: faiss.IndexFlatL2 = None
        self.dimension: int = 0

    def load_documents(self) -> None:
        """Load financial knowledge base from JSON file."""
        if not KNOWLEDGE_BASE_PATH.exists():
            logger.warning(f"Knowledge base not found at {KNOWLEDGE_BASE_PATH}")
            self.documents = _get_default_knowledge()
        else:
            with open(KNOWLEDGE_BASE_PATH) as f:
                self.documents = json.load(f)
        logger.info(f"Loaded {len(self.documents)} knowledge base documents")

    def build_index(self) -> None:
        """
        Create FAISS index from document embeddings.

        FAISS IndexFlatL2 does exact L2 (Euclidean) distance search.
        For <10k documents this is fast enough. For millions of docs,
        you'd switch to IndexIVFFlat (approximate but faster).
        """
        if not self.documents:
            self.load_documents()

        texts = [f"{doc['title']}: {doc['content']}" for doc in self.documents]

        try:
            embeddings = _get_embeddings(texts)
        except Exception as e:
            logger.error(f"Embedding failed: {e}. Using random vectors for demo.")
            embeddings = np.random.rand(len(texts), 256).astype("float32")

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={self.dimension}")

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Find the top_k most relevant documents for a query.

        STEPS:
        1. Embed the query text into a vector
        2. Search FAISS for nearest neighbours (L2 distance)
        3. Return the corresponding documents with distance scores
        """
        if self.index is None:
            self.build_index()

        try:
            query_vector = _get_embeddings([query])
        except Exception:
            logger.error("Query embedding failed, returning first documents")
            return self.documents[:top_k]

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["relevance_score"] = round(1 / (1 + float(dist)), 3)
                results.append(doc)

        return results


# Module-level instance so it's built once and reused across requests
_knowledge_base = KnowledgeBase()


def _get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embedding vectors using LiteLLM.

    LiteLLM is a unified interface that works with OpenAI, Anthropic,
    Cohere, HuggingFace, Ollama, and 100+ other providers. Changing
    EMBEDDING_MODEL in the env var switches providers without code changes.

    Falls back to simple hash-based vectors when LiteLLM is unavailable.
    """
    if LITELLM_AVAILABLE:
        try:
            response = litellm.embedding(model=EMBEDDING_MODEL, input=texts)
            vectors = [item["embedding"] for item in response.data]
            return np.array(vectors, dtype="float32")
        except Exception as e:
            logger.warning(f"LiteLLM embedding failed ({e}), using fallback")

    # Fallback: deterministic hash-based vectors (for demos without API key)
    dim = 128
    vectors = []
    for text in texts:
        np.random.seed(hash(text) % 2**31)
        vectors.append(np.random.rand(dim).astype("float32"))
    return np.array(vectors, dtype="float32")


def _get_default_knowledge() -> List[dict]:
    """Fallback knowledge base if the JSON file doesn't exist."""
    return [
        {
            "id": 1,
            "title": "The 50/30/20 Budget Rule",
            "content": (
                "Allocate 50% of after-tax income to needs (rent, utilities, "
                "groceries, insurance), 30% to wants (dining out, entertainment, "
                "hobbies), and 20% to savings and debt repayment. This is a "
                "simple framework that works for most income levels."
            ),
            "category": "budgeting",
        },
        {
            "id": 2,
            "title": "Emergency Fund Basics",
            "content": (
                "An emergency fund should cover 3-6 months of essential expenses. "
                "Keep it in a high-yield savings account for easy access. Start "
                "small: even $500 covers most minor emergencies. Build it before "
                "investing in stocks or crypto."
            ),
            "category": "savings",
        },
        {
            "id": 3,
            "title": "Managing Lifestyle Inflation",
            "content": (
                "When your income rises, resist the urge to increase spending "
                "proportionally. If you get a $500/month raise, save at least "
                "half of it. Lifestyle inflation is the #1 reason high earners "
                "still live paycheck to paycheck."
            ),
            "category": "budgeting",
        },
        {
            "id": 4,
            "title": "Inflation Impact on Savings",
            "content": (
                "If inflation is 4% and your savings account earns 2%, your "
                "money loses 2% purchasing power per year. Combat this by: "
                "1) Using high-yield savings accounts, 2) Investing in "
                "inflation-protected securities, 3) Diversifying into assets "
                "that historically beat inflation (equities, real estate)."
            ),
            "category": "inflation",
        },
        {
            "id": 5,
            "title": "Grocery Spending Optimisation",
            "content": (
                "Groceries are often the most flexible budget category. "
                "Strategies: 1) Meal plan weekly to avoid impulse buys, "
                "2) Buy store brands (30-50% cheaper), 3) Use cashback apps, "
                "4) Shop seasonal produce, 5) Buy in bulk for non-perishables. "
                "Target: keep groceries under 10-15% of take-home pay."
            ),
            "category": "spending",
        },
        {
            "id": 6,
            "title": "Debt Avalanche vs Snowball Method",
            "content": (
                "Avalanche: pay minimums on all debts, put extra toward the "
                "highest interest rate debt. Saves the most money mathematically. "
                "Snowball: pay off smallest balance first for quick wins. Better "
                "for motivation. Pick avalanche if you're disciplined, snowball "
                "if you need momentum."
            ),
            "category": "debt",
        },
        {
            "id": 7,
            "title": "Singapore CPF Optimisation",
            "content": (
                "CPF (Central Provident Fund) is Singapore's social security "
                "system. Maximise it by: 1) Voluntary contributions for tax "
                "relief (up to $8,000/year), 2) Transfer OA to SA for higher "
                "interest (4% vs 2.5%), 3) Use CPF Investment Scheme for "
                "potentially higher returns. Don't touch it unless buying a home."
            ),
            "category": "savings",
        },
        {
            "id": 8,
            "title": "Subscription Audit Strategy",
            "content": (
                "The average person spends $200+/month on subscriptions they "
                "barely use. Monthly audit: 1) List every recurring charge, "
                "2) Rate each 1-5 on actual usage, 3) Cancel anything below 3, "
                "4) Downgrade premium to basic where possible. Set a calendar "
                "reminder to audit quarterly."
            ),
            "category": "spending",
        },
        {
            "id": 9,
            "title": "Investing During High Inflation",
            "content": (
                "During high inflation: 1) Avoid holding excess cash, "
                "2) Consider TIPS (Treasury Inflation-Protected Securities), "
                "3) REITs can hedge against inflation via rising rents, "
                "4) Commodities often rise with inflation, 5) Keep investing "
                "regularly (dollar-cost averaging) regardless of inflation. "
                "Don't try to time the market."
            ),
            "category": "inflation",
        },
        {
            "id": 10,
            "title": "Building a Side Income",
            "content": (
                "If cutting expenses isn't enough, increase income: "
                "1) Freelance your professional skills (highest ROI), "
                "2) Sell unused items, 3) Tutoring or teaching, "
                "4) Part-time consulting. Even $500/month extra directed "
                "to savings compounds significantly over 5-10 years."
            ),
            "category": "income",
        },
    ]


# ---------------------------------------------------------------------------
# Tool: RAG Retrieval + Advice Generation
# ---------------------------------------------------------------------------


class AdviceQueryInput(BaseModel):
    question: str = Field(description="The user's financial question")
    user_context: str = Field(
        default="",
        description="Additional context about the user's financial situation",
    )


class FinancialAdvisorTool(BaseTool):
    """
    Retrieves relevant financial knowledge and generates personalised advice.

    THE RAG PIPELINE:
    1. User asks a question ("How do I save with high inflation?")
    2. We embed the question → search FAISS for relevant docs
    3. We build a prompt: system instructions + retrieved docs + question
    4. LiteLLM generates a grounded answer citing the retrieved docs
    """

    name: str = "financial_advisor_rag"
    description: str = (
        "Answers financial questions using a knowledge base of budgeting, "
        "saving, investing, and inflation strategies. Provides personalised "
        "advice grounded in retrieved documents."
    )
    args_schema: Type[BaseModel] = AdviceQueryInput

    def _run(self, question: str, user_context: str = "") -> str:
        # Step 1: Retrieve relevant documents
        relevant_docs = _knowledge_base.search(question, top_k=3)

        # Step 2: Build context from retrieved documents
        context_parts = []
        sources = []
        for doc in relevant_docs:
            context_parts.append(f"[{doc['title']}]: {doc['content']}")
            sources.append(doc["title"])

        retrieved_context = "\n\n".join(context_parts)

        # Step 3: Build the RAG prompt
        system_prompt = (
            "You are a certified financial advisor specialising in personal "
            "budgeting for Singapore residents. Give specific, actionable advice. "
            "Base your answer on the provided knowledge base documents. "
            "If the documents don't cover the question, say so honestly."
        )

        user_prompt = (
            f"KNOWLEDGE BASE CONTEXT:\n{retrieved_context}\n\n"
            f"USER SITUATION:\n{user_context}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide specific, actionable advice based on the above context."
        )

        # Step 4: Generate answer via LiteLLM (or fallback to document summary)
        if LITELLM_AVAILABLE:
            try:
                response = litellm.completion(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                advice = response.choices[0].message.content
            except Exception as e:
                advice = (
                    f"LLM generation failed ({e}). Based on retrieved documents, "
                    f"key advice areas: {', '.join(sources)}"
                )
        else:
            advice = (
                "Based on retrieved knowledge base documents:\n\n"
                + "\n\n".join(
                    f"- **{doc['title']}**: {doc['content'][:150]}..."
                    for doc in relevant_docs
                )
            )

        return json.dumps(
            {
                "question": question,
                "advice": advice,
                "sources": sources,
                "documents_retrieved": len(relevant_docs),
            }
        )


# ---------------------------------------------------------------------------
# Agent + Task Factory
# ---------------------------------------------------------------------------


def create_rag_agent(llm):
    if not CREWAI_AVAILABLE:
        raise ImportError("crewai is required to create agents. pip install crewai")
    return Agent(
        role="Personalised Financial Advisor",
        goal=(
            "Provide context-aware, personalised financial advice by retrieving "
            "relevant knowledge and synthesising it with the user's specific situation."
        ),
        backstory=(
            "You are a certified financial planner with expertise in Singapore's "
            "financial landscape including CPF, HDB, and local banking products. "
            "You combine textbook knowledge with practical advice. You never "
            "recommend risky investments without clearly stating the risks."
        ),
        tools=[FinancialAdvisorTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_rag_task(agent, question: str, user_context: str = ""):
    if not CREWAI_AVAILABLE:
        raise ImportError("crewai is required to create tasks. pip install crewai")
    return Task(
        description=(
            f"The user asks: '{question}'. "
            f"User context: {user_context or 'No additional context provided.'}. "
            "Use the financial advisor RAG tool to retrieve relevant knowledge "
            "and provide personalised, actionable advice."
        ),
        expected_output=(
            "A personalised financial recommendation containing:\n"
            "1. Direct answer to the user's question\n"
            "2. Specific action steps (numbered)\n"
            "3. Sources from the knowledge base that informed the advice\n"
            "4. Any caveats or risks to be aware of"
        ),
        agent=agent,
    )


def initialize_knowledge_base():
    """Call this at startup to pre-build the FAISS index."""
    _knowledge_base.load_documents()
    _knowledge_base.build_index()
    return _knowledge_base
