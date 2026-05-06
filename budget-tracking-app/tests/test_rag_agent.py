"""
Tests for the RAG Financial Advisor Agent.

We test the KnowledgeBase class independently (without LLM calls)
to verify document loading and FAISS indexing work correctly.
LLM-dependent tests are marked with @pytest.mark.skipif.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, ".")

from services.agents.rag_agent import KnowledgeBase, _get_default_knowledge


class TestKnowledgeBase:
    """Tests for the knowledge base and FAISS index."""

    def test_default_knowledge_has_documents(self):
        """The fallback knowledge base should have content."""
        docs = _get_default_knowledge()
        assert len(docs) >= 5
        for doc in docs:
            assert "title" in doc
            assert "content" in doc
            assert "category" in doc

    def test_load_documents_uses_defaults_if_file_missing(self):
        """Should use defaults when JSON file doesn't exist."""
        kb = KnowledgeBase()
        with patch("services.agents.rag_agent.KNOWLEDGE_BASE_PATH") as mock_path:
            mock_path.exists.return_value = False
            kb.load_documents()

        assert len(kb.documents) > 0

    def test_build_index_creates_faiss_index(self):
        """FAISS index should be created with correct dimensions."""
        kb = KnowledgeBase()
        kb.documents = _get_default_knowledge()

        # Mock embeddings since we might not have an API key
        fake_dim = 128
        fake_embeddings = np.random.rand(len(kb.documents), fake_dim).astype("float32")

        with patch("services.agents.rag_agent._get_embeddings", return_value=fake_embeddings):
            with patch("services.agents.rag_agent.faiss.write_index"):
                kb.build_index()

        assert kb.index is not None
        assert kb.index.ntotal == len(kb.documents)
        assert kb.dimension == fake_dim

    def test_search_returns_correct_number_of_results(self):
        """Search should return exactly top_k results."""
        kb = KnowledgeBase()
        kb.documents = _get_default_knowledge()

        fake_dim = 128
        fake_embeddings = np.random.rand(len(kb.documents), fake_dim).astype("float32")

        with patch("services.agents.rag_agent._get_embeddings", return_value=fake_embeddings):
            with patch("services.agents.rag_agent.faiss.write_index"):
                kb.build_index()

        query_vector = np.random.rand(1, fake_dim).astype("float32")
        with patch("services.agents.rag_agent._get_embeddings", return_value=query_vector):
            results = kb.search("how to save money", top_k=3)

        assert len(results) == 3
        for r in results:
            assert "relevance_score" in r
            assert 0 <= r["relevance_score"] <= 1

    def test_search_returns_fewer_if_not_enough_docs(self):
        """If we ask for top_k=5 but only have 3 docs, return 3."""
        kb = KnowledgeBase()
        kb.documents = _get_default_knowledge()[:3]

        fake_dim = 64
        fake_embeddings = np.random.rand(3, fake_dim).astype("float32")

        with patch("services.agents.rag_agent._get_embeddings", return_value=fake_embeddings):
            with patch("services.agents.rag_agent.faiss.write_index"):
                kb.build_index()

        query_vector = np.random.rand(1, fake_dim).astype("float32")
        with patch("services.agents.rag_agent._get_embeddings", return_value=query_vector):
            results = kb.search("budgeting", top_k=5)

        assert len(results) <= 3


class TestDefaultKnowledge:
    """Verify the quality of the default knowledge base."""

    def test_covers_key_categories(self):
        """Knowledge base should cover core personal finance topics."""
        docs = _get_default_knowledge()
        categories = {doc["category"] for doc in docs}

        assert "budgeting" in categories
        assert "savings" in categories
        assert "inflation" in categories
        assert "spending" in categories

    def test_documents_are_substantial(self):
        """Each document should have meaningful content (not stubs)."""
        docs = _get_default_knowledge()
        for doc in docs:
            assert len(doc["content"]) > 50, f"Document '{doc['title']}' is too short"

    def test_unique_ids(self):
        """All document IDs should be unique."""
        docs = _get_default_knowledge()
        ids = [doc["id"] for doc in docs]
        assert len(ids) == len(set(ids)), "Duplicate document IDs found"
