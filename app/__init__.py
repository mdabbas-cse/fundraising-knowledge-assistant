"""Fundraising Knowledge Assistant - a Phase 1 RAG (Retrieval-Augmented Generation) baseline.

Pipeline:
    knowledge_base      -> load raw documents from disk
    document_processor  -> clean text and split documents into chunks
    retriever           -> TF-IDF + cosine similarity ranking
    prompt_builder      -> grounded, citation-aware LLM prompt
    answer_service      -> orchestrates the steps (LLM is optional / Phase 2)

All bundled documents are synthetic training data.
"""

__version__ = "0.1.0"
