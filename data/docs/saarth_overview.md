# SAARTH Overview

SAARTH is a retrieval-augmented, multi-agent AI assistant. The name comes from the Sanskrit word for "meaningful".

SAARTH uses five agents orchestrated with LangGraph: a planner, a tool agent, a retriever, a writer and a critic. The critic measures how well the answer is grounded in the retrieved passages and asks the retriever to search more widely when grounding is weak.

Documents are split into overlapping chunks of 800 characters with an overlap of 120 characters. Each chunk is embedded and stored in a vector store such as FAISS, Chroma or PostgreSQL with pgvector.

Retrieval is hybrid: dense vector similarity contributes 70 percent of the score and keyword overlap contributes 30 percent.
