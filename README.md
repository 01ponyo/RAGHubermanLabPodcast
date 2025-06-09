# Science-based Tools For Everyday Life RAG System (Huberman’s Lab-based)

## Idea

Andrew Huberman is a professor of neurobiology and ophthalmology at Stanford School of Medicine. He holds a podcast series called Huberman’s Lab Podcasts, where he discusses science and science-based tools for everyday life, alone or with guests.

This project presents final Retrieval-Augmented Generation (RAG) system that enables users to ask questions about science-based health and performance insights from 106 Huberman Lab podcast episodes.

## Overview
This system processes Andrew Huberman's podcast transcripts and provides accurate, source-cited responses to user queries about neuroscience, protocols, supplements, and health optimization strategies.


## Key Features

- **Hybrid Retrieval**: Combines BM25 keyword search with FAISS semantic similarity
- **Parent chunks retrieval**: Retrieves surrounding chunks for better context preservation
- **LLM Reranking**: Uses GPT-3.5-turbo to rerank results first ranked by RRF
- **Source Attribution**: All responses include specific episode references to ensure there are no hallucinations
- **Interactive UI**: Gradio interface for easy querying

## Dataset

106 Huberman Lab podcast episode transcripts (EP002-EP107) covering topics like:

- Sleep optimization and circadian rhythms
- Neuroscience and brain function
- Exercise and performance protocols
- Nutrition and supplementation
- Stress management and mental health

## Architecture

Main features:

- **Data Processing**: Converts .docx transcripts into searchable chunks
- **Hybrid Indexing**: Builds both BM25 and FAISS indexes for hybrid search
- **Reciprocal Rank Fusion**: Combination of keyword and semantic search results
- **Parent Chunk Retrieval**: Expands context by including adjacent chunks
- **LLM Reranking**: Scores chunk relevance using language model reasoning
- **Response Generation**: Creates comprehensive answers with episode citations


## Quick start
### Prerequisites
- OpenAI API account

### Step-by-Step Setup

1. Clone repository
2. Create virtual environment: python -m venv .venv
3. Activate environment: source .venv/bin/activate
4. Install requirements: pip install -r requirements.txt
5. Get OpenAI API key
6. Create .env file: echo "OPENAI_API_KEY=your-key" > src/.env
7. Run: python src/gradio_interface.py


