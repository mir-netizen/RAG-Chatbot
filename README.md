# Retrieval-Augmented Generation (RAG) Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and query their contents interactively.

## Overview

The system processes uploaded PDFs through the following pipeline:

### 1. Text Extraction
The uploaded PDF is parsed to extract raw textual content.

### 2. Text Normalization & Tokenization
The extracted text is cleaned and normalized, then converted into token IDs for processing.

### 3. Chunking Strategy
The text is divided into smaller chunks:
- **Chunk size:** 300 tokens  
- **Overlap:** 50 tokens  

This overlap helps maintain context between chunks.

### 4. Embedding Generation
Each chunk is converted into vector embeddings using OpenAI’s embedding model.

### 5. Vector Storage (ChromaDB)
The embeddings are stored in **ChromaDB**, an open-source vector database optimized for similarity search.

### 6. Retrieval & Augmentation
When a user submits a query:
- Relevant chunks are retrieved from ChromaDB
- These chunks are used to augment the prompt

### 7. Response Generation
The augmented prompt is passed to **gpt-4o-mini**, which generates a context-aware response.
