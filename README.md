# Intelligent Document Assistant

An AI-powered document question-answering and summarisation application built using Retrieval-Augmented Generation (RAG).

## Overview

The Intelligent Document Assistant allows users to upload PDF documents, ask questions based on their contents, and generate summaries using a locally running Large Language Model.

It uses semantic search and vector embeddings to retrieve relevant document information before generating answers.

## Features

- PDF document upload and management
- PDF text extraction with page-level metadata
- Text cleaning and overlapping chunk generation
- Semantic search using Sentence Transformers
- Vector storage and retrieval using ChromaDB
- Context-based question answering
- Source citations with PDF filenames and page numbers
- Hallucination control for unsupported questions
- Short and detailed document summaries
- Section-wise summarisation
- Chat history and clear-chat functionality
- Duplicate document protection
- Local LLM integration using Ollama

## Technology Stack

- Python
- Streamlit
- Sentence Transformers
- ChromaDB
- Ollama
- Qwen 2.5 3B
- PyMuPDF
- NumPy

## Architecture

1. Upload PDF documents.
2. Extract text page by page.
3. Clean and split text into overlapping chunks.
4. Generate vector embeddings.
5. Store embeddings and metadata in ChromaDB.
6. Convert user questions into embeddings.
7. Retrieve relevant document chunks using semantic search.
8. Pass retrieved context to the local LLM.
9. Generate answers with source citations.

## Project Structure

```text
Intelligent Document Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── modules/
│   ├── __init__.py
│   ├── pdf_extractor.py
│   ├── day2.py
│   ├── day3.py
│   ├── embedding_generator.py
│   ├── vector_database.py
│   ├── retrival.py
│   ├── llm.py
│   └── rag_pipeline.py
│
└── chroma_db/
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd intelligent-document-assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama and download the model:

```bash
ollama pull qwen2.5:3b
```

Ensure the Ollama service is running.

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Important Notes

- The application uses a locally running Ollama model.
- Uploaded PDFs and the local ChromaDB database are excluded from GitHub.
- A fresh installation requires uploading documents again to build its own vector database.
- Public deployment requires an accessible LLM inference service and suitable persistent vector storage.

## Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation
- Natural Language Processing
- Text embeddings
- Vector databases
- Semantic information retrieval
- Local Large Language Models
- Streamlit application development

## Author

Developed as a hands-on AI and Data Science project.