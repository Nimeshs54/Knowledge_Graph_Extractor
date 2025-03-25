# Knowledge Graph Research Paper or DOC Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![Flask](https://img.shields.io/badge/Flask-2.3-green.svg) ![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-orange.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A scalable AI-driven solution for extracting Knowledge Graphs (KGs) from research papers and enabling contextual querying via a chatbot interface. This project leverages Large Language Models (LLMs) such as LLaMA 3.2 and DeepSeek R1, integrated with LangChain and Neo4j Aura, to process full PDF documents, construct structured KGs, and provide accurate, real-time responses.

## Overview
The Knowledge Graph Research Paper Analysis tool transforms unstructured research papers into structured Knowledge Graphs, enabling users to query document content interactively. By combining advanced LLMs with graph database technology, this project addresses challenges like hallucination in generative AI and enhances insight generation from complex documents. It’s designed for researchers, data scientists, and AI enthusiasts seeking to explore paper content efficiently.

## Features
- **Full Document Processing**: Extracts entities and relationships from entire PDFs, not just limited sections, using text chunking.
- **Knowledge Graph Construction**: Builds KGs with Neo4j Aura, storing triples (subject-predicate-object) for structured data representation.
- **LLM Integration**: Supports LLaMA 3.2 (local) and DeepSeek R1 (via Groq API) for triple extraction and querying.
- **Chatbot Interface**: Professional Bootstrap-based UI for real-time, multi-turn conversations with the KG.
- **Scalable Deployment**: Deployable on cloud platforms (e.g., AWS) with Docker support.
- **Extensible**: Modular design allows integration of additional LLMs or graph databases.

## Technologies
- **Python**: 3.9+
- **Flask**: Web framework for the chatbot backend
- **LangChain**: LLM orchestration and prompt management
- **Neo4j Aura**: Cloud-hosted graph database for KG storage
- **LLaMA 3.2**: Local LLM for triple extraction and querying (via Ollama)
- **DeepSeek R1**: Cloud-based LLM for enhanced performance (via Groq API)
- **RDFlib**: RDF graph manipulation and OWL reasoning
- **Bootstrap 5**: Frontend styling for a modern UI
- **PyPDF2**: PDF text extraction
- **Docker**: Containerization (optional)

## Installation

### Prerequisites
- Python 3.9 or higher
- Neo4j Aura account ([sign up here](https://neo4j.com/cloud/aura/))
- Ollama installed locally for LLaMA 3.2 ([instructions](https://ollama.ai/))
- Groq API key for DeepSeek R1 ([get one here](https://groq.com/))

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nimeshs54/Knowledge_Graph_Extractor.git
   cd Knowledge_Graph_Extractor
   ```

2. **Install Dependencies**:
    ```bash
   pip install -r requirements.txt
   ```

3. **Run Ollama Server (for LLaMA)**:
    ```bash
    ollama serve
    ollama pull llama3.2
   ```

4. **Launch the Application**:
    ```bash
   python app.py
   ```