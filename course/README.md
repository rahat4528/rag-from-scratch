# RAG From Scratch - Beginner Hands-On Course

A hands-on, runnable Jupyter notebook course designed for learners who want to master Retrieval-Augmented Generation (RAG) from the ground up.

## Quick Setup
1. Clone the repository and enter the course folder:
   ```bash
   git clone https://github.com/langchain-ai/rag-from-scratch.git
   cd rag-from-scratch
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r course/requirements.txt
   ```
4. Configure your environment:
   ```bash
   cp course/.env.example .env
   ```
   By default, the course runs in local mode (100% free, runs offline on CPU). If you have an OpenAI or Anthropic API key, or a local Ollama server, set `RAG_PROVIDER=openai` (or `anthropic` or `ollama`) in `.env`.

5. Launch Jupyter:
   ```bash
   jupyter notebook course/notebooks
   ```
