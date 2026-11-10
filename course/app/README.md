# Interactive Companion App: Novasphere RAG Assistant

This application is the interactive companion to the **RAG From Scratch** course. It provides a visual, hands-on user interface for exploring and inspecting the complete question-answering pipeline built across the course notebooks.

---

## Features

- **Grounded Question Answering**: Query the fictional Novasphere Dynamics corpus with strict source citations.
- **Inspectable Context**: View the exact chunks retrieved by the vector store alongside end-to-end latency breakdowns.
- **Interactive Controls**:
  - **Top-K Slider**: Adjust how many chunks are fed to the model (1 to 5).
  - **Cross-Encoder Reranker Toggle**: Switch the reranker ON/OFF to witness how reranking improves precision.
  - **Temperature Slider**: Control model determinism.
- **Sample Presets**: Test pricing, technical error codes (`ERR_QUANTUM_TIMEOUT`), policy rules, and unanswerable questions with one click.
- **Provider Agnostic**: Runs against OpenAI, Anthropic, Ollama, or completely offline via the local fallback model.

---

## How to Run

1. Ensure the course virtual environment is active:
   ```bash
   # Windows
   .\.venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

2. (Optional) Configure your provider in `course/.env` (or leave unset to use the offline local model):
   ```bash
   RAG_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   ```

3. Launch the app:
   ```bash
   python course/app/app.py
   ```

4. Open your web browser at:
   ```
   http://127.0.0.1:7860
   ```
