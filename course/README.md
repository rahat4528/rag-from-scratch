# RAG From Scratch: Complete Beginner Course

Welcome to the hands-on **RAG From Scratch** course! This comprehensive, runnable curriculum takes you from knowing nothing about embeddings, vector databases, or LangChain to building, evaluating, and safely deploying production-grade Retrieval-Augmented Generation systems.

Every notebook is self-contained, adheres to a strict pedagogical structure, and can be executed top-to-bottom using either hosted cloud providers (OpenAI, Anthropic) or a **100% free, fully local and offline path**.

---

## Course Curriculum & Notebook Map

| # | Notebook Title | Promise | Est. Runtime | Key LangChain / RAG Concepts |
|---|---|---|---|---|
| **00** | [00. Why RAG?](notebooks/00_why_rag.ipynb) | Understand why LLMs hallucinate on private or recent data and how RAG solves it. | ~15 sec | Knowledge cutoffs, Context windows, Fine-tuning vs RAG |
| **01** | [01. The Big Picture](notebooks/01_rag_big_picture.ipynb) | Build your very first end-to-end RAG pipeline in under 15 lines of Python. | ~20 sec | Indexing, Retrieval, In-Context Generation, Grounding |
| **02** | [02. Loading Documents](notebooks/02_loading_documents.ipynb) | Ingest Text, Markdown, Web pages, and PDFs with rich metadata attachments. | ~15 sec | `TextLoader`, `PyPDFLoader`, `WebBaseLoader`, `Document` schema |
| **03** | [03. Splitting Text](notebooks/03_splitting_text.ipynb) | Master chunking strategies and character vs token boundary safety. | ~10 sec | `RecursiveCharacterTextSplitter`, Chunk size, Overlap |
| **04** | [04. Embeddings & Vector Stores](notebooks/04_embeddings_and_vector_stores.ipynb) | Understand semantic vector spaces, compute cosine similarity, and index in vector stores. | ~25 sec | Embeddings, Cosine similarity, `InMemoryVectorStore`, Persistence |
| **05** | [05. Retrieval](notebooks/05_retrieval.ipynb) | Retrieve relevant chunks using similarity, MMR diversity, and metadata filtering. | ~20 sec | Top-k, Maximal Marginal Relevance (MMR), Metadata filtering |
| **06** | [06. First RAG Chain](notebooks/06_generation_first_rag_chain.ipynb) | Assemble a clean LangChain Expression Language (LCEL) question-answering chain. | ~20 sec | `ChatPromptTemplate`, LCEL pipe operator `\|`, `StrOutputParser` |
| **07** | [07. Mini-Project: Own Files](notebooks/07_rag_with_your_own_files.ipynb) | Ingest an entire multi-format folder, persist the index, and evaluate accuracy. | ~45 sec | Multi-file batch crawler, Persistent retriever, Evaluation score |
| **08** | [08. Query Translation](notebooks/08_query_translation.ipynb) | Implement Multi-Query, RAG-Fusion (RRF), Decomposition, Step-Back, and HyDE. | ~30 sec | Multi-Query, Reciprocal Rank Fusion, Step-Back, HyDE |
| **09** | [09. Routing & Query Construction](notebooks/09_routing_and_query_construction.ipynb) | Route queries logically with Pydantic and semantically with zero-token prototypes. | ~25 sec | `with_structured_output`, Zero-LLM routing, Metadata filters |
| **10** | [10. Advanced Indexing](notebooks/10_advanced_indexing.ipynb) | Build Parent-Document retrievers, BM25 + dense hybrid search, and explore RAPTOR. | ~30 sec | Parent-Document, BM25Okapi, Hybrid search, ColBERT MaxSim |
| **11** | [11. Reranking & Compression](notebooks/11_reranking_and_compression.ipynb) | Implement two-stage "retrieve wide, then rerank" and sentence-level compression. | ~25 sec | Cross-encoders, Two-stage retrieval, Contextual compression |
| **12** | [12. Agentic & Self-Correcting RAG](notebooks/12_agentic_and_self_correcting_rag.ipynb) | Build autonomous agent loops using LangGraph for CRAG grading and self-reflection. | ~30 sec | `langgraph`, `StateGraph`, CRAG document grading, Self-RAG loops |
| **13** | [13. Evaluating RAG](notebooks/13_evaluating_rag.ipynb) | Measure Hit Rate@k, MRR, LLM-as-a-judge faithfulness, and run ablation studies. | ~35 sec | Hit Rate, Mean Reciprocal Rank (MRR), Faithfulness judge |
| **14** | [14. Production & Safety](notebooks/14_production_and_safety.ipynb) | Handle document lifecycle, embedding caches, prompt injection defense, and RBAC. | ~25 sec | Incremental updates, LRU caching, XML perimeter, Access control |
| **15** | [15. Capstone Project](notebooks/15_capstone_project.ipynb) | Assemble an enterprise-grade grounded assistant and achieve >=12/14 on benchmark. | ~40 sec | Hybrid two-stage RAG, Source citations, Benchmark validation |

---

## Learning Journey Visual Architecture

```mermaid
flowchart TD
    subgraph P1["Phase 1: Core RAG Pipeline"]
        N00["00. Why RAG?"] --> N01["01. The Big Picture"]
        N01 --> N02["02. Document Loading"]
        N02 --> N03["03. Text Splitting"]
        N03 --> N04["04. Embeddings & Vector Stores"]
        N04 --> N05["05. Retrieval Techniques"]
        N05 --> N06["06. First RAG Chain"]
        N06 --> N07["07. Mini-Project: Own Files"]
    end

    subgraph P2["Phase 2: Improving the Question"]
        N07 --> N08["08. Query Translation (Multi-Query, RRF, HyDE)"]
        N08 --> N09["09. Routing & Query Construction"]
    end

    subgraph P3["Phase 3: Advanced Indexing & Agents"]
        N09 --> N10["10. Advanced Indexing (Parent-Doc, Hybrid BM25)"]
        N10 --> N11["11. Reranking & Contextual Compression"]
        N11 --> N12["12. Agentic & Self-Correcting RAG (LangGraph)"]
    end

    subgraph P4["Phase 4: Evaluation, Safety & Shipping"]
        N12 --> N13["13. Evaluating RAG (Hit Rate, MRR, Judges)"]
        N13 --> N14["14. Production RAG & Security Guardrails"]
        N14 --> N15["15. Capstone Project (14/14 Benchmark)"]
        N15 --> APP["Interactive Gradio Web App (course/app)"]
    end
```

---

## Setup & Getting Started

### 1. Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/rahat4528/rag-from-scratch.git
cd rag-from-scratch

# Create and activate Python virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install all dependencies
pip install -r course/requirements.txt
```

### 2. Choose Your Execution Provider

Create a `.env` file in the `course/` folder (or copy `course/.env.example`):
```bash
cp course/.env.example course/.env
```

#### Option A: Hosted Cloud Providers (OpenAI or Anthropic)
Edit `course/.env`:
```env
# For OpenAI
RAG_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# For Anthropic (automatically paired with local sentence embeddings)
RAG_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
ANTHROPIC_CHAT_MODEL=claude-3-5-haiku-20241022
```

#### Option B: 100% Free Local Path (Ollama)
If you have [Ollama](https://ollama.ai) installed and running locally:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
Edit `course/.env`:
```env
RAG_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

#### Option C: 100% Offline Zero-API Fallback (Runs Anywhere)
Leave `RAG_PROVIDER=local` (or leave `.env` empty!). The shared helper module `course/utils/models.py` will automatically activate the local grounded chat model and local sentence-transformer embeddings. **No API keys, no internet connection, and no GPU required!**

---

## Running the Interactive Companion App

Launch the full-featured Gradio web app:
```bash
python course/app/app.py
```
Open your browser at `http://127.0.0.1:7860` to test questions, inspect retrieved chunks, and toggle cross-encoder reranking interactively!

---

## Troubleshooting FAQ

### 1. Windows ZMQ / DeadKernelError when running notebooks in batch
- **Cause**: On Windows with Python 3.8+, pyzmq requires the `WindowsSelectorEventLoopPolicy`.
- **Solution**: The course runner `course/utils/run_notebooks.py` automatically configures this policy and executes notebooks in isolated subprocesses. Run `python course/utils/run_notebooks.py course/notebooks/*.ipynb`.

### 2. Dimension Mismatch Error in Vector Search
- **Cause**: Indexing documents with one embedding model (e.g. 1536 dimensions) and querying with another (e.g. 384 dimensions).
- **Solution**: The Golden Rule of Embeddings: Always use the exact same embedding model for both indexing and query retrieval. If you change models, re-index your documents.

### 3. Rate Limit / 429 Errors with Paid APIs
- **Solution**: Set `temperature=0.0` and lower `k` (e.g. `k=2`). Alternatively, switch `RAG_PROVIDER=local` in `course/.env` to run with zero API limits.

### 4. Missing Optional Keys (Cohere, Tavily)
- **Solution**: All paid third-party features (e.g. Cohere reranking) include explicit safe guards. If the environment variable is missing, the code automatically falls back to our free local implementation without crashing.

---

## Glossary of Key Terms

- **RAG (Retrieval-Augmented Generation)**: An architecture that enriches an LLM's prompt with relevant factual chunks retrieved from external knowledge bases before generating an answer.
- **Embedding**: A numerical vector (e.g. 384 or 1536 floats) representing the semantic meaning of a piece of text.
- **Cosine Similarity**: A mathematical metric measuring the cosine of the angle between two vectors (1.0 = identical direction, 0.0 = orthogonal/unrelated).
- **Vector Store**: A specialized database optimized for fast approximate nearest neighbor (ANN) vector search.
- **Chunking**: The process of dividing long documents into smaller, overlapping passages suitable for embedding models and LLM context windows.
- **Top-K**: The number of most relevant candidate document chunks returned by a search query.
- **MMR (Maximal Marginal Relevance)**: An algorithm that balances relevance to the query with diversity among retrieved chunks, preventing redundant answers.
- **Multi-Query Expansion**: Prompting an LLM to generate multiple alternative phrasings of a user query to improve retrieval recall.
- **RRF (Reciprocal Rank Fusion)**: An algorithm that combines ranked lists from multiple search queries by summing reciprocal ranks: $\sum \frac{1}{k + \text{rank}_i}$.
- **HyDE (Hypothetical Document Embeddings)**: Generating a hypothetical ideal answer chunk before retrieving, and embedding that hypothetical answer to find real matching documents.
- **Parent-Document Retrieval**: Indexing small child chunks for precise vector search, while retrieving the larger parent document chunk for rich LLM generation.
- **Hybrid Search**: Combining lexical keyword matching (BM25) with semantic vector search.
- **Cross-Encoder**: A model that jointly evaluates query and document tokens with full cross-attention to output a precision relevance score.
- **Contextual Compression**: Extracting only the sentences directly relevant to the query from retrieved chunks to eliminate token bloat.
- **CRAG (Corrective RAG)**: An agentic pattern that grades retrieved documents and triggers query rewrites or web search fallbacks if chunks are irrelevant.
- **Self-RAG**: An agentic pattern where the model self-evaluates whether its generated answer is faithful to the context and directly resolves the user query.
- **Hit Rate@k**: The proportion of test questions where the correct document was included in the top-k retrieved chunks.
- **MRR (Mean Reciprocal Rank)**: The average of the reciprocal ranks of the first relevant document across all test queries.
- **Indirect Prompt Injection**: Malicious instructions embedded inside document text intended to hijack the LLM's system instructions.
- **RBAC (Role-Based Access Control)**: Enforcing user role permissions during document retrieval so unauthorized context is never surfaced.
