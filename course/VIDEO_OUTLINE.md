# RAG From Scratch: Complete 16-Video Playlist Outline

This document provides a comprehensive production outline for recording a 16-video hands-on tutorial playlist accompanying the **RAG From Scratch** course. Each video maps directly to one runnable notebook in `course/notebooks/`.

---

## Playlist Overview

- **Series Title**: RAG From Scratch: Build Production AI Document Assistants
- **Total Videos**: 16 Episodes (00 to 15)
- **Target Audience**: Python developers, AI enthusiasts, and software engineers looking to master RAG from fundamentals to production systems.
- **Repository**: [https://github.com/rahat4528/rag-from-scratch](https://github.com/rahat4528/rag-from-scratch)

---

## Episode 00: Why Do Language Models Need RAG?

- **Duration Estimate**: 10 minutes
- **Audience Level**: Absolute Beginner
- **Key Visual / Diagram**:
  - The Knowledge Cutoff Cliff: Side-by-side comparison of fixed pre-trained model weights vs real-time dynamic retrieval grounding.
- **Timestamped Chapter Outline**:
  - `00:00 - The Illusion of Omniscience`: Why ChatGPT doesn't know your company's internal policies.
  - `02:15 - Three Critical LLM Failure Modes`: Knowledge cutoff dates, hallucinations, and private data isolation.
  - `05:30 - Why Fine-Tuning Fails for Factual Recall`: Fine-tuning is for style and form, not facts; catastrophic forgetting and high GPU retraining costs.
  - `08:00 - The RAG Revolution`: How In-Context Learning (ICL) turns closed-book exams into open-book exams.
  - `09:30 - Course Roadmap`: What we will build across the next 15 episodes.
- **Common Beginner Mistake to Highlight**: Thinking fine-tuning is the way to teach an LLM private company documents.
- **Hands-on Coding Demo**: Querying a model about a fictional company and watching it fabricate facts vs providing reference context.
- **End-of-Video Exercise**: Pick a recent news event from last week and test whether your favorite LLM can answer without RAG.

---

## Episode 01: The Big Picture: RAG in 15 Lines of Code

- **Duration Estimate**: 12 minutes
- **Audience Level**: Beginner
- **Key Visual / Diagram**:
  - The Core RAG Triangle: Ingestion -> Vector Search -> Grounded In-Context Generation.
- **Timestamped Chapter Outline**:
  - `00:00 - The Complete RAG Lifecycle`: High-level architectural walkthrough.
  - `02:30 - Loading & Chunking`: Breaking documents into bite-sized passages.
  - `05:15 - Vectors & Similarity`: How meaning becomes numbers.
  - `07:45 - The Grounded Prompt`: Directing the LLM to use ONLY provided facts.
  - `10:00 - Running the 15-Line Pipeline`: From zero to grounded answer in 10 seconds.
- **Common Beginner Mistake to Highlight**: Stuffing the entire raw 50-page document into the prompt instead of retrieving relevant chunks.
- **Hands-on Coding Demo**: Writing a minimal RAG pipeline using `InMemoryVectorStore` and `ChatPromptTemplate` in one cell.
- **End-of-Video Exercise**: Change the question to ask about an unmentioned policy and inspect the model's refusal behavior.

---

## Episode 02: Loading Documents: Text, Markdown, Web & PDFs

- **Duration Estimate**: 14 minutes
- **Audience Level**: Beginner
- **Key Visual / Diagram**:
  - Document Object Anatomy: `page_content: str` + `metadata: dict(source, author, date, page)`.
- **Timestamped Chapter Outline**:
  - `00:00 - Ingestion is Where Most Systems Fail`: Garbage in, garbage out.
  - `02:20 - TextLoader & MarkdownLoader`: Extracting structured content.
  - `05:00 - WebBaseLoader & HTML Cleaning`: Stripping navigation bars and cookie popups.
  - `07:30 - PyPDFLoader`: Extracting text and page number metadata from PDFs.
  - `10:15 - Standardizing Metadata`: Why downstream filtering requires uniform metadata tags.
  - `12:30 - Inspecting Loaded Documents`: Verifying clean ingestion.
- **Common Beginner Mistake to Highlight**: Forgetting to attach metadata during loading, making citations impossible later.
- **Hands-on Coding Demo**: Loading `company_handbook.txt`, `recent_changelog.md`, and `policy_brief.pdf` into standardized `Document` objects.
- **End-of-Video Exercise**: Write a small loop to load all `.txt` files in a folder and attach file size to metadata.

---

## Episode 03: Splitting Text: Chunk Sizes & Boundary Safety

- **Duration Estimate**: 15 minutes
- **Audience Level**: Beginner
- **Key Visual / Diagram**:
  - Character Splitting vs Recursive Splitting: Visualizing how sentences break across chunk boundaries and how overlap protects context.
- **Timestamped Chapter Outline**:
  - `00:00 - Why We Must Chunk`: Embedding context limits and retrieval granularity.
  - `02:45 - The Naive Character Splitter Trap`: Slicing mid-word and breaking sentences.
  - `05:30 - RecursiveCharacterTextSplitter Explained`: The `["\n\n", "\n", " ", ""]` hierarchy.
  - `08:15 - Tuning Chunk Size & Chunk Overlap`: Why 10-20% overlap is the industry sweet spot.
  - `11:00 - Chunking Markdown with Header Awareness`: Preserving section headers.
  - `13:30 - Verifying Chunk Integrity`: Inspecting split documents.
- **Common Beginner Mistake to Highlight**: Setting chunk overlap to 0, which cuts sentences in half right at the boundary.
- **Hands-on Coding Demo**: Splitting a multi-page handbook into 380-character chunks with 40-character overlap and inspecting boundary overlaps.
- **End-of-Video Exercise**: Experiment with chunk sizes of 100 vs 500 characters on a complex policy and observe where sentences split.

---

## Episode 04: Embeddings & Vector Stores: Meaning as Numbers

- **Duration Estimate**: 18 minutes
- **Audience Level**: Beginner-to-Intermediate
- **Key Visual / Diagram**:
  - 2D Embedding Scatter Plot: Showing semantic clustering of puppy sentences vs canine sentences vs database server sentences.
- **Timestamped Chapter Outline**:
  - `00:00 - What is an Embedding?`: Translating human concepts into high-dimensional coordinate space.
  - `03:00 - Cosine Similarity by Hand`: The dot product and vector norms in NumPy.
  - `06:30 - Visualizing Embeddings in 2D`: PCA projection and Matplotlib scatter plotting.
  - `09:45 - In-Memory Vector Stores`: LangChain's `InMemoryVectorStore`.
  - `12:30 - Disk Persistence with Dump and Load`: Saving index files locally.
  - `15:15 - The Golden Rule of Embedding Models`: Why you can NEVER query with a model different from your indexer.
- **Common Beginner Mistake to Highlight**: Indexing documents with OpenAI text-embedding-3-small and trying to search with sentence-transformers.
- **Hands-on Coding Demo**: Calculating cosine similarity by hand between 3 sentences, plotting them with Matplotlib, and dumping a vector index to disk.
- **End-of-Video Exercise**: Calculate cosine similarity between two sentences of your own choosing using pure NumPy.

---

## Episode 05: Retrieval: Top-k, MMR & Metadata Filtering

- **Duration Estimate**: 16 minutes
- **Audience Level**: Intermediate
- **Key Visual / Diagram**:
  - Top-k Redundancy vs MMR Diversity: Showing how MMR rejects redundant duplicate chunks in favor of diverse angles.
- **Timestamped Chapter Outline**:
  - `00:00 - The Retrieval Bottleneck`: Why more chunks isn't always better.
  - `02:30 - Tuning Top-k`: Balancing recall against prompt noise.
  - `05:15 - Maximal Marginal Relevance (MMR)`: Balancing relevance ($\lambda$) and diversity ($1-\lambda$).
  - `08:30 - Metadata Filtering`: Combining semantic search with exact document filters.
  - `11:45 - Inspecting Distance Scores`: When to reject chunks below a similarity threshold.
  - `14:00 - Choosing the Right Retriever Settings`: Production guidelines.
- **Common Beginner Mistake to Highlight**: Always setting $k=10$ and wondering why the model produces unfocused, contradictory answers.
- **Hands-on Coding Demo**: Comparing pure similarity search with MMR search on a clustered document corpus.
- **End-of-Video Exercise**: Set $\lambda = 0.2$ in MMR and see how diverse the retrieved chunks become.

---

## Episode 06: Generation: Building Your First LCEL RAG Chain

- **Duration Estimate**: 15 minutes
- **Audience Level**: Intermediate
- **Key Visual / Diagram**:
  - LCEL Pipeline Stream: `{"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`.
- **Timestamped Chapter Outline**:
  - `00:00 - Moving Beyond Manual Gluing`: Why production needs declarative pipelines.
  - `02:15 - The Anatomy of a RAG System Prompt`: Roles, context constraints, refusal directives.
  - `05:30 - LangChain Expression Language (LCEL)`: The Unix pipe operator `|` for AI components.
  - `08:45 - Source Citation Extraction`: Forcing the model to cite bracketed document tags.
  - `11:30 - Streaming Answers in Real Time`: Lowering time-to-first-token.
  - `13:45 - Refusal Testing`: Verifying that out-of-scope questions trigger "I don't know".
- **Common Beginner Mistake to Highlight**: Omitting the explicit refusal instruction (*"If context does not contain the answer, say I don't know"*), allowing the LLM to hallucinate from pre-trained memory.
- **Hands-on Coding Demo**: Assembling an LCEL RAG chain from scratch and streaming citations.
- **End-of-Video Exercise**: Modify the prompt to format citations as numbered footnotes at the bottom of the answer.

---

## Episode 07: Mini-Project: RAG With Your Own Files

- **Duration Estimate**: 20 minutes
- **Audience Level**: Intermediate
- **Key Visual / Diagram**:
  - Mini-Project Architecture: Multi-format Folder Crawler -> Splitter -> Vector Index -> QA Function -> Benchmark Runner.
- **Timestamped Chapter Outline**:
  - `00:00 - Phase 1 Milestone`: Packaging everything into a production module.
  - `02:30 - Writing the Multi-Format Ingestion Crawler`: Auto-detecting file types.
  - `06:00 - Indexing and Disk Persistence`: Saving the index to disk.
  - `09:15 - Reloading Without Re-Embedding`: Simulating fast server startup.
  - `12:30 - Executing the 14 Benchmark Questions`: Automated accuracy testing.
  - `16:45 - Troubleshooting Decision Tree`: Debugging retrieval vs generation errors.
- **Common Beginner Mistake to Highlight**: Re-indexing all documents every time the backend server restarts.
- **Hands-on Coding Demo**: Running the complete automated test suite across all 14 reference evaluation questions.
- **End-of-Video Exercise**: Add a new `.md` file to the corpus and re-run the benchmark to verify incremental ingestion.

---

## Episode 08: Query Translation: Multi-Query, RAG-Fusion & HyDE

- **Duration Estimate**: 22 minutes
- **Audience Level**: Intermediate-to-Advanced
- **Key Visual / Diagram**:
  - RRF Scoring Matrix: Hand-worked table showing how reciprocal ranks promote consensus documents across multiple query angles.
- **Timestamped Chapter Outline**:
  - `00:00 - The Vocabulary Mismatch Problem`: Why user queries miss document chunks.
  - `03:00 - Multi-Query Expansion`: Generating 3 perspectives with an LLM.
  - `06:30 - The Math of Reciprocal Rank Fusion (RRF)`: Step-by-step numerical example ($k=60$).
  - `10:15 - Query Decomposition`: Handling multi-part compound questions.
  - `13:45 - Step-Back Prompting`: Retrieving high-level foundational concepts.
  - `16:30 - Hypothetical Document Embeddings (HyDE)`: Document-to-document vector search.
  - `19:15 - Architectural Trade-Off Table`: Comparing latency, recall, and token cost.
- **Common Beginner Mistake to Highlight**: Adding raw cosine similarity scores from different queries together instead of using rank-based fusion (RRF).
- **Hands-on Coding Demo**: Writing the `reciprocal_rank_fusion` algorithm and comparing Multi-Query vs HyDE results.
- **End-of-Video Exercise**: Calculate RRF scores by hand for 3 documents across 2 query rankings with $k=20$.

---

## Episode 09: Routing & Query Construction: Directing Queries

- **Duration Estimate**: 18 minutes
- **Audience Level**: Intermediate-to-Advanced
- **Key Visual / Diagram**:
  - Partitioned Routing Tree: Logical Router (Pydantic) + Semantic Router (Zero-token cosine prototypes).
- **Timestamped Chapter Outline**:
  - `00:00 - The Monolithic Vector Store Anti-Pattern`: Context pollution and slow searches.
  - `02:30 - Partitioned Knowledge Stores`: Splitting by department and doc type.
  - `05:15 - Logical Routing with Pydantic`: Type-safe schema routing with graceful fallbacks.
  - `08:45 - Zero-LLM Semantic Routing`: Sub-millisecond routing with embedding prototypes.
  - `12:15 - Query Construction`: Extracting scalar filters from natural language queries.
  - `15:30 - Filtered Vector Search Execution`: Executing filtered retrievals.
- **Common Beginner Mistake to Highlight**: Calling a slow, expensive LLM for routing when a 1ms embedding prototype comparison produces identical routing accuracy.
- **Hands-on Coding Demo**: Building a zero-LLM semantic router that routes HR vs Technical questions in 2 milliseconds.
- **End-of-Video Exercise**: Add a 4th route for `legal_contracts` and verify query routing with a sample NDA query.

---

## Episode 10: Advanced Indexing: Parent-Document & Hybrid Search

- **Duration Estimate**: 20 minutes
- **Audience Level**: Advanced
- **Key Visual / Diagram**:
  - Parent-Child Chunk Mapping: Small children pointing to large parent chunks; BM25 sparse + Dense vector fusion.
- **Timestamped Chapter Outline**:
  - `00:00 - The Chunk Size Trade-Off Revisited`: High search precision vs rich generation context.
  - `03:00 - Parent-Document Architecture`: Indexing child chunks, retrieving parent chunks.
  - `06:45 - BM25 Sparse Keyword Search`: Why dense vectors fail on error codes and serial numbers.
  - `10:15 - Hybrid Search with Reciprocal Rank Fusion`: Uniting sparse and dense rankings.
  - `13:45 - RAPTOR Conceptual Demo`: Recursive summary trees for abstract questions.
  - `16:30 - ColBERT Late Interaction`: Token-level MaxSim matching without heavy cross-encoders.
- **Common Beginner Mistake to Highlight**: Relying solely on dense embeddings for software error codes like `ERR_QUANTUM_TIMEOUT`.
- **Hands-on Coding Demo**: Implementing Parent-Document retrieval and benchmarking Hybrid Search against dense-only search.
- **End-of-Video Exercise**: Create a query with an exact alphanumeric code and verify that BM25 retrieves the correct document at Rank #1.

---

## Episode 11: Reranking & Contextual Compression

- **Duration Estimate**: 18 minutes
- **Audience Level**: Advanced
- **Key Visual / Diagram**:
  - Two-Stage Retrieval Funnel: 50,000 chunks -> Fast Bi-Encoder (Top-15) -> Cross-Encoder Reranker (Top-3) -> Contextual Compression.
- **Timestamped Chapter Outline**:
  - `00:00 - The Two-Stage Paradigm`: Why over-retrieving and reranking is industry standard.
  - `02:30 - Bi-Encoders vs Cross-Encoders`: Full cross-attention mechanics explained.
  - `06:00 - Implementing Local Cross-Encoder Reranking`: Scoring candidate chunks locally.
  - `09:30 - Optional Managed Rerankers (Cohere)`: Enterprise cloud reranking with safe guards.
  - `12:15 - Contextual Compression`: Dropping irrelevant sentences to combat "Lost in the Middle".
  - `15:30 - Measuring Token Reduction`: Saving 80% of context tokens without losing key facts.
- **Common Beginner Mistake to Highlight**: Trying to run a heavy cross-encoder over all 100,000 documents instead of using it as a second-stage filter on the top 15-20 candidates.
- **Hands-on Coding Demo**: Building a local cross-encoder reranker and sentence-level compression pipeline.
- **End-of-Video Exercise**: Profile the token count before and after contextual compression on a 500-word retrieved passage.

---

## Episode 12: Agentic & Self-Correcting RAG with LangGraph

- **Duration Estimate**: 24 minutes
- **Audience Level**: Advanced
- **Key Visual / Diagram**:
  - LangGraph State Machine: Nodes (`retrieve`, `grade_docs`, `generate`, `rewrite`), conditional edges, and loop counter bounds.
- **Timestamped Chapter Outline**:
  - `00:00 - Why Static Pipelines Fail`: The need for autonomous error correction.
  - `03:00 - Corrective RAG (CRAG)`: Document grading and query rewrite triggers.
  - `06:30 - Self-RAG & Hallucination Checks`: Validating factual grounding before answering.
  - `10:00 - Defining LangGraph State`: Building the `GraphState` TypedDict.
  - `13:15 - Implementing Nodes & Conditional Edges`: Wiring the state graph.
  - `17:00 - Guarding Against Infinite Loops`: Enforcing `max_loops = 3`.
  - `20:30 - Tracing Autonomous Execution`: Watching the agent self-correct in real time.
- **Common Beginner Mistake to Highlight**: Writing an agentic loop without a strict loop counter guard, causing runaway API costs on impossible queries.
- **Hands-on Coding Demo**: Compiling and invoking a full LangGraph self-correcting agent on ambiguous queries.
- **End-of-Video Exercise**: Add a mock web search fallback node to the graph that triggers when `loop_count >= 3`.

---

## Episode 13: Evaluating RAG: Hit Rate, MRR & LLM-as-a-Judge

- **Duration Estimate**: 20 minutes
- **Audience Level**: Advanced
- **Key Visual / Diagram**:
  - The Evaluation Dashboard: Hit Rate@k curve, MRR bar chart, and LLM-as-a-judge faithfulness scoring matrix.
- **Timestamped Chapter Outline**:
  - `00:00 - If You Can't Measure It, You Can't Ship It`: The necessity of offline evaluation.
  - `02:30 - Retrieval Metrics`: Defining Hit Rate@k and Mean Reciprocal Rank (MRR).
  - `06:00 - Automated Generation Evaluation`: Using an LLM as an impartial judge for faithfulness.
  - `09:30 - Designing an Ablation Study`: Testing chunk size, top-k, and reranking configurations.
  - `13:15 - Running the Experiment`: Computing metrics across all 14 benchmark questions.
  - `16:45 - Reading the Ablation Summary Table`: Identifying the winning architecture.
- **Common Beginner Mistake to Highlight**: Evaluating RAG only by eyeballing 2 or 3 queries in a chat UI rather than running a standardized benchmark dataset.
- **Hands-on Coding Demo**: Running an ablation study across 3 configurations and printing a formatted comparison table.
- **End-of-Video Exercise**: Compute Precision@k for your retrieval pipeline and compare it against Hit Rate@k.

---

## Episode 14: Production RAG & Security Guardrails

- **Duration Estimate**: 22 minutes
- **Audience Level**: Advanced
- **Key Visual / Diagram**:
  - Security Perimeter Architecture: XML delimiter containment, embedding LRU cache, and Role-Based Access Control (RBAC).
- **Timestamped Chapter Outline**:
  - `00:00 - Taking RAG to Production`: Non-functional requirements that make or break apps.
  - `02:45 - Incremental Document Updates & Deletions`: Updating without rebuilding.
  - `06:00 - Building an In-Memory Embedding Cache`: 100x query speedups with zero token cost.
  - `09:15 - Indirect Prompt Injection Attacks`: How untrusted documents hijack LLMs.
  - `12:45 - Implementing Perimeter Defenses`: XML isolation and strict system framing.
  - `15:30 - Role-Based Access Control (RBAC)`: Document-level security filtering.
  - `18:15 - When NOT to Use RAG`: The strategic decision matrix (Fine-Tuning vs RAG vs SQL).
- **Common Beginner Mistake to Highlight**: Inserting raw retrieved document text into the prompt without delimiter isolation, exposing the system to prompt injection.
- **Hands-on Coding Demo**: Injecting a malicious override into a document chunk, demonstrating model takeover, and mitigating it with XML perimeter framing.
- **End-of-Video Exercise**: Implement a disk-backed JSON cache for computed embeddings.

---

## Episode 15: Capstone Project: Enterprise QA Assistant & Graduation

- **Duration Estimate**: 25 minutes
- **Audience Level**: Advanced / Comprehensive
- **Key Visual / Diagram**:
  - Complete Enterprise Architecture: Ingestion -> Two-Stage Hybrid Search -> Cross-Encoder Rerank -> Guarded Prompt -> Verified Answer with Citations -> Companion App.
- **Timestamped Chapter Outline**:
  - `00:00 - Capstone Introduction`: Assembling all course patterns into one production assistant.
  - `03:00 - End-to-End Ingestion & Hybrid Indexing`: Crawling Markdown, Text, and PDFs.
  - `06:45 - Two-Stage Retrieval Execution`: Fast hybrid search + cross-encoder reranker.
  - `10:15 - Grounded Answering with Citations`: Extracting source tags and handling refusals.
  - `13:30 - Running the 14-Question Benchmark`: Scoring 100% (14/14) on the evaluation set.
  - `17:00 - Tour of the Gradio Companion App`: Testing the interactive web interface in `course/app/`.
  - `20:45 - Course Graduation & Production Checklist`: Best practices for shipping real RAG systems.
- **Common Beginner Mistake to Highlight**: Shipping to production without end-to-end citation verification and refusal guardrails.
- **Hands-on Coding Demo**: Running the capstone pipeline, achieving a 100% benchmark score, and launching the interactive Gradio web application.
- **End-of-Video Exercise**: Fork the repository, index your own personal or company documents, and deploy the Gradio app!
