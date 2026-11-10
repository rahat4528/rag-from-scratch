"""
Builder for Phase 1 - Notebooks 00 to 03.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from course.utils.notebook_builder import create_course_notebook

NOTEBOOKS_DIR = Path("course/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_nb00():
    create_course_notebook(
        title="00. Why RAG? The Limits of LLMs and the Retrieval Fix",
        promise="By the end of this notebook, you will understand why standard AI models fail on private or recent knowledge, see hallucinations first-hand, and grasp the core mechanism of Retrieval-Augmented Generation.",
        objectives=[
            "Explain the knowledge cutoff and why models hallucinate when asked about private corporate facts.",
            "Contrast prompt-stuffing, fine-tuning, and RAG across cost, latency, update speed, and reliability.",
            "Articulate the one-sentence definition of RAG and recognize the open-book vs closed-book paradigm."
        ],
        prerequisites=[],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.models import get_llm, detect_provider
from course.utils.helpers import get_data_dir

llm = get_llm(temperature=0.0)
print(f"Active Provider: {detect_provider()}")
print(f"Loaded Chat Model: {llm.__class__.__name__}")""",
        concept_md="""### The Problem: LLMs Are Brilliant but Blind to Your Private Data

Large Language Models (LLMs) like GPT-4, Claude, and Llama are trained on massive public snapshots of the internet. Once training stops, their knowledge is frozen in time (**knowledge cutoff**). Furthermore, they have never seen your company's internal documents, customer databases, or confidential PDFs.

When you ask a model about something it does not know, it does not reliably say "I don't know." Instead, it predicts the next statistically likely words, creating believable but entirely fabricated claims—a phenomenon called **hallucination**.

#### The Analogy: Closed-Book vs. Open-Book Exam
- **Standard LLM (Closed-Book Exam):** Imagine taking a difficult engineering exam with zero notes. If you forgot a formula, you might guess or make up something plausible.
- **RAG (Open-Book Exam):** Imagine taking that same exam, but before answering each question, a research assistant hands you the exact 2 textbook paragraphs relevant to that question. You simply read the paragraphs and formulate a grounded answer.

```
CLOSED-BOOK (Standard LLM):
[User Question] ───────────────> [LLM Memory] ───────────────> [Guesses / Hallucination]

OPEN-BOOK (RAG Pipeline):
[User Question] ───> [Search Documents] ───> [Top Facts + Question] ───> [LLM] ───> [Grounded Answer + Citations]
```""",
        steps=[
            {
                "title": "Ask the Model a Question About Private Data",
                "before": "Let's ask our model a factual question about our fictional company, **Novasphere Dynamics**, without providing any documents. Let's see how it responds.",
                "code": """question = "What is the monthly price of Novasphere Dynamics' Enterprise Plan?"
response = llm.invoke(question)
print("Question:", question)
print("Model Response:", response.content)""",
                "after": "The model either admits it doesn't know, or guesses a fictional price! It has never seen Novasphere's confidential handbook."
            },
            {
                "title": "Attempting the Naive Fix: Prompt-Stuffing",
                "before": "The simplest workaround is **prompt-stuffing**: manually loading the entire handbook and pasting it into the prompt alongside the question.",
                "code": """handbook_path = get_data_dir() / "company_handbook.txt"
with open(handbook_path, "r", encoding="utf-8") as f:
    handbook_text = f.read()

prompt_stuffed = (
    "Read the company handbook below and answer the question.\\n"
    "If the answer is not in the text, say 'I don't know'.\\n\\n"
    "--- COMPANY HANDBOOK ---\\n"
    + handbook_text +
    f"\\n\\nQuestion: {question}\\nAnswer:"
)

response_stuffed = llm.invoke(prompt_stuffed)
print("Stuffed Prompt Answer:", response_stuffed.content)""",
                "after": "It worked! By pasting the handbook into context, the model correctly identified the Enterprise Plan price ($4,200/mo)."
            },
            {
                "title": "Why Prompt-Stuffing Does Not Scale",
                "before": "Prompt-stuffing works for a 2-page text file, but real organizations possess thousands of documents. Let's calculate why prompt-stuffing breaks down at scale.",
                "code": """chars_per_token = 4
num_docs = 5000
pages_per_doc = 10
chars_per_page = 2000

total_chars = num_docs * pages_per_doc * chars_per_page
total_tokens = total_chars / chars_per_token

print(f"Total tokens for 5,000 corporate docs: {total_tokens:,.0f} tokens")
print("Problems with prompt-stuffing at this scale:")
print("1. Exceeds model context windows.")
print("2. Massive latency: Processing 500k tokens per query takes 15-30+ seconds.")
print("3. Extreme cost: At $2.50 per 1M input tokens, 100 queries/day costs thousands per month!")
print("4. 'Lost in the middle': Models struggle to attend to facts buried in giant prompts.")""",
                "after": "We cannot stuff all documents into every query. We need a system that retrieves only the exact 2 or 3 paragraphs needed."
            },
            {
                "title": "Architectural Comparison: Prompt-Stuffing vs Fine-Tuning vs RAG",
                "before": "Beginners often ask: 'Why not just fine-tune an LLM on our company data?' Let's inspect the differences in a structured comparison table.",
                "code": """comparison_table = \"\"\"
| Dimension              | Prompt-Stuffing       | Fine-Tuning                  | RAG (Retrieval-Augmented)      |
|------------------------|-----------------------|------------------------------|--------------------------------|
| **Factual Accuracy**   | High (for short text) | Low (prone to hallucinations)| Very High (grounded in chunks) |
| **Data Recency**       | Instant               | Requires retraining pipeline | Instant (update index anytime) |
| **Auditability/Citations**| None (entire doc)  | Zero (black-box weights)     | Exact source chunk & metadata  |
| **Cost per Query**     | Very High (huge tokens)| Low                          | Low (only k small chunks sent) |
| **Best Used For**      | Single short memo     | Tone, style, domain syntax   | Dynamic knowledge bases & QA   |
\"\"\"
print(comparison_table)""",
                "after": "Fine-tuning teaches a model a new style or dialect, but is unreliable for factual recall. RAG is the gold standard for factual knowledge."
            },
            {
                "title": "The One-Sentence Definition of RAG",
                "before": "Let's summarize everything we learned into the core guiding definition of this entire course.",
                "code": """rag_definition = (
    "Retrieval-Augmented Generation (RAG) is a pattern that dynamically searches "
    "an external knowledge base for the most relevant document chunks and injects them "
    "into the LLM's prompt at query time to generate grounded, verifiable answers."
)
print("="*75)
print("DEFINITION OF RAG:")
print(rag_definition)
print("="*75)""",
                "after": "Keep this definition in mind: RAG = Retrieval (finding the facts) + Augmentation (injecting the facts into the prompt) + Generation (model writes the answer)."
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Recent Knowledge Test):** Ask the baseline LLM: *"Who was appointed Chief AI Ethics Officer at Novasphere Dynamics in October 2026?"* Observe the response.
2. **Exercise 2 (Grounding Test):** Load `course/data/recent_changelog.md`, inject only its text into a prompt, and ask the same question. Confirm the model extracts `Dr. Elina Vance`.""",
        solutions_code="""q_ethics = "Who was appointed Chief AI Ethics Officer at Novasphere Dynamics in October 2026?"
print("--- Exercise 1: Without Context ---")
print(llm.invoke(q_ethics).content)

print("\\n--- Exercise 2: With Grounded Context ---")
changelog_path = get_data_dir() / "recent_changelog.md"
with open(changelog_path, "r", encoding="utf-8") as f:
    changelog_text = f.read()

prompt_grounded = (
    "Answer based only on the context below:\\n"
    f"Context: {changelog_text}\\n"
    f"Question: {q_ethics}"
)

print(llm.invoke(prompt_grounded).content)""",
        common_mistakes=[
            "Confusing Fine-Tuning with RAG: Fine-tuning teaches how to speak (style/format); RAG teaches what to know (facts and documents).",
            "Assuming Large Context Windows Kill RAG: Million-token models are expensive, slow, and suffer from attention degradation on large documents.",
            "Treating LLMs as Databases: LLMs do not store facts in structured tables; they store word prediction weights.",
            "Omitting 'Say I don't know': Without explicit instructions to decline answering ungrounded questions, models will cheerfully hallucinate."
        ],
        recap=[
            "LLMs have a fixed knowledge cutoff and have never seen private organizational documents.",
            "Prompt-stuffing works on tiny files but fails rapidly on cost, latency, and context limits.",
            "Fine-tuning is expensive and poor at factual precision; RAG provides verifiable, citable answers.",
            "RAG operates like an open-book exam: retrieve the relevant paragraphs first, then generate the answer."
        ],
        next_teaser="In **01_rag_big_picture.ipynb**, we will build a complete, runnable RAG system in 25 lines of pure Python with zero external AI frameworks! You will see every moving part before touching a single library.",
        video_chapters=[
            {"timestamp": "00:00", "title": "Welcome to RAG From Scratch", "talking_points": "Course roadmap, prerequisites, and what we will build."},
            {"timestamp": "02:15", "title": "The Closed-Book Dilemma", "talking_points": "Why GPT and Claude hallucinate on internal company data."},
            {"timestamp": "05:10", "title": "Testing the LLM on Novasphere Dynamics", "talking_points": "Live demonstration of knowledge cutoff and refusal."},
            {"timestamp": "07:45", "title": "The Limits of Prompt-Stuffing", "talking_points": "Calculating token costs and latency for 5,000 corporate documents."},
            {"timestamp": "10:30", "title": "Prompt-Stuffing vs Fine-Tuning vs RAG", "talking_points": "When to fine-tune, when to stuff prompts, and why RAG wins for facts."},
            {"timestamp": "13:15", "title": "The 3 Pillars of RAG", "talking_points": "Ingestion, retrieval, and generation defined."}
        ],
        output_path=NOTEBOOKS_DIR / "00_why_rag.ipynb"
    )


def build_nb01():
    create_course_notebook(
        title="01. RAG The Big Picture: Build a Tiny RAG in Pure Python",
        promise="By the end of this notebook, you will build a complete, runnable RAG system in 25 lines of pure Python with zero frameworks, and map every single step to its LangChain equivalent.",
        objectives=[
            "Visualize the entire RAG lifecycle: Index Time (offline) vs. Query Time (online).",
            "Build a working RAG pipeline in pure Python using sentence splitting and word-overlap scoring.",
            "Map each plain Python operation directly to its corresponding LangChain primitive."
        ],
        prerequisites=["Notebook 00 (Why RAG? The Limits of LLMs)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.models import get_llm, detect_provider
from course.utils.helpers import get_data_dir

llm = get_llm(temperature=0.0)
print(f"Active Provider: {detect_provider()}")
print(f"Loaded Chat Model: {llm.__class__.__name__}")""",
        concept_md="""### The Two Halves of RAG: Index Time vs. Query Time

Every RAG system operates in two distinct phases:

1. **Index Time (The Preparation / Ingestion Phase - Run Once or Periodically):**
   - You take raw documents (PDFs, Markdown, Web pages).
   - You break them into smaller pieces called **chunks** (so you can search precisely).
   - You transform chunks into searchable mathematical representations (**embeddings**) and store them in a **vector store**.

2. **Query Time (The Search and Answer Phase - Run on Every User Question):**
   - The user asks a question.
   - The **retriever** searches the index and selects the top $k$ most relevant chunks.
   - The system injects those chunks into a **prompt template**.
   - The LLM reads the context and **generates** a grounded answer.

```
INDEX TIME (Offline Preparation):
[Documents] ──> [Text Splitter] ──> [Chunks] ──> [Embedding Model] ──> [Vector Database]

QUERY TIME (Live User Query):
[User Question] ──> [Search Query] ──> [Vector Database]
                                             │
                                             ▼
[Retrieved Chunks] + [User Question] ──> [Prompt] ──> [LLM] ──> [Grounded Answer]
```""",
        steps=[
            {
                "title": "Load Raw Document Text in Plain Python",
                "before": "Let's read our company handbook into a simple Python string using the standard `open()` function.",
                "code": """handbook_path = get_data_dir() / "company_handbook.txt"
with open(handbook_path, "r", encoding="utf-8") as f:
    raw_document = f.read()

print(f"Loaded document: {len(raw_document)} characters, {len(raw_document.split())} words.")""",
                "after": "We have loaded the raw text directly into memory."
            },
            {
                "title": "Split the Document into Sentences (Primitive Chunking)",
                "before": "Instead of passing the entire document to the model, we split the text into individual lines and sentences.",
                "code": """# Primitive chunking: split by line and filter out short empty lines
chunks = [line.strip() for line in raw_document.split("\\n") if len(line.strip()) > 30]
print(f"Split document into {len(chunks)} candidate sentence chunks.")
print("Sample chunk #1:", chunks[0])
print("Sample chunk #2:", chunks[1])""",
                "after": "We now have a list of small, focused text chunks."
            },
            {
                "title": "Score Chunks with Simple Word Overlap (Primitive Search)",
                "before": "When a user asks a question, how do we find the relevant chunks? Let's write a simple scoring function that counts shared words.",
                "code": """def score_overlap(query: str, chunk: str) -> float:
    q_words = set(query.lower().replace("?", "").replace("$", "").split())
    c_words = set(chunk.lower().replace(",", "").replace(".", "").split())
    stop_words = {"what", "is", "the", "of", "and", "a", "for", "in", "to"}
    meaningful_q = q_words - stop_words
    if not meaningful_q:
        return 0.0
    shared = meaningful_q.intersection(c_words)
    return len(shared) / len(meaningful_q)

query = "What is the price of the Starter Plan?"
scored = [(score_overlap(query, c), c) for c in chunks]
scored.sort(key=lambda x: x[0], reverse=True)

top_k = 2
retrieved_chunks = [c for score, c in scored[:top_k]]
for rank, (score, c) in enumerate(scored[:top_k], start=1):
    print(f"Rank {rank} (score: {score:.2f}): {c}")""",
                "after": "The word-overlap scorer successfully ranked the Starter Plan sentence at the very top!"
            },
            {
                "title": "Inject Retrieved Chunks into a Grounded Prompt",
                "before": "Now we combine the retrieved chunks into a prompt and instruct the model to answer using only that context.",
                "code": """context_block = "\\n".join(retrieved_chunks)
rag_prompt = (
    "You are a helpful customer support agent for Novasphere Dynamics.\\n"
    "Answer the user's question using ONLY the provided context. If the answer cannot be determined, say 'I don't know'.\\n\\n"
    f"CONTEXT:\\n{context_block}\\n\\n"
    f"QUESTION: {query}\\nANSWER:"
)

response = llm.invoke(rag_prompt)
print("="*60)
print(response.content)
print("="*60)""",
                "after": "You just built and executed a complete RAG system in pure Python with no external AI frameworks!"
            },
            {
                "title": "Mapping Pure Python to LangChain Components",
                "before": "While 25 lines of Python proved the concept, real applications require semantic search, robust document loaders, and production vector stores. Let's map each step to its LangChain equivalent.",
                "code": """mapping = \"\"\"
| Step in Pure Python              | LangChain Primitive                      | Why the Library Helps                    |
|-----------------------------------|------------------------------------------|------------------------------------------|
| `open('handbook.txt').read()`     | `DocumentLoader` (`TextLoader`, `PyPDFLoader`) | Parses PDFs, DOCX, HTML, extracts metadata |
| `raw_doc.split('\\n')`             | `RecursiveCharacterTextSplitter`         | Handles paragraphs, code, sentence boundaries|
| `score_overlap()` (word overlap)  | `Embeddings` (`OpenAI`, `SentenceTransformers`) | Understands synonyms & semantic meaning  |
| `scored[:k]` list sorting         | `VectorStore` (`Chroma`, `InMemoryVectorStore`) | Sub-millisecond ANN search across millions |
| `f"CONTEXT: {context}..."`        | `ChatPromptTemplate`                     | Composable system/human chat roles       |
| `llm.invoke(prompt)`              | `ChatModel` (`init_chat_model`)          | Streaming, tool calling, provider agnosticism|
\"\"\"
print(mapping)""",
                "after": "In the upcoming notebooks, we will replace each primitive step with its production-grade LangChain counterpart."
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Synonym Failure Test):** Change the query to *"What is the monthly fee for entry-level customers?"* (Notice that 'fee' and 'entry-level' are synonyms for 'price' and 'Starter Plan'). Run the word-overlap scorer. Does it find the Starter Plan chunk?
2. **Exercise 2 (Adding Source Metadata):** Modify the pure Python chunks list into a list of tuples `(text, source_name)`. Modify the prompt injection to print the source name alongside the answer.""",
        solutions_code="""synonym_query = "What is the monthly fee for entry-level customers?"
synonym_scored = [(score_overlap(synonym_query, c), c) for c in chunks]
synonym_scored.sort(key=lambda x: x[0], reverse=True)
print("Top chunk for synonym query:")
print(f"Score: {synonym_scored[0][0]:.2f} -> {synonym_scored[0][1]}")
print("-> NOTICE: Word overlap scored 0.0 or missed the target because 'fee' != 'price'!")
print("-> This is exactly why we need Vector Embeddings in Notebook 04.")""",
        common_mistakes=[
            "Confusing Index Time and Query Time: Indexing happens ahead of time; retrieval happens dynamically on every user prompt.",
            "Relying on Keyword Search Alone: Pure keyword search fails whenever users use synonyms (e.g. 'cost' vs 'pricing', 'error' vs 'bug').",
            "Retrieving Too Many Chunks: Sending 20 chunks clutters context and increases latency; start with k=2 to 4.",
            "Missing Fallback Instructions: Always tell the model to say 'I don't know' if context does not contain the answer."
        ],
        recap=[
            "RAG is conceptually simple: Split text -> Search relevant pieces -> Inject into prompt -> Generate grounded answer.",
            "Keyword matching works when words overlap exactly, but fails on synonyms and paraphrasing.",
            "Index time prepares the database; query time handles the live user interaction.",
            "Every complex RAG framework is simply an industrial-strength version of these 4 steps."
        ],
        next_teaser="In **02_loading_documents.ipynb**, we begin replacing our primitive Python steps with production LangChain components, starting with the `Document` object and loading text, Markdown, and real PDF files.",
        video_chapters=[
            {"timestamp": "00:00", "title": "Index Time vs Query Time", "talking_points": "The two operational phases of all RAG architectures."},
            {"timestamp": "02:30", "title": "Loading and Splitting in Pure Python", "talking_points": "Reading text and creating chunks with zero libraries."},
            {"timestamp": "05:15", "title": "Writing a Word-Overlap Scorer", "talking_points": "Understanding search ranking from first principles."},
            {"timestamp": "08:00", "title": "Assembling the Grounded Prompt", "talking_points": "Passing context to the model and seeing the output."},
            {"timestamp": "10:45", "title": "The Synonym Failure Case", "talking_points": "Why keyword search fails on real-world queries."},
            {"timestamp": "13:00", "title": "The LangChain Architecture Map", "talking_points": "Previewing the components we will build in notebooks 02-06."}
        ],
        output_path=NOTEBOOKS_DIR / "01_rag_big_picture.ipynb"
    )


def build_nb02():
    create_course_notebook(
        title="02. Loading Documents: Text, Markdown, and Real PDFs",
        promise="By the end of this notebook, you will be able to load, inspect, clean, and attach metadata to raw text, Markdown, and PDF documents using modern LangChain loaders.",
        objectives=[
            "Understand the anatomy of a LangChain Document object (page_content string and metadata dictionary).",
            "Load .txt, .md, and real PDF files using TextLoader, PyPDFLoader, and directory scanners.",
            "Clean and standardize document metadata to power downstream search filters and citations."
        ],
        prerequisites=["Notebook 01 (RAG The Big Picture)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.helpers import get_data_dir

data_dir = get_data_dir()
print(f"Data directory: {data_dir}")
print("Files available:", [f.name for f in data_dir.iterdir() if f.is_file()])""",
        concept_md="""### The Fundamental Unit: The `Document` Object

In LangChain, all textual data flows through a single standardized dataclass called a **Document**.

A Document has exactly two fields:
1. `page_content: str` — The actual text (the raw knowledge).
2. `metadata: dict` — A dictionary of key-value pairs describing where the text came from (file path, page number, creation date, author, department).

#### Why Metadata Matters
Imagine a library with 10,000 loose sheets of paper. Without titles or page numbers, finding anything or verifying a fact is impossible. Metadata is the index card attached to each sheet.

```
┌────────────────────────────────────────────────────────┐
│ Document Object                                        │
├────────────────────────────────────────────────────────┤
│ metadata: {                                            │
│    "source": "company_handbook.txt",                   │
│    "page": 1,                                          │
│    "category": "policy"                                │
│ }                                                      │
├────────────────────────────────────────────────────────┤
│ page_content:                                          │
│ "Employees traveling on company business are           │
│  reimbursed up to $150 per day for meals..."           │
└────────────────────────────────────────────────────────┘
```""",
        steps=[
            {
                "title": "Creating a Document Object by Hand",
                "before": "Let's import `Document` from `langchain_core.documents` and construct one manually to see how simple it is.",
                "code": """from langchain_core.documents import Document

manual_doc = Document(
    page_content="Novasphere provides 24/7 mission-critical support on the Enterprise plan.",
    metadata={"source": "manual_entry", "department": "Customer Engineering"}
)

print("Type:", type(manual_doc))
print("Content:", manual_doc.page_content)
print("Metadata:", manual_doc.metadata)""",
                "after": "Every document loader in LangChain produces objects following this exact structure."
            },
            {
                "title": "Loading Text and Markdown Files",
                "before": "Now let's use `TextLoader` from `langchain_community.document_loaders` to load `company_handbook.txt` and `product_faq.md`.",
                "code": """from langchain_community.document_loaders import TextLoader

handbook_loader = TextLoader(str(data_dir / "company_handbook.txt"), encoding="utf-8")
handbook_docs = handbook_loader.load()

faq_loader = TextLoader(str(data_dir / "product_faq.md"), encoding="utf-8")
faq_docs = faq_loader.load()

print(f"Handbook docs: {len(handbook_docs)}, chars: {len(handbook_docs[0].page_content)}")
print("Handbook Metadata:", handbook_docs[0].metadata)
print(f"FAQ docs: {len(faq_docs)}, chars: {len(faq_docs[0].page_content)}")""",
                "after": "Notice that TextLoader loads the entire file into a single Document object and attaches the file path to metadata['source']."
            },
            {
                "title": "Loading Real PDF Files with PyPDFLoader",
                "before": "PDFs are multi-page binary documents. `PyPDFLoader` automatically parses page streams and produces one Document object per page.",
                "code": """from langchain_community.document_loaders import PyPDFLoader

pdf_path = data_dir / "policy_brief.pdf"
pdf_loader = PyPDFLoader(str(pdf_path))
pdf_docs = pdf_loader.load()

print(f"Loaded {len(pdf_docs)} page(s) from {pdf_path.name}")
for idx, page_doc in enumerate(pdf_docs):
    print(f"\\n--- Page {idx+1} (Source: {Path(page_doc.metadata.get('source', '')).name}, Page: {page_doc.metadata.get('page')}) ---")
    print("Content preview:", page_doc.page_content[:180].strip())""",
                "after": "PyPDFLoader automatically attached the 'page' number to the metadata! This enables page-level citations in user interfaces."
            },
            {
                "title": "Loading an Entire Directory of Mixed Files",
                "before": "In production, you want to point your pipeline at a folder of mixed `.txt`, `.md`, and `.pdf` files. Let's write a robust loader loop.",
                "code": """all_docs = []

for file_path in data_dir.iterdir():
    if file_path.suffix.lower() == ".pdf":
        all_docs.extend(PyPDFLoader(str(file_path)).load())
    elif file_path.suffix.lower() in [".txt", ".md"]:
        all_docs.extend(TextLoader(str(file_path), encoding="utf-8").load())

print(f"Total documents loaded from directory: {len(all_docs)}")
for d in all_docs:
    print(f"  * {Path(d.metadata['source']).name} ({len(d.page_content)} characters)")""",
                "after": "We now have a clean collection of Document objects ready for splitting and indexing."
            },
            {
                "title": "Cleaning and Normalizing Metadata",
                "before": "Raw metadata often contains long absolute filesystem paths (`C:\\project 03\\...`). Let's normalize metadata for readability and auditability.",
                "code": """for doc in all_docs:
    full_source = doc.metadata.get("source", "")
    doc.metadata["filename"] = Path(full_source).name
    doc.metadata["char_count"] = len(doc.page_content)
    if "handbook" in doc.metadata["filename"]:
        doc.metadata["category"] = "policy"
    elif "faq" in doc.metadata["filename"]:
        doc.metadata["category"] = "product"
    elif "changelog" in doc.metadata["filename"]:
        doc.metadata["category"] = "releases"
    else:
        doc.metadata["category"] = "general"

print("Sample Cleaned Metadata:")
print(all_docs[0].metadata)""",
                "after": "Our documents now have clean, consistent metadata categories for search filtering!"
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Empty Document Guard):** Write a list comprehension that filters out any documents that have fewer than 20 characters of `page_content`.
2. **Exercise 2 (Web Loader with Offline Fallback):** Write a minimal function that attempts to load text from a URL using `requests` and `BeautifulSoup`, falling back to a local sample string if offline.""",
        solutions_code="""non_empty_docs = [d for d in all_docs if len(d.page_content.strip()) >= 20]
print(f"Retained {len(non_empty_docs)} of {len(all_docs)} documents.")

import requests
from bs4 import BeautifulSoup

def safe_load_web(url: str) -> Document:
    try:
        resp = requests.get(url, timeout=3.0)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return Document(page_content=soup.get_text()[:500], metadata={"source": url})
    except Exception as e:
        print(f"[Offline Fallback] Network request failed ({e}). Using cached document.")
        return Document(page_content="Cached Web Content: Novasphere Robotics Documentation", metadata={"source": url, "cached": True})

sample_web_doc = safe_load_web("https://example.com/novasphere-docs")
print("Web Doc Source:", sample_web_doc.metadata["source"])""",
        common_mistakes=[
            "Ignoring File Encodings: Always specify `encoding='utf-8'` in TextLoader on Windows, or UTF-8 characters will raise UnicodeDecodeError.",
            "Overwriting Metadata: When adding custom fields, do not overwrite existing keys like `source` or `page`.",
            "Loading Scanned PDFs Without OCR: Standard PDF loaders only extract embedded digital text. Scanned images will return empty strings.",
            "Discarding Document Objects: Don't convert Documents to raw strings too early; you will lose the metadata needed for citations."
        ],
        recap=[
            "The Document object (`page_content` + `metadata`) is the universal data unit in LangChain.",
            "TextLoader loads plain text and Markdown; PyPDFLoader extracts text and records page numbers.",
            "Clean metadata (filename, page, category) is essential for filtering and source citations.",
            "Always handle character encodings and empty pages defensively."
        ],
        next_teaser="In **03_splitting_text.ipynb**, we take our loaded documents and solve the chunking puzzle: how large should chunks be, what is chunk overlap, and how do we prevent answers from being cut in half?",
        video_chapters=[
            {"timestamp": "00:00", "title": "The LangChain Document Object", "talking_points": "Understanding page_content and metadata."},
            {"timestamp": "02:30", "title": "Loading Text and Markdown", "talking_points": "TextLoader and UTF-8 encoding best practices."},
            {"timestamp": "05:10", "title": "Parsing PDFs with PyPDFLoader", "talking_points": "Extracting page numbers and handling multi-page documents."},
            {"timestamp": "08:20", "title": "Directory Loading and File Iteration", "talking_points": "Loading heterogeneous folders of company files."},
            {"timestamp": "11:00", "title": "Metadata Cleaning and Normalization", "talking_points": "Preparing metadata for downstream vector filtering."}
        ],
        output_path=NOTEBOOKS_DIR / "02_loading_documents.ipynb"
    )


def build_nb03():
    create_course_notebook(
        title="03. Splitting Text: Chunk Size, Overlap, and Boundary Safety",
        promise="By the end of this notebook, you will master document chunking, understand the trade-offs of chunk size versus overlap, and prevent information loss when cutting text.",
        objectives=[
            "Explain why chunking is required for search precision and model context limits.",
            "Master RecursiveCharacterTextSplitter and its delimiter hierarchy (\\n\\n, \\n, space, empty string).",
            "Demonstrate a real split failure that cuts an answer in half and resolve it using chunk overlap."
        ],
        prerequisites=["Notebook 02 (Loading Documents)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.helpers import get_data_dir
from langchain_community.document_loaders import TextLoader

data_dir = get_data_dir()
handbook_doc = TextLoader(str(data_dir / "company_handbook.txt"), encoding="utf-8").load()[0]
print(f"Loaded handbook: {len(handbook_doc.page_content)} characters.")""",
        concept_md="""### Why Can't We Just Search Full Documents?

1. **Context Limits & Cost:** Sending a 50-page document to an LLM for a 1-sentence answer wastes tokens and causes high latency.
2. **Search Precision:** If an entire 50-page document is represented by a single coordinate vector, specific detailed facts (like a meal expense limit) get drowned out in the average.
3. **Chunking** cuts documents into bite-sized semantic passages so search finds the exact needle in the haystack.

#### Chunk Size vs. Chunk Overlap
- **Chunk Size:** The maximum number of characters (or tokens) per piece.
  - *Too small (e.g. 50 chars):* Cuts sentences in half; loses context.
  - *Too large (e.g. 3000 chars):* Dilutes search precision; wastes tokens.
- **Chunk Overlap:** The number of characters shared between adjacent chunks. It creates a sliding window that prevents facts at the boundary from being severed.

```
NO OVERLAP (Risk of cut sentence):
[ Chunk 1: Employees are reimbursed up to $150 ] [ Chunk 2: per day for meals without VP approval ]
                                                ▲
                                          Broken Meaning!

WITH OVERLAP (Boundary Preserved):
[ Chunk 1: Employees are reimbursed up to $150 per day for meals... ]
             [ Chunk 2: ...up to $150 per day for meals without VP approval. ]
```""",
        steps=[
            {
                "title": "The Recursive Splitting Hierarchy",
                "before": "The gold standard text splitter is `RecursiveCharacterTextSplitter`. It attempts to split by paragraphs (`\\n\\n`) first. If chunks are still too large, it tries lines (`\\n`), then spaces (` `), and only as a last resort splits by characters.",
                "code": """from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\\n\\n", "\\n", " ", ""]
)

chunks = splitter.split_documents([handbook_doc])
print(f"Document split into {len(chunks)} chunks.")
print("Chunk #1 preview:\\n", chunks[0].page_content)
print("\\nChunk #2 preview:\\n", chunks[1].page_content)""",
                "after": "Notice that the splitter cleanly kept headings and paragraphs together where possible."
            },
            {
                "title": "Experiment: The Impact of Chunk Size",
                "before": "Let's run an experiment table comparing chunk sizes of 100, 300, and 800 characters on the exact same document.",
                "code": """results = []
for size in [100, 300, 800]:
    s = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=20)
    c = s.split_documents([handbook_doc])
    lengths = [len(x.page_content) for x in c]
    results.append({
        "chunk_size": size,
        "num_chunks": len(c),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": sum(lengths) // len(lengths)
    })

print(f"{'Chunk Size':<12} | {'Num Chunks':<12} | {'Min Len':<10} | {'Max Len':<10} | {'Avg Len':<10}")
print("-" * 65)
for r in results:
    print(f"{r['chunk_size']:<12} | {r['num_chunks']:<12} | {r['min_len']:<10} | {r['max_len']:<10} | {r['avg_len']:<10}")""",
                "after": "Smaller chunk size produces many small snippets; larger chunk size produces fewer, broader chunks."
            },
            {
                "title": "The Failure Case: A Bad Split That Cuts an Answer in Half",
                "before": "Let's demonstrate a critical failure: split with `chunk_size=100` and `chunk_overlap=0` on the travel expense policy.",
                "code": """fragile_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
bad_chunks = fragile_splitter.split_documents([handbook_doc])

target_phrase = "$150 per day"
found_chunks = [c for c in bad_chunks if target_phrase in c.page_content]

print(f"Found {len(found_chunks)} chunk containing '{target_phrase}':")
for c in found_chunks:
    print(">>>", repr(c.page_content))""",
                "after": "Look at the chunk: it ends right after '$150 per day for meals'! The critical condition 'without requiring prior VP approval' was cut off into the next chunk!"
            },
            {
                "title": "The Overlap Fix: Preserving Boundary Context",
                "before": "Now let's add `chunk_overlap=45`. Notice how the sliding window retains the complete policy in both chunks.",
                "code": """safe_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=45)
fixed_chunks = safe_splitter.split_documents([handbook_doc])

fixed_targets = [c for c in fixed_chunks if target_phrase in c.page_content]
print(f"With overlap=45, found {len(fixed_targets)} chunk(s):")
for idx, c in enumerate(fixed_targets, 1):
    print(f"\\n--- Safe Chunk #{idx} ---")
    print(c.page_content)""",
                "after": "Both the dollar amount AND the VP approval rule are preserved together. Overlap saved the answer!"
            },
            {
                "title": "Preserving Metadata Across Chunks",
                "before": "When a document is split into 10 chunks, does the metadata survive? Let's verify.",
                "code": """for idx, chunk in enumerate(chunks[:3]):
    chunk.metadata["chunk_index"] = idx
    print(f"Chunk #{idx} Metadata: {chunk.metadata}")""",
                "after": "Every chunk automatically inherited the source metadata from the parent document."
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Markdown Splitting):** Load `product_faq.md` and split it using `chunk_size=300, chunk_overlap=30`. Print the total number of chunks.
2. **Exercise 2 (Inspect Overlap):** Write a function `print_overlap(chunk_a, chunk_b)` that finds and prints the overlapping text string between two consecutive chunks.""",
        solutions_code="""faq_doc = TextLoader(str(data_dir / "product_faq.md"), encoding="utf-8").load()[0]
faq_chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents([faq_doc])
print(f"Split FAQ into {len(faq_chunks)} chunks.")

def print_overlap(a: str, b: str):
    for i in range(len(a), 0, -1):
        suffix = a[-i:]
        if b.startswith(suffix):
            print(f"Shared overlap ({len(suffix)} chars): {repr(suffix)}")
            return
    print("No direct boundary overlap found.")

print_overlap(chunks[0].page_content, chunks[1].page_content)""",
        common_mistakes=[
            "Zero Overlap: Setting `chunk_overlap=0` will inevitably cut key sentences in half at arbitrary character limits.",
            "Excessive Overlap: Setting overlap to 50%+ of chunk size creates redundant chunks and bloats vector store costs.",
            "Character Count vs Token Count: LLMs measure limits in tokens (~4 characters per token in English). Keep this in mind when sizing chunks.",
            "Splitting Code or Markdown Tables Naively: Naive splitters can break table rows. Use larger chunks or specialized Markdown splitters for structured tables."
        ],
        recap=[
            "Chunking balances search precision against context completeness.",
            "RecursiveCharacterTextSplitter respects natural paragraphs and sentences before character splits.",
            "Chunk overlap (typically 10-20% of chunk size) prevents facts at boundaries from being broken.",
            "Metadata is preserved across all child chunks."
        ],
        next_teaser="In **04_embeddings_and_vector_stores.ipynb**, we turn these text chunks into mathematical vectors of meaning, compute cosine similarity by hand, and build our first vector database.",
        video_chapters=[
            {"timestamp": "00:00", "title": "The Purpose of Chunking", "talking_points": "Why we cannot search entire documents at once."},
            {"timestamp": "02:20", "title": "Understanding Recursive Splitting", "talking_points": "The paragraph -> sentence -> word hierarchy."},
            {"timestamp": "05:10", "title": "The Chunk Size Experiment", "talking_points": "Comparing sizes 100, 300, and 800."},
            {"timestamp": "08:15", "title": "The Boundary Severance Bug", "talking_points": "How bad chunking cuts facts in half."},
            {"timestamp": "11:00", "title": "The Overlap Remedy", "talking_points": "Sliding windows and metadata inheritance."}
        ],
        output_path=NOTEBOOKS_DIR / "03_splitting_text.ipynb"
    )

if __name__ == "__main__":
    build_nb00()
    build_nb01()
    build_nb02()
    build_nb03()
    print("Notebooks 00-03 built successfully!")
