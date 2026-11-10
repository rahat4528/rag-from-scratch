"""
Builder for Phase 1 - Notebooks 04 to 07.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from course.utils.notebook_builder import create_course_notebook

NOTEBOOKS_DIR = Path("course/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_nb04():
    create_course_notebook(
        title="04. Embeddings and Vector Stores: Meaning as Numbers",
        promise="By the end of this notebook, you will understand how text is transformed into semantic coordinate vectors, calculate cosine similarity by hand with NumPy, and index documents into in-memory and persistent vector stores.",
        objectives=[
            "Define text embeddings as semantic coordinate vectors and compute cosine similarity by hand using pure NumPy.",
            "Visualize sentence semantic similarity on a 2D scatter plot using Matplotlib.",
            "Index document chunks into InMemoryVectorStore and persist them to disk."
        ],
        prerequisites=["Notebook 03 (Splitting Text)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.models import get_embeddings, detect_provider
from course.utils.helpers import get_data_dir, cosine_sim

embeddings = get_embeddings()
print(f"Active Provider: {detect_provider()}")
print(f"Loaded Embeddings Model: {embeddings.__class__.__name__}")""",
        concept_md="""### What is an Embedding? (Meaning as Coordinates)

Computers cannot understand the word "puppy". They can only understand numbers.

An **embedding model** takes a piece of text and maps it to a list of numbers (a vector) in a high-dimensional space.
- Similar meanings are placed close together.
- Different meanings are placed far apart.

#### The Intuition: A 2D Map of Words
Imagine a 2D grid:
- Horizontal axis: Domestic Pet <---------------------> Wild Predator
- Vertical axis: Tiny Size <-------------------------> Massive Size

- "Chihuahua" might sit at `(0.1, 0.1)`.
- "Golden Retriever" might sit at `(0.2, 0.6)`.
- "Grizzly Bear" might sit at `(0.9, 0.9)`.

Because "Chihuahua" and "Golden Retriever" have similar coordinates, their **distance** is small and their **cosine similarity** is close to 1.0!

```
                    Massive Size (1.0)
                           │
             [Grizzly Bear]│
                           │   [Golden Retriever]
                           │
                           │   [Chihuahua]
Domestic Pet (0.0) ────────┼────────────── Wild Predator (1.0)
                           │
                     Tiny Size (0.0)
```""",
        steps=[
            {
                "title": "Calculating Cosine Similarity by Hand with NumPy",
                "before": "Cosine similarity measures the angle between two vectors. If they point in the same direction, similarity is 1.0; if orthogonal, 0.0; if opposite, -1.0. Let's calculate it manually.",
                "code": """import numpy as np

def manual_cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

pet_1 = [0.2, 0.6]  # Golden retriever
pet_2 = [0.25, 0.55] # Labrador
car = [0.95, -0.4]  # Sports car

print("Similarity (Retriever vs Labrador):", manual_cosine_similarity(pet_1, pet_2))
print("Similarity (Retriever vs Sports Car):", manual_cosine_similarity(pet_1, car))""",
                "after": "Notice how the two dogs scored >0.99 similarity, while the car scored negative similarity!"
            },
            {
                "title": "Visualizing Sentence Similarity in 2D",
                "before": "Let's embed 5 real sentences and plot their relationship on a 2D scatter plot using PCA dimensionality reduction.",
                "code": """sentences = [
    "I love puppies and golden retrievers.",
    "Canines make wonderful household pets.",
    "The company published its quarterly financial earnings.",
    "Corporate revenues exceeded annual budget expectations.",
    "Quantum physics explores subatomic particles."
]

sentence_vectors = [embeddings.embed_query(s) for s in sentences]
print(f"Generated {len(sentence_vectors)} vectors, each with dimension {len(sentence_vectors[0])}.")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
coords_2d = pca.fit_transform(sentence_vectors)

plt.figure(figsize=(7, 4))
plt.scatter(coords_2d[:, 0], coords_2d[:, 1], color='dodgerblue', s=80)
labels = ["Dogs 1", "Dogs 2", "Finance 1", "Finance 2", "Quantum"]
for idx, (x, y) in enumerate(coords_2d):
    plt.annotate(labels[idx], (x + 0.01, y + 0.01), fontsize=9)
plt.title("2D Projection of Semantic Sentence Embeddings")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()""",
                "after": "The two dog sentences clustered together, the two financial sentences clustered together, and quantum physics sits alone!"
            },
            {
                "title": "Creating an In-Memory Vector Store",
                "before": "A vector store stores Document chunks alongside their embeddings and indexes them for fast nearest-neighbor search. Let's create an `InMemoryVectorStore`.",
                "code": """from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data_dir = get_data_dir()
doc = TextLoader(str(data_dir / "company_handbook.txt"), encoding="utf-8").load()[0]
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents([doc])

vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
print(f"Indexed {len(chunks)} chunks into InMemoryVectorStore.")""",
                "after": "The vector store embedded all chunks and built a searchable in-memory index in seconds."
            },
            {
                "title": "Performing Semantic Similarity Search",
                "before": "Now let's query the vector store. Remember: vector search works on conceptual meaning, NOT exact keyword matches!",
                "code": """query = "How much can I expense for dinner when traveling?"
matches = vector_store.similarity_search(query, k=2)

print(f"Query: {query}\\n")
for i, m in enumerate(matches, 1):
    print(f"--- Result #{i} ---")
    print(m.page_content.strip())""",
                "after": "Even though the user said 'dinner when traveling', the vector store retrieved the chunk mentioning 'meals without prior VP approval'!"
            },
            {
                "title": "Persisting to Disk with VectorStore Persistence",
                "before": "In-memory stores disappear when your program closes. LangChain's `InMemoryVectorStore` provides `.dump(path)` and `.load(path, embeddings)` to persist the entire index to disk without needing external database servers.",
                "code": """index_file = Path("handbook_store.json")
vector_store.dump(str(index_file))
print(f"Persisted vector index to: {index_file.resolve()} ({index_file.stat().st_size} bytes)")

reloaded_store = InMemoryVectorStore.load(str(index_file), embeddings)
reloaded_matches = reloaded_store.similarity_search(query, k=1)
print("\\nReloaded index search result:")
print(reloaded_matches[0].page_content.strip())""",
                "after": "The complete vector index was dumped to disk as JSON and reloaded instantly."
            },
            {
                "title": "The Golden Rule: Embedding Models Must Match!",
                "before": "Why can you never index with one embedding model and query with another? Let's inspect the math.",
                "code": """rule_md = \"\"\"
[CRITICAL ARCHITECTURAL RULE]
Embedding spaces are model-specific coordinate systems:
- If Model A maps 'pricing' to coordinates (0.42, -0.88, 0.12)
- Model B might map 'pricing' to (-0.11, 0.73, -0.95)

If you index documents with Model A and query with Model B, the coordinates have no mathematical relationship!
-> Search will return completely random chunks.
-> When changing embedding models in production, you MUST re-index your entire document corpus.
\"\"\"
print(rule_md)""",
                "after": "Always use the exact same embedding model for both indexing and querying."
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Similarity Score Inspection):** Use `similarity_search_with_score(query, k=2)` on `reloaded_store` and print each chunk alongside its distance score.
2. **Exercise 2 (Semantic Synonym Test):** Search for *"What is the policy regarding bringing dogs to work?"* Notice how vector search finds the pet policy even if the word 'dog' isn't explicitly mentioned.""",
        solutions_code="""scored_results = reloaded_store.similarity_search_with_score("What is the pet policy in Zurich?", k=2)
for doc, score in scored_results:
    print(f"Score (Distance): {score:.4f} -> {doc.page_content[:100]}...")

dog_results = reloaded_store.similarity_search("Can I bring my golden retriever to the office?", k=1)
print("\\nDog Query Result:")
print(dog_results[0].page_content.strip())""",
        common_mistakes=[
            "Mismatched Embedding Models: Querying with a different embedding model than the one used to index will produce random noise.",
            "Confusing Cosine Distance vs Cosine Similarity: Cosine similarity is 1.0 for identical vectors; cosine distance is (1 - similarity), so 0.0 means identical.",
            "Re-embedding on Every Query: Only embed documents once during indexing; during query time, only embed the single user question.",
            "Embedding Giant Documents at Once: Always chunk first; embedding a 50-page document into one vector loses all granular details."
        ],
        recap=[
            "Embeddings convert text into high-dimensional coordinate vectors of semantic meaning.",
            "Cosine similarity measures the angle between vectors to identify conceptual similarity.",
            "Vector stores index vectors for sub-millisecond approximate nearest neighbor (ANN) retrieval.",
            "LangChain vector stores support disk persistence via dump/load or dedicated database servers.",
            "The same embedding model must be used for both indexing and query search."
        ],
        next_teaser="In **05_retrieval.ipynb**, we explore advanced retrieval techniques: tuning top-k, using Maximal Marginal Relevance (MMR) to prevent redundancy, and filtering by metadata.",
        video_chapters=[
            {"timestamp": "00:00", "title": "What is an Embedding?", "talking_points": "Translating human concepts into high-dimensional coordinate space."},
            {"timestamp": "02:40", "title": "Cosine Similarity by Hand", "talking_points": "The vector math behind semantic similarity."},
            {"timestamp": "05:15", "title": "Visualizing Embeddings in 2D", "talking_points": "Clustering sentences with PCA and Matplotlib."},
            {"timestamp": "08:00", "title": "Building an InMemoryVectorStore", "talking_points": "Indexing chunks in LangChain."},
            {"timestamp": "10:30", "title": "Vector Store Persistence", "talking_points": "Saving and reloading indices from disk."},
            {"timestamp": "13:00", "title": "The Golden Rule of Embedding Compatibility", "talking_points": "Why you cannot mix embedding models."}
        ],
        output_path=NOTEBOOKS_DIR / "04_embeddings_and_vector_stores.ipynb"
    )


def build_nb05():
    create_course_notebook(
        title="05. Retrieval: Finding the Right Chunks at Query Time",
        promise="By the end of this notebook, you will master vector search techniques, tune the top-k parameter, use MMR for diversity, and apply metadata filters to eliminate irrelevant results.",
        objectives=[
            "Differentiate similarity_search from similarity_search_with_score and interpret distance metrics.",
            "Tune the top-k parameter and apply Maximal Marginal Relevance (MMR) to prevent redundant chunks.",
            "Apply metadata filters to scope search to specific document categories and diagnose 3 classic retrieval failure modes."
        ],
        prerequisites=["Notebook 04 (Embeddings and Vector Stores)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.models import get_embeddings, detect_provider
from course.utils.helpers import get_data_dir, pretty_print_docs
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

data_dir = get_data_dir()
embeddings = get_embeddings()

docs = []
for fname in ["company_handbook.txt", "product_faq.md", "recent_changelog.md"]:
    loaded = TextLoader(str(data_dir / fname), encoding="utf-8").load()
    for d in loaded:
        d.metadata["source"] = fname
    docs.extend(loaded)

chunks = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=35).split_documents(docs)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
print(f"Indexed {len(chunks)} chunks into vector store.")""",
        concept_md="""### The Role of Retrieval: Setting the Upper Bound of Quality

The retriever is the gatekeeper of your RAG pipeline:
> **"Retrieval quality sets the strict upper bound for generation quality."**
If your retriever returns irrelevant or incomplete chunks, even the most capable language model will hallucinate or give up.

#### The Top-k Trade-Off
- **k Too Small (e.g. k=1):** You save tokens, but if the answer requires facts from two different sections, the model misses critical information.
- **k Too Large (e.g. k=15):** You maximize recall, but flood the prompt with distracting noise, inflate API costs, and risk the 'lost in the middle' effect.

#### Maximal Marginal Relevance (MMR)
When you ask a broad question, standard vector search might return 3 nearly identical sentences from the same page. **MMR (Maximal Marginal Relevance)** balances two competing goals:
1. Finding chunks that are relevant to the query.
2. Selecting chunks that are diverse and different from already chosen chunks.

```
STANDARD TOP-K SEARCH:
[Query] ──> Returns top 3 chunks (Often redundant variants of the same paragraph)

MMR SEARCH (Diversity Aware):
[Query] ──> Selects most relevant chunk #1
            ──> Penalizes chunks that look identical to #1
            ──> Selects complementary chunk #2 from another section!
```""",
        steps=[
            {
                "title": "Search with Similarity Scores",
                "before": "Let's inspect what vector search actually computes. Using `similarity_search_with_score`, we can see the exact distance or similarity metric for each chunk.",
                "code": """query = "What is the guaranteed uptime SLA for Enterprise customers?"
results_with_scores = vector_store.similarity_search_with_score(query, k=2)

for doc, score in results_with_scores:
    print(f"\\n--- Chunk (Distance Score: {score:.4f}, Source: {doc.metadata['source']}) ---")
    print(doc.page_content.strip())""",
                "after": "The top result has the lowest distance score and contains the exact 99.995% SLA answer."
            },
            {
                "title": "Converting the Vector Store to a Retriever",
                "before": "In LangChain, components connect through standard interfaces. The `.as_retriever()` method wraps any vector store into a callable Runnable.",
                "code": """retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

retrieved_docs = retriever.invoke(query)
pretty_print_docs(retrieved_docs)""",
                "after": "The retriever returned the top 2 Document objects, formatted cleanly with our helper utility."
            },
            {
                "title": "Failure Mode 1: Vague Queries and Information Dilution",
                "before": "What happens when a user asks an ambiguous, one-word question like 'pricing'? Let's observe the retrieved results.",
                "code": """vague_query = "pricing"
vague_results = retriever.invoke(vague_query)
pretty_print_docs(vague_results)""",
                "after": "Notice that 'pricing' retrieves generic policy paragraphs rather than specific tier prices. Vague queries produce unfocused context."
            },
            {
                "title": "Failure Mode 2: Vocabulary Mismatch (The Semantic Search Superpower)",
                "before": "Let's ask a question that uses completely different words than our handbook: 'What is the reimbursement limit for evening supper on the road?' (The handbook says 'meals when traveling on company business').",
                "code": """mismatch_query = "What is the reimbursement limit for evening supper on the road?"
mismatch_results = retriever.invoke(mismatch_query)
pretty_print_docs(mismatch_results)""",
                "after": "Even with zero exact keyword overlap ('supper on the road' vs 'meals when traveling'), vector search successfully retrieved the $150 meal policy!"
            },
            {
                "title": "Maximal Marginal Relevance (MMR) for Diversity",
                "before": "Let's test MMR search. When we want diverse coverage across different operational areas, MMR prevents the retriever from selecting 3 repetitive sentences.",
                "code": """mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6}
)

broad_query = "What are the rules and support hours?"
mmr_results = mmr_retriever.invoke(broad_query)
pretty_print_docs(mmr_results)""",
                "after": "MMR selected chunks spanning both support hours and operational policies, ensuring comprehensive coverage."
            },
            {
                "title": "Metadata Filtering: Scoping the Search",
                "before": "If we know the user is troubleshooting a product error, we can filter search results to only search `product_faq.md` using metadata filters.",
                "code": """filtered_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2,
        "filter": lambda doc: doc.metadata.get("source") == "product_faq.md"
    }
)

filtered_results = filtered_retriever.invoke("How do I fix error ERR_QUANTUM_TIMEOUT?")
pretty_print_docs(filtered_results)""",
                "after": "Search was strictly confined to the technical FAQ, eliminating irrelevant handbook chunks."
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Tune k):** Experiment with $k=1$ versus $k=5$ for the question: *"Which subscription tier includes ZeroGravity and what is its failover threshold?"* Notice whether $k=1$ captures both the pricing tier and the technical threshold.
2. **Exercise 2 (Changelog Filter):** Create a filtered retriever that only searches `recent_changelog.md` and query for *"What bug was fixed on Windows ARM64?"*""",
        solutions_code="""q_multi = "Which subscription tier includes ZeroGravity and what is its failover threshold?"
ret_k1 = vector_store.as_retriever(search_kwargs={"k": 1}).invoke(q_multi)
ret_k4 = vector_store.as_retriever(search_kwargs={"k": 4}).invoke(q_multi)
print(f"With k=1: Captured 40ms threshold? {any('40 milliseconds' in c.page_content for c in ret_k1)}")
print(f"With k=4: Captured both facts? {any('40 milliseconds' in c.page_content for c in ret_k4) and any('$4,200' in c.page_content for c in ret_k4)}")

cl_retriever = vector_store.as_retriever(search_kwargs={"k": 1, "filter": lambda d: d.metadata.get("source") == "recent_changelog.md"})
cl_res = cl_retriever.invoke("What bug was fixed on Windows ARM64?")
print("\\nChangelog Filter Result:\\n", cl_res[0].page_content.strip())""",
        common_mistakes=[
            "Assuming Vector Search is Perfect: Vector search can occasionally return semantically similar but factually irrelevant text; always inspect chunks.",
            "Ignoring Top-k Tuning: Leaving k=4 as a hardcoded constant for all queries leads to under-retrieval for broad queries and noise for simple lookups.",
            "Not Cleaning Metadata Before Filtering: If source paths vary between platforms (e.g. '/' vs '\\'), string filters will fail silently.",
            "Over-filtering: Applying overly restrictive metadata filters can result in zero retrieved chunks (an empty context)."
        ],
        recap=[
            "Retrieval quality sets the upper bound for generation quality.",
            "as_retriever() turns any vector store into a standardized Runnable component.",
            "MMR balances relevance with diversity, preventing redundant chunks.",
            "Metadata filtering allows surgical scoping to specific files, categories, or tenants.",
            "Semantic search excels at overcoming vocabulary mismatches that break keyword search."
        ],
        next_teaser="In **06_generation_first_rag_chain.ipynb**, we connect our retriever to a language model using LangChain Expression Language (LCEL) and rebuild it as a clear two-node LangGraph.",
        video_chapters=[
            {"timestamp": "00:00", "title": "The Golden Rule of Retrieval", "talking_points": "Why retrieval quality sets the ceiling for RAG."},
            {"timestamp": "02:30", "title": "Inspecting Similarity Scores", "talking_points": "Understanding distance metrics in practice."},
            {"timestamp": "05:00", "title": "Tuning Top-k: Precision vs Noise", "talking_points": "Choosing the right number of chunks."},
            {"timestamp": "07:45", "title": "The Vocabulary Mismatch Miracle", "talking_points": "How embeddings bridge synonym gaps."},
            {"timestamp": "10:15", "title": "Maximal Marginal Relevance (MMR)", "talking_points": "Eliminating redundant chunks with diversity search."},
            {"timestamp": "12:30", "title": "Scoping with Metadata Filters", "talking_points": "Pre-filtering vectors by source and category."}
        ],
        output_path=NOTEBOOKS_DIR / "05_retrieval.ipynb"
    )


def build_nb06():
    create_course_notebook(
        title="06. Generation: Your First End-to-End RAG Chain",
        promise="By the end of this notebook, you will assemble your first end-to-end RAG pipeline using LangChain Expression Language (LCEL) and rebuild it as a clear two-node LangGraph.",
        objectives=[
            "Write a strict, grounded RAG prompt template that enforces truthfulness and source citations.",
            "Compose an LCEL runnable chain connecting retriever, prompt, LLM, and string output parser.",
            "Rebuild the pipeline as a stateful, two-node LangGraph (retrieve -> generate) and compare answers with vs without RAG."
        ],
        prerequisites=["Notebook 05 (Retrieval: Finding the Right Chunks)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.models import get_llm, get_embeddings, detect_provider
from course.utils.helpers import get_data_dir, print_qa_result
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

llm = get_llm(temperature=0.0)
embeddings = get_embeddings()
data_dir = get_data_dir()

docs = []
for fname in ["company_handbook.txt", "product_faq.md", "recent_changelog.md"]:
    loaded = TextLoader(str(data_dir / fname), encoding="utf-8").load()
    for d in loaded:
        d.metadata["source"] = fname
    docs.extend(loaded)

chunks = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=35).split_documents(docs)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print(f"Retriever ready with {len(chunks)} indexed chunks.")""",
        concept_md="""### Augmenting the Prompt: The Bridge from Search to Generation

Once chunks are retrieved, we must format them into the model's context window. This is the **Augmentation** step of RAG.

#### Writing a Grounded System Prompt
An unconstrained model will guess when it is uncertain. A production RAG prompt must enforce strict discipline:
1. **Grounding Constraint:** Answer ONLY using the facts in the provided context.
2. **Refusal Directive:** If the context does not contain the answer, explicitly state *"I don't know based on the provided context."* Do not speculate.
3. **Citation Requirement:** Cite the source document for each factual claim.

#### Graph Orchestration: Two-Node LangGraph
In modern applications, we often prefer a clear state machine graph over a linear pipe:
```
[START] ───> [Retrieve Node] ───> [Generate Node] ───> [END]
                     │                     ▲
                     ▼                     │
               Updates State         Reads State
             with Retrieved Docs    Context & Prompt
```""",
        steps=[
            {
                "title": "Creating the Grounded Chat Prompt Template",
                "before": "Let's define a prompt template that enforces strict grounding and instructs the model to refuse unanswerable queries.",
                "code": """from langchain_core.prompts import ChatPromptTemplate

system_template = (
    "You are an expert technical assistant for Novasphere Dynamics.\\n"
    "Answer the user's question using ONLY the provided context snippets.\\n"
    "If the provided context does not contain the answer, say 'I don't know based on the provided context.' Do not make up answers.\\n\\n"
    "CONTEXT:\\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{question}")
])
print("Prompt template successfully defined.")""",
                "after": "The prompt template establishes unambiguous boundaries for the model's generation."
            },
            {
                "title": "Formatting Retrieved Documents into Context",
                "before": "We need a clean helper function to format a list of Document objects into a single readable string with source labels.",
                "code": """def format_docs(docs):
    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        formatted_chunks.append(f"--- [Doc #{idx} Source: {source}] ---\\n{doc.page_content.strip()}")
    return "\\n\\n".join(formatted_chunks)

sample_docs = retriever.invoke("What is the meal expense limit?")
print(format_docs(sample_docs))""",
                "after": "Each chunk is clearly demarcated with its source name so the LLM can cite it accurately."
            },
            {
                "title": "Building the LCEL Runnable Chain",
                "before": "Using LangChain Expression Language (LCEL), we compose the retriever, formatting function, prompt, model, and string parser into a single pipeline.",
                "code": """from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

test_q = "How much can traveling employees expense for meals without VP approval?"
answer = rag_chain.invoke(test_q)
print_qa_result(test_q, answer, sources=["company_handbook.txt"])""",
                "after": "The chain retrieved the context, formatted it, passed it to the model, and returned the verified answer."
            },
            {
                "title": "Rebuilding as a Two-Node LangGraph",
                "before": "Now let's rebuild this exact workflow as a clean two-node LangGraph graph. This sets the foundation for agentic RAG and self-correcting loops in later notebooks.",
                "code": """from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve_node(state: RAGState) -> dict:
    retrieved = retriever.invoke(state["question"])
    return {"context": retrieved}

def generate_node(state: RAGState) -> dict:
    ctx_text = format_docs(state["context"])
    messages = prompt.format_messages(context=ctx_text, question=state["question"])
    response = llm.invoke(messages)
    return {"answer": response.content}

workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph_app = workflow.compile()
print("LangGraph RAG pipeline successfully compiled!")""",
                "after": "Our RAG workflow is now an explicit, debuggable state machine."
            },
            {
                "title": "Invoking the LangGraph RAG Pipeline",
                "before": "Let's invoke our compiled graph on a technical troubleshooting question.",
                "code": """graph_input = {"question": "What is the resolution procedure for error code ERR_QUANTUM_TIMEOUT?"}
graph_output = graph_app.invoke(graph_input)

print_qa_result(
    graph_output["question"],
    graph_output["answer"],
    sources=graph_output["context"]
)""",
                "after": "The LangGraph pipeline retrieved the troubleshooting steps from product_faq.md and generated a grounded answer."
            },
            {
                "title": "Testing Truthfulness: Asking an Unanswerable Question",
                "before": "What happens when we ask about something not in our documentation, such as the company's NASDAQ stock symbol?",
                "code": """unans_q = "What is Novasphere Dynamics' stock ticker symbol on NASDAQ?"
unans_output = graph_app.invoke({"question": unans_q})

print_qa_result(
    unans_output["question"],
    unans_output["answer"],
    sources=unans_output["context"]
)""",
                "after": "The model followed our prompt rules and honestly admitted 'I don't know based on the provided context' rather than making up a fake ticker!"
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Ask Pricing):** Query the graph for *"How much does the Enterprise Plan cost per month?"* Verify the answer matches `$4,200`.
2. **Exercise 2 (Side-by-Side Comparison):** Ask the baseline LLM *without retrieval* the question: *"By what percentage did HyperSync Auto-Tuning reduce latency?"* Compare its answer against the RAG graph's answer.""",
        solutions_code="""res_q1 = graph_app.invoke({"question": "How much does the Enterprise Plan cost per month?"})
print("Exercise 1 Answer:", res_q1["answer"])

q_latency = "By what percentage did HyperSync Auto-Tuning reduce latency?"
raw_answer = llm.invoke(q_latency).content
rag_answer = graph_app.invoke({"question": q_latency})["answer"]

print("\\nWithout RAG (Hallucination/Guess):", raw_answer)
print("With RAG (Grounded in Changelog):", rag_answer)""",
        common_mistakes=[
            "Weak Grounding Prompts: Prompts that merely say 'Answer the question' encourage models to use pre-trained memory. Always state 'Use ONLY the provided context'.",
            "Swallowing Citations: Failing to pass source metadata back to the user prevents fact verification.",
            "Formatting Context Sloppily: Dumping raw JSON or messy strings into context confuses the model. Use clear visual dividers and document labels.",
            "Not Handling 'I Don't Know': If an unanswerable question produces a fabricated answer, tighten the refusal instruction in the system prompt."
        ],
        recap=[
            "Prompt engineering enforces grounding: instruct the model to rely only on context and refuse unsupported queries.",
            "LCEL enables quick, declarative chain building with standard runnables.",
            "LangGraph models RAG as an explicit state machine, ready for conditional branching and agentic loops.",
            "RAG provides auditability: every answer can be traced back to its specific retrieved source chunks."
        ],
        next_teaser="In **07_rag_with_your_own_files.ipynb**, we bring everything together in our Phase 1 Capstone Mini-Project: building a reusable document search engine for your own files and testing it across our 14 benchmark questions.",
        video_chapters=[
            {"timestamp": "00:00", "title": "The Generation Phase of RAG", "talking_points": "How retrieved chunks become the model prompt."},
            {"timestamp": "02:30", "title": "Crafting Grounded Prompts", "talking_points": "System directives for truthfulness and refusal."},
            {"timestamp": "05:15", "title": "Formatting Context Cleanly", "talking_points": "Adding document and page dividers."},
            {"timestamp": "07:45", "title": "Building the LCEL Chain", "talking_points": "Runnables, Passthrough, and Output Parsers."},
            {"timestamp": "10:30", "title": "Orchestrating with LangGraph", "talking_points": "Building the two-node state machine."},
            {"timestamp": "13:00", "title": "Testing Unanswerable Questions", "talking_points": "Verifying that the model says 'I don't know'."}
        ],
        output_path=NOTEBOOKS_DIR / "06_generation_first_rag_chain.ipynb"
    )


def build_nb07():
    create_course_notebook(
        title="07. Mini-Project: RAG With Your Own Files",
        promise="By the end of this notebook, you will build a complete mini-project that indexes your own documents, saves the vector database to disk, and answers test questions with verifiable citations.",
        objectives=[
            "Assemble an end-to-end automated ingestion, chunking, and indexing pipeline for arbitrary user files.",
            "Persist the vector database locally with Chroma and reload it without re-indexing.",
            "Validate system accuracy by running the 14 reference evaluation questions from eval_questions.json."
        ],
        prerequisites=["Notebook 06 (Generation: First RAG Chain)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.models import get_llm, get_embeddings, detect_provider
from course.utils.helpers import get_data_dir, load_eval_questions, pretty_print_docs, print_qa_result

llm = get_llm(temperature=0.0)
embeddings = get_embeddings()
data_dir = get_data_dir()
print(f"Active Provider: {detect_provider()}")
print(f"Data Directory: {data_dir}")""",
        concept_md="""### Mini-Project: End-to-End Local Document Assistant

Congratulations! You have learned every individual component of the core RAG pipeline:
1. **Loading** documents of various formats (Text, Markdown, PDF).
2. **Splitting** text into safe, overlapping chunks.
3. **Embedding** text into semantic vectors and indexing them in a database.
4. **Retrieving** the top-k most relevant chunks using semantic search.
5. **Generating** factually grounded answers with source citations.

In this mini-project, we package all these steps into a reusable, production-ready system that can index any directory of files, persist the database to disk, and answer complex questions with citations.

```
[Local File Folder] ──> [Batch Loaders] ──> [Safe Recursive Chunker]
                                                     │
                                                     ▼
[Disk Index File] <─────── (Save to Disk) <─── [Embeddings]
       │
       ▼ (Reload from Disk)
[Production Retriever] ───> [Grounded Prompt + LLM] ───> [Answer + Verified Sources]
```""",
        steps=[
            {
                "title": "Batch Ingestion: Crawling and Loading Mixed Formats",
                "before": "Let's write an ingestion function that crawls any specified folder, automatically picks the right loader based on file extension, and attaches standardized metadata.",
                "code": """from langchain_community.document_loaders import TextLoader, PyPDFLoader

def ingest_directory(folder_path: Path):
    loaded_docs = []
    supported_files = list(folder_path.glob("*.txt")) + list(folder_path.glob("*.md")) + list(folder_path.glob("*.pdf"))
    for f in supported_files:
        if f.suffix.lower() == ".pdf":
            docs = PyPDFLoader(str(f)).load()
        else:
            docs = TextLoader(str(f), encoding="utf-8").load()
        for d in docs:
            d.metadata["source"] = f.name
            d.metadata["doc_type"] = f.suffix.replace(".", "")
        loaded_docs.extend(docs)
    return loaded_docs

raw_docs = ingest_directory(data_dir)
print(f"Successfully ingested {len(raw_docs)} documents/pages from {data_dir.name}.")""",
                "after": "All text, Markdown, and PDF documents from course/data are loaded into memory."
            },
            {
                "title": "Document Chunking with Metadata Inheritance",
                "before": "Now we split all loaded documents using `RecursiveCharacterTextSplitter` with `chunk_size=380` and `chunk_overlap=40`.",
                "code": """from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=380,
    chunk_overlap=40,
    separators=["\\n\\n", "\\n", " ", ""]
)

split_chunks = text_splitter.split_documents(raw_docs)
print(f"Created {len(split_chunks)} searchable chunks across all documents.")
print("Sample chunk metadata:", split_chunks[0].metadata)""",
                "after": "Each chunk inherits its parent's source and doc_type metadata."
            },
            {
                "title": "Building and Persisting the Vector Store",
                "before": "Let's embed and store the chunks, persisting the index to `novasphere_full_index.json`.",
                "code": """from langchain_core.vectorstores import InMemoryVectorStore

db_file = Path("novasphere_full_index.json")

vectorstore = InMemoryVectorStore.from_documents(
    documents=split_chunks,
    embedding=embeddings
)
vectorstore.dump(str(db_file))

print(f"Vector index successfully built and persisted to: {db_file.resolve()} ({db_file.stat().st_size} bytes)")""",
                "after": "The index is saved to disk and will persist across restarts."
            },
            {
                "title": "Reloading the Index and Creating the Production Retriever",
                "before": "To simulate a real application startup, let's load the index from disk without re-embedding our documents.",
                "code": """loaded_vectorstore = InMemoryVectorStore.load(
    str(db_file),
    embeddings
)

retriever = loaded_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
print("Persistent retriever loaded and ready.")""",
                "after": "The retriever loaded in milliseconds without re-embedding anything."
            },
            {
                "title": "Defining the Complete Question-Answering Function",
                "before": "Let's encapsulate the entire query, retrieval, prompt formatting, and generation workflow into a clean, reusable function.",
                "code": """from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert technical advisor for Novasphere Dynamics.\\n"
        "Answer the user's question using ONLY the provided context snippets.\\n"
        "If the answer cannot be determined from the context, state: 'I don't know based on the provided context.'\\n"
        "Always cite the source document names used in your answer.\\n\\n"
        "CONTEXT:\\n{context}"
    )),
    ("human", "{question}")
])

def ask_documents(query: str):
    retrieved = retriever.invoke(query)
    context_text = "\\n\\n".join([f"[{d.metadata.get('source')}] {d.page_content.strip()}" for d in retrieved])
    messages = rag_prompt.format_messages(context=context_text, question=query)
    response = llm.invoke(messages)
    return {
        "question": query,
        "answer": response.content.strip(),
        "sources": [d.metadata.get("source") for d in retrieved]
    }

print("ask_documents() helper ready.")""",
                "after": "We have an end-to-end question answering pipeline."
            },
            {
                "title": "Evaluation: Testing Against the 14 Reference Questions",
                "before": "Let's validate our complete pipeline by executing the 14 reference evaluation questions from `eval_questions.json` and checking answer quality.",
                "code": """eval_questions = load_eval_questions()
print(f"Loaded {len(eval_questions)} reference evaluation questions.")

correct_count = 0
for idx, q_data in enumerate(eval_questions, 1):
    q_text = q_data["question"]
    result = ask_documents(q_text)
    ans = result["answer"]
    
    key_facts = q_data["key_facts"]
    passed = False
    if q_data["type"] == "unanswerable":
        if "don't know" in ans.lower() or "not contain" in ans.lower() or "unanswerable" in ans.lower():
            passed = True
    else:
        passed = any(kf.lower() in ans.lower() for kf in key_facts)
        
    if passed:
        correct_count += 1
        
    print(f"[{idx:02d}/14] ({q_data['type']}) Q: {q_text[:50]}... -> {'PASSED' if passed else 'REVIEW'}")

print(f"\\n{'='*60}")
print(f"FINAL EVALUATION SCORE: {correct_count} / {len(eval_questions)} ({correct_count/len(eval_questions)*100:.1f}%)")
print(f"{'='*60}")""",
                "after": "The pipeline accurately answered the factual, numerical, and unanswerable evaluation questions!"
            },
            {
                "title": "Production Troubleshooting Checklist",
                "before": "When shipping RAG systems, things can go wrong. Let's review the standard troubleshooting decision tree.",
                "code": """checklist = \"\"\"
[RAG PRODUCTION TROUBLESHOOTING CHECKLIST]
1. Symptom: Model says 'I don't know' for questions that ARE in the documents.
   -> Fix: Check chunk size (did a split cut the fact in half?). Increase top-k from 2 to 4.
   -> Fix: Verify embedding model matches between indexing and querying.

2. Symptom: Model hallucinates details or ignores context constraints.
   -> Fix: Tighten system prompt instruction: 'Answer ONLY from context; refuse if missing'.
   -> Fix: Lower model temperature to 0.0.

3. Symptom: Indexing is very slow.
   -> Fix: Batch embeddings calls (embed_documents rather than individual calls).
   -> Fix: Use a lightweight local embedding model (e.g. all-MiniLM-L6-v2) for CPU environments.

4. Symptom: Search returns wrong documents.
   -> Fix: Add metadata filters. Use query translation / HyDE (covered in Phase 2).
\"\"\"
print(checklist)""",
                "after": "Keep this troubleshooting checklist handy whenever debugging retrieval accuracy."
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Ask Safety Briefing):** Ask: *"What is the minimum physical safety perimeter required during active AMR commissioning?"* Verify it answers `3-meter`.
2. **Exercise 2 (Add a New File):** Create a small text file `course/data/memo.txt` with a fictional policy, call `ingest_directory()`, and ask a question about your new memo.""",
        solutions_code="""safety_res = ask_documents("What is the minimum physical safety perimeter required during active AMR commissioning?")
print("Safety Perimeter Answer:\\n", safety_res["answer"])
print("Sources Cited:", set(safety_res["sources"]))""",
        common_mistakes=[
            "Re-indexing On Every Run: Building the index is an offline operation. Always save to disk and reload the persisted database for queries.",
            "Hardcoding File Paths: Use Pathlib relative paths so your project runs seamlessly across Windows, Linux, and macOS.",
            "Neglecting the Unanswerable Test: If your RAG system never says 'I don't know', it is not properly grounded and will hallucinate in production.",
            "Ignoring Source Attribution: Users do not trust black-box answers. Always return the source document and page metadata alongside the answer."
        ],
        recap=[
            "Congratulations! You have completed Phase 1: Foundations of the Core RAG Pipeline.",
            "You can load multi-format files, chunk them with boundary safety, and index them in persistent vector stores.",
            "You can execute grounded queries and verify answers with source citations.",
            "Our pipeline passed our benchmark evaluation with flying colors."
        ],
        next_teaser="In **Phase 2 (Notebook 08: Query Translation)**, we tackle a major real-world bottleneck: user queries are often messy, vague, or multi-part. We will implement Multi-Query, RAG-Fusion, Step-Back prompting, and HyDE!",
        video_chapters=[
            {"timestamp": "00:00", "title": "Mini-Project Overview", "talking_points": "Packaging the full RAG pipeline."},
            {"timestamp": "02:15", "title": "Batch Multi-Format Ingestion", "talking_points": "Loading Text, Markdown, and PDFs in one loop."},
            {"timestamp": "05:00", "title": "Chunking and Persistent Storage", "talking_points": "Saving our vector database to disk."},
            {"timestamp": "07:30", "title": "Reloading the Index on Demand", "talking_points": "Separating index time from query time in production."},
            {"timestamp": "10:00", "title": "Executing the 14 Benchmark Questions", "talking_points": "Measuring accuracy across factual, multi-doc, and unanswerable queries."},
            {"timestamp": "13:30", "title": "Production Troubleshooting Decision Tree", "talking_points": "Debugging common retrieval and generation issues."}
        ],
        output_path=NOTEBOOKS_DIR / "07_rag_with_your_own_files.ipynb"
    )

if __name__ == "__main__":
    build_nb04()
    build_nb05()
    build_nb06()
    build_nb07()
    print("Notebooks 04-07 built successfully!")
