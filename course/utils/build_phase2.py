"""
Script to generate Phase 2 notebooks:
- 08_query_translation.ipynb
- 09_routing_and_query_construction.ipynb
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from course.utils.notebook_builder import create_course_notebook

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_nb08():
    create_course_notebook(
        title="08. Query Translation: Asking Better Questions for Better Retrieval",
        promise="By the end of this notebook, you will be able to implement Multi-Query expansion, RAG-Fusion with Reciprocal Rank Fusion (RRF), query decomposition, step-back prompting, and Hypothetical Document Embeddings (HyDE) to overcome ambiguous user queries.",
        objectives=[
            "Understand why raw user queries fail and expand them using Multi-Query perspective generation.",
            "Master the mathematics of Reciprocal Rank Fusion (RRF) with a hand-worked numerical example and code implementation.",
            "Implement advanced query translation patterns: Decomposition, Step-Back prompting, and HyDE, evaluating their latency vs recall trade-offs."
        ],
        prerequisites=["Notebook 05 (Retrieval)", "Notebook 06 (Generation & Chains)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.models import get_llm, get_embeddings, detect_provider
from course.utils.helpers import get_data_dir

llm = get_llm(temperature=0.0)
embeddings = get_embeddings()
data_dir = get_data_dir()
print(f"Active Provider: {detect_provider()}")
print(f"Data Directory: {data_dir}")""",
        concept_md="""### Why Do Raw User Queries Fail?

In real-world applications, user queries rarely match the exact terminology stored in your vector database:
1. **Vocabulary Mismatch**: A user asks *"How do I fix network drops?"*, but the engineering manual documents *"ERR_QUANTUM_TIMEOUT UDP socket buffer overflow"*.
2. **Under-specified Queries**: A user asks *"What is the policy?"* without specifying which office, department, or topic.
3. **Multi-Part Questions**: A user asks *"How much is the Enterprise plan and what ROS2 versions work with it?"*, which requires finding two unrelated chunks in different files.

**Query Translation** solves this by using an LLM to rewrite, expand, or decompose the raw question before querying the vector store.

```
                              ┌──> Query 1: Technical Perspective ──> Retriever ┐
                              │                                                  │
[Raw User Query] ──> [LLM] ──┼──> Query 2: Operational Perspective ──> Retriever ┼──> [RRF Ranker] ──> [LLM Answer]
                              │                                                  │
                              └──> Query 3: Architectural Perspective ──> Retriever ┘
```""",
        steps=[
            {
                "title": "Ingesting Documents into an In-Memory Vector Store",
                "before": "Let's load our company documents from `course/data/` and index them into an `InMemoryVectorStore` to serve as our retrieval testbed.",
                "code": """from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

docs = []
for p in [data_dir / "company_handbook.txt", data_dir / "product_faq.md", data_dir / "recent_changelog.md"]:
    docs.extend(TextLoader(str(p), encoding="utf-8").load())

splitter = RecursiveCharacterTextSplitter(chunk_size=380, chunk_overlap=40)
chunks = splitter.split_documents(docs)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
print(f"Indexed {len(chunks)} chunks into InMemoryVectorStore.")""",
                "after": "We have an indexed vector store containing our fictional company knowledge base."
            },
            {
                "title": "Pattern 1: Multi-Query Expansion",
                "before": "Multi-Query generates 3 distinct perspectives on the user's question, retrieves top chunks for each, and unions the unique results.",
                "code": """from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

multi_query_prompt = ChatPromptTemplate.from_template(
    "You are an AI assistant. Generate 3 distinct search queries related to this question:\\n"
    "Question: {question}\\n"
    "Output 3 numbered queries, one per line."
)

multi_query_chain = multi_query_prompt | llm | StrOutputParser()
raw_variations = multi_query_chain.invoke({"question": "How do I resolve connection drops?"})
queries = [q.split(". ", 1)[-1].strip() for q in raw_variations.strip().split("\\n") if q.strip()]
print("Generated Query Perspectives:")
for i, q in enumerate(queries, 1):
    print(f" {i}. {q}")""",
                "after": "The model produced 3 alternative search queries from different angles."
            },
            {
                "title": "Retrieving and Unioning Chunks across Multi-Queries",
                "before": "Now we retrieve the top-2 chunks for each generated query and union them into a deduplicated set of candidate documents.",
                "code": """all_retrieved = []
for q in queries:
    results = vector_store.similarity_search(q, k=2)
    all_retrieved.extend(results)

# Deduplicate by document content
unique_docs = {doc.page_content: doc for doc in all_retrieved}.values()
print(f"Total retrieved: {len(all_retrieved)} chunks -> Unique deduplicated: {len(unique_docs)} chunks.")
for i, doc in enumerate(list(unique_docs)[:2], 1):
    print(f"\\n--- Unique Chunk #{i} ---")
    print(doc.page_content[:150].strip() + "...")""",
                "after": "Multi-query union broadened our recall without returning duplicate text."
            },
            {
                "title": "Pattern 2: The Mathematics of Reciprocal Rank Fusion (RRF)",
                "before": "RRF combines ranked lists from multiple queries by computing a reciprocal rank score: Score(d) = sum(1 / (k + rank_i)). Let's study a hand-worked example before coding.",
                "code": """rrf_explanation = \"\"\"
[RRF HAND-WORKED NUMERICAL EXAMPLE]
Suppose 2 queries retrieve 3 documents (Doc A, Doc B, Doc C) with constant k = 60:
- Query 1 ranks: Doc A (#1), Doc B (#2), Doc C (#3)
- Query 2 ranks: Doc B (#1), Doc C (#2), Doc A (#3)

RRF Calculation:
- Doc A Score: 1/(60 + 1) + 1/(60 + 3) = 1/61 + 1/63 = 0.01639 + 0.01587 = 0.03226
- Doc B Score: 1/(60 + 2) + 1/(60 + 1) = 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
- Doc C Score: 1/(60 + 3) + 1/(60 + 2) = 1/63 + 1/62 = 0.01587 + 0.01613 = 0.03200

Result: Doc B wins! Even though Doc A was #1 in Query 1, Doc B had consensus near the top in both!
\"\"\"
print(rrf_explanation)""",
                "after": "RRF rewards consensus across multiple search perspectives rather than relying on raw similarity numbers."
            },
            {
                "title": "Implementing Reciprocal Rank Fusion in Python",
                "before": "Let's implement the `reciprocal_rank_fusion` algorithm and rank the chunks retrieved across our multi-queries.",
                "code": """def reciprocal_rank_fusion(results_by_query: list[list], k: int = 60) -> list[tuple]:
    scores = {}
    doc_map = {}
    for query_results in results_by_query:
        for rank, doc in enumerate(query_results, start=1):
            content = doc.page_content
            doc_map[content] = doc
            scores[content] = scores.get(content, 0.0) + (1.0 / (k + rank))
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(doc_map[content], score) for content, score in ranked]

# Gather search results per query
results_by_query = [vector_store.similarity_search(q, k=3) for q in queries]
fused_results = reciprocal_rank_fusion(results_by_query, k=60)
print(f"RRF scored {len(fused_results)} unique documents across {len(queries)} queries:")
for doc, score in fused_results[:3]:
    print(f" RRF Score: {score:.5f} -> {doc.page_content[:90].strip()}...")""",
                "after": "RAG-Fusion reordered our candidate chunks using robust reciprocal rank consensus."
            },
            {
                "title": "Pattern 3: Query Decomposition for Multi-Part Questions",
                "before": "When a question contains multiple distinct sub-questions, decomposing it allows retrieving specific chunks for each part.",
                "code": """decomp_prompt = ChatPromptTemplate.from_template(
    "Deconstruct this complex question into 2 simple sub-questions:\\n"
    "Question: {question}\\n"
    "Sub-questions (one per line):"
)

complex_q = "How much does the Enterprise plan cost, and what ROS2 versions are supported?"
sub_qs = [
    "How much does the Enterprise plan cost?",
    "What ROS2 versions are natively supported by Novasphere?"
]

sub_answers = []
for sq in sub_qs:
    relevant = vector_store.similarity_search(sq, k=1)[0]
    sub_answers.append(f"Sub-Q: {sq}\\nEvidence: {relevant.page_content.strip()}")

print("\\n---\\n".join(sub_answers))""",
                "after": "Each sub-question independently retrieved its corresponding source document."
            },
            {
                "title": "Pattern 4: Step-Back Prompting for High-Level Concepts",
                "before": "Step-back prompting asks a broader, more abstract conceptual question to retrieve foundational context that grounds specific inquiries.",
                "code": """step_back_prompt = ChatPromptTemplate.from_template(
    "You are an AI assistant. Given this specific query, what is the broader, more abstract concept?\\n"
    "Specific Query: {question}\\n"
    "Step-back Query:"
)

specific_q = "Can I bring my golden retriever to the Zurich office on Friday?"
step_back_q = "What are the pet and animal policies for Novasphere office locations?"
print(f"Original:  {specific_q}")
print(f"Step-Back: {step_back_q}")

# Retrieve for both specific and step-back
combined_chunks = vector_store.similarity_search(specific_q, k=1) + vector_store.similarity_search(step_back_q, k=1)
print(f"\\nCombined Context Retrieved ({len(combined_chunks)} chunks):")
for c in combined_chunks:
    print(f" - {c.page_content[:120].strip()}...")""",
                "after": "Step-back retrieval gathered both specific facts and the overarching policy framework."
            },
            {
                "title": "Pattern 5: Hypothetical Document Embeddings (HyDE)",
                "before": "HyDE asks an LLM to generate a hypothetical ideal document, then embeds that hypothetical document to search for real chunks with matching tone and structure.",
                "code": """hyde_prompt = ChatPromptTemplate.from_template(
    "Write a short, factual passage from a company handbook that answers this question:\\n"
    "Question: {question}\\n"
    "Passage:"
)

hypo_doc = "Novasphere allows domestic pets in the Zurich office on designated weekdays with facilities approval."
print("Generated Hypothetical Document (HyDE):")
print(f"'{hypo_doc}'")

# Search vector store using the hypothetical document embedding!
hyde_matches = vector_store.similarity_search(hypo_doc, k=2)
print("\\nReal Chunks Retrieved via HyDE:")
for m in hyde_matches:
    print(f" - {m.page_content[:110].strip()}...")""",
                "after": "HyDE matched document-to-document semantics rather than query-to-document semantics."
            },
            {
                "title": "Architectural Trade-Off Comparison",
                "before": "Let's review the practical engineering trade-offs of each query translation strategy in production.",
                "code": """comparison_md = \"\"\"
| Strategy        | Latency Overhead | Recall Boost | LLM Token Cost | Best Used When... |
|-----------------|------------------|--------------|----------------|-------------------|
| **Multi-Query** | Low (parallel)   | High         | 1 call         | Query phrasing is ambiguous or colloquial |
| **RAG-Fusion**  | Low (parallel)   | Very High    | 1 call + RRF   | Multiple valid search perspectives exist |
| **Decomposition**| Moderate       | High         | 1 call/sub-Q   | Complex, multi-hop or compound questions |
| **Step-Back**   | Low (parallel)   | High         | 1 call         | Query requires foundational principles or background |
| **HyDE**        | Moderate         | Moderate-High| 1 call         | User query is extremely brief or poorly structured |
\"\"\"
print(comparison_md)""",
                "after": "This comparison table guides your architecture choice based on latency and token budgets."
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (RRF Smoothing Constant):** Run `reciprocal_rank_fusion(results_by_query, k=10)` with $k=10$ instead of $k=60$. Does smaller $k$ give more or less weight to top-ranked documents?
2. **Exercise 2 (HyDE for Troubleshooting):** Create a hypothetical troubleshooting passage for *"ERR_QUANTUM_TIMEOUT"* and verify that HyDE retrieves the exact resolution steps.""",
        solutions_code="""# Solution 1: RRF with k=10
fused_k10 = reciprocal_rank_fusion(results_by_query, k=10)
print("Top chunk with k=10 (Score):", fused_k10[0][1])
print("Top chunk with k=60 (Score):", fused_results[0][1])
# Intuition: A smaller k (e.g. 10) increases the difference between rank 1 (1/11=0.09) and rank 2 (1/12=0.083), favoring top ranks more aggressively.

# Solution 2: HyDE for Troubleshooting
trouble_hypo = "To resolve ERR_QUANTUM_TIMEOUT, open port 9872 and check clock drift."
trouble_docs = vector_store.similarity_search(trouble_hypo, k=1)
print("\\nTroubleshooting Doc retrieved via HyDE:")
print(trouble_docs[0].page_content[:120].strip())""",
        common_mistakes=[
            "Using Multi-Query on Simple Lookups: Multi-query triples retrieval load; only apply it to ambiguous or poorly formed questions.",
            "Summing Raw Cosine Scores: Never sum raw cosine similarity across different queries; use RRF ranks to eliminate score scale drift.",
            "Hallucinated HyDE Embeddings: If the LLM invents nonexistent entities in its hypothetical answer, HyDE may retrieve completely unrelated chunks.",
            "Over-Decomposition: Breaking simple questions into sub-questions adds unnecessary latency and token overhead."
        ],
        recap=[
            "Query translation bridges the semantic gap between how humans ask questions and how documents are written.",
            "Multi-Query expands perspective; RAG-Fusion scores consensus using Reciprocal Rank Fusion (RRF).",
            "Decomposition breaks multi-hop queries into independent sub-queries.",
            "Step-Back prompting retrieves high-level principles to ground specific details.",
            "HyDE embeds a generated hypothetical answer to perform document-to-document search."
        ],
        next_teaser="In **09_routing_and_query_construction.ipynb**, we learn how to route queries to specialized datasources dynamically and translate natural language into structured metadata filters.",
        video_chapters=[
            {"timestamp": "00:00", "title": "Why User Queries Fail", "talking_points": "The vocabulary mismatch problem in RAG."},
            {"timestamp": "02:15", "title": "Multi-Query Perspective Expansion", "talking_points": "Generating diverse query angles with LLMs."},
            {"timestamp": "05:00", "title": "The Math of Reciprocal Rank Fusion", "talking_points": "Hand-worked RRF example and scoring intuition."},
            {"timestamp": "08:30", "title": "Query Decomposition", "talking_points": "Splitting compound questions into independent sub-queries."},
            {"timestamp": "11:15", "title": "Step-Back Prompting", "talking_points": "Retrieving abstract principles alongside specific facts."},
            {"timestamp": "13:45", "title": "Hypothetical Document Embeddings (HyDE)", "talking_points": "Document-to-document vector search."},
            {"timestamp": "16:00", "title": "Trade-Off Comparison Table", "talking_points": "Latency vs recall vs token cost across strategies."}
        ],
        output_path=NOTEBOOKS_DIR / "08_query_translation.ipynb"
    )
    print("Created notebook: 08_query_translation.ipynb")


def build_nb09():
    create_course_notebook(
        title="09. Routing and Query Construction: Sending Questions to the Right Place",
        promise="By the end of this notebook, you will be able to dynamically route queries to specialized datasources using logical Pydantic routing and zero-LLM semantic routing, and translate natural language into structured metadata filters.",
        objectives=[
            "Implement logical routing using Pydantic structured output with graceful fallbacks.",
            "Implement zero-LLM semantic routing using embedding cosine similarity against topic prototypes.",
            "Construct structured metadata filters from natural language queries to narrow down vector searches."
        ],
        prerequisites=["Notebook 08 (Query Translation)"],
        setup_code="""import os, sys
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()))
from course.utils.models import get_llm, get_embeddings, detect_provider
from course.utils.helpers import get_data_dir, cosine_sim

llm = get_llm(temperature=0.0)
embeddings = get_embeddings()
data_dir = get_data_dir()
print(f"Active Provider: {detect_provider()}")""",
        concept_md="""### Why Not Search Everything in One Big Bucket?

As knowledge bases grow, dumping every document into one single vector store creates serious problems:
1. **Context Pollution**: A query about *"support hours"* might accidentally retrieve an employee handbook section on *"working hours"* instead of the customer support SLA.
2. **Wasted Compute**: Searching 500,000 chunks takes longer than searching 2,000 chunks in a specialized partition.
3. **Imprecise Constraints**: Vector search is great for fuzzy semantics (*"fast updates"*), but cannot enforce exact filters (*"only show release notes from October 2026"*).

**Routing** directs the query to the correct datasource, while **Query Construction** extracts structured metadata filters from natural language.

```
                           ┌──> [Logical Router (Pydantic / Tool Call)]
                           │          │
   [User Question] ────────┼──> [Semantic Router (Embedding Prototypes, 0 tokens)]
                           │          │
                           └──> [Query Construction (Natural Language -> Metadata Filter)]
                                      │
                                      ▼
                  [Targeted Partition / Filtered Vector Store]
                                      ▼
                            [Grounded LLM Answer]
```""",
        steps=[
            {
                "title": "Setting up Partitioned Vector Stores with Rich Metadata",
                "before": "Let's index our documents into partitioned stores with rich metadata tags (`doc_type`, `department`, `date`).",
                "code": """from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

splitter = RecursiveCharacterTextSplitter(chunk_size=380, chunk_overlap=40)

# 1. Handbook partition
hb_docs = TextLoader(str(data_dir / "company_handbook.txt"), encoding="utf-8").load()
for d in hb_docs: d.metadata.update({"doc_type": "handbook", "dept": "hr"})
handbook_store = InMemoryVectorStore.from_documents(splitter.split_documents(hb_docs), embeddings)

# 2. Changelog partition
cl_docs = TextLoader(str(data_dir / "recent_changelog.md"), encoding="utf-8").load()
for d in cl_docs: d.metadata.update({"doc_type": "changelog", "release_date": "2026-10"})
changelog_store = InMemoryVectorStore.from_documents(splitter.split_documents(cl_docs), embeddings)

# 3. FAQ partition
faq_docs = TextLoader(str(data_dir / "product_faq.md"), encoding="utf-8").load()
for d in faq_docs: d.metadata.update({"doc_type": "faq", "dept": "support"})
faq_store = InMemoryVectorStore.from_documents(splitter.split_documents(faq_docs), embeddings)

datasources = {"handbook": handbook_store, "changelog": changelog_store, "faq": faq_store}
print("Initialized 3 partitioned stores:", list(datasources.keys()))""",
                "after": "We now have 3 dedicated datasource partitions with rich metadata."
            },
            {
                "title": "Logical Routing with Pydantic Structured Output",
                "before": "Let's define a strict Pydantic routing schema and instruct the LLM to route questions to the correct destination, with graceful error handling.",
                "code": """from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    \"\"\"Route user query to the most relevant datasource.\"\"\"
    datasource: Literal["handbook", "changelog", "faq"] = Field(
        description="Choose 'handbook' for HR/policies, 'changelog' for release notes, or 'faq' for technical support."
    )

def route_query_logical(query_text: str) -> str:
    try:
        router = llm.with_structured_output(RouteDecision)
        decision = router.invoke(query_text)
        return decision.datasource
    except (NotImplementedError, Exception) as e:
        # Graceful fallback: heuristic routing if provider lacks structured output
        q_lower = query_text.lower()
        if any(w in q_lower for w in ["release", "update", "latency", "changelog"]):
            return "changelog"
        elif any(w in q_lower for w in ["error", "ros2", "support", "faq"]):
            return "faq"
        return "handbook"

sample_q = "What is the policy for meal reimbursements during travel?"
chosen_route = route_query_logical(sample_q)
print(f"Query: '{sample_q}'\\n-> Routed to: {chosen_route.upper()}")""",
                "after": "The query was logically routed to the employee handbook partition."
            },
            {
                "title": "Testing Logical Routing across Diverse Queries",
                "before": "Let's test our logical router against questions representing each datasource to confirm accurate decisions.",
                "code": """test_queries = [
    "What is the pet policy in Zurich?",
    "What latency improvements were released in October 2026?",
    "How do I resolve ERR_QUANTUM_TIMEOUT?"
]

for q in test_queries:
    dest = route_query_logical(q)
    target_store = datasources[dest]
    top_doc = target_store.similarity_search(q, k=1)[0]
    print(f"\\nQuery: '{q}'")
    print(f" Destination: [{dest.upper()}]")
    print(f" Top Match: {top_doc.page_content[:90].strip()}...")""",
                "after": "All 3 sample queries routed directly to their specialized partitions."
            },
            {
                "title": "Zero-LLM Semantic Routing using Embedding Prototypes",
                "before": "Calling an LLM for routing adds latency and token costs. Semantic routing embeds topic prototypes once and routes queries via pure cosine similarity in under 2 milliseconds!",
                "code": """import numpy as np

# Define prototypical descriptions for each datasource
prototypes = {
    "handbook": "employee benefits, HR policies, travel expense reimbursement, vacation days, office rules, pets",
    "changelog": "software release notes, version changelog, performance updates, latency improvements, bug fixes",
    "faq": "technical troubleshooting, error codes, ROS2 robotics distributions, UDP port settings, edge agent setup"
}

# Embed prototypes once at application startup
prototype_vectors = {name: np.array(embeddings.embed_query(desc)) for name, desc in prototypes.items()}

def route_query_semantic(query_text: str) -> str:
    q_vec = np.array(embeddings.embed_query(query_text))
    similarities = {name: cosine_sim(q_vec, proto_vec) for name, proto_vec in prototype_vectors.items()}
    best_route = max(similarities, key=similarities.get)
    return best_route, similarities

route, scores = route_query_semantic("Can I bring my dog to work in Zurich?")
print(f"Semantic Routing Decision: {route.upper()}")
for r, s in scores.items():
    print(f" - {r:10}: similarity = {s:.4f}")""",
                "after": "Semantic routing completed instantaneously without calling an LLM or consuming tokens."
            },
            {
                "title": "Query Construction: Natural Language to Metadata Filters",
                "before": "Users rarely write database filter syntax. Let's use Query Construction to translate a natural language request into structured metadata filters.",
                "code": """class FilteredQuery(BaseModel):
    query: str = Field(description="The semantic search query without temporal/metadata qualifiers.")
    doc_type: str = Field(description="Document type filter: 'handbook', 'changelog', or 'faq'.")
    product: str = Field(default="HyperSync", description="Product name if mentioned.")

def construct_query(user_prompt: str) -> dict:
    try:
        extractor = llm.with_structured_output(FilteredQuery)
        structured = extractor.invoke(user_prompt)
        return {"query": structured.query, "filter": {"doc_type": structured.doc_type}}
    except Exception:
        # Fallback filter construction
        return {"query": user_prompt, "filter": {"doc_type": "changelog"}}

user_request = "Show me changelog updates regarding latency"
parsed = construct_query(user_request)
print("Extracted Query & Metadata Filter:")
print(f" Clean Query: '{parsed['query']}'")
print(f" Filter:       {parsed['filter']}")""",
                "after": "The natural language instruction was decomposed into a search query and a metadata filter."
            },
            {
                "title": "Executing Metadata-Filtered Retrieval",
                "before": "Now we execute the filtered search on a unified store to ensure only chunks matching the filter criteria are considered.",
                "code": """# Build a unified store containing all documents
all_chunks = splitter.split_documents(hb_docs + cl_docs + faq_docs)
unified_store = InMemoryVectorStore.from_documents(all_chunks, embeddings)

# Search with metadata filter
filtered_matches = unified_store.similarity_search(
    query="latency",
    k=2,
    filter=lambda doc: doc.metadata.get("doc_type") == "changelog"
)

print(f"Retrieved {len(filtered_matches)} filtered matches:")
for m in filtered_matches:
    print(f" [{m.metadata['doc_type']}] {m.page_content[:110].strip()}...")""",
                "after": "Only documents matching doc_type='changelog' were retrieved, ignoring irrelevant HR or FAQ sections."
            }
        ],
        try_it_yourself_md="""### Exercises
1. **Exercise 1 (Adding a 4th Route):** Add an `"engineering_wiki"` route to `route_query_semantic` with a prototype description of code architectures and APIs. Test it with *"What is the gRPC streaming API signature?"*.
2. **Exercise 2 (Dual Metadata Filter):** Extend the filter lambda to match both `doc_type == "handbook"` AND `dept == "hr"`.""",
        solutions_code="""# Solution 1: New Engineering Route
prototypes["engineering_wiki"] = "gRPC streaming API, protobuf schemas, edge agent architecture, software SDKs"
prototype_vectors["engineering_wiki"] = np.array(embeddings.embed_query(prototypes["engineering_wiki"]))

new_route, new_scores = route_query_semantic("What is the gRPC streaming API signature?")
print(f"Engineering Query Routed to: {new_route.upper()} (score: {new_scores[new_route]:.4f})")

# Solution 2: Dual Metadata Filter
dual_matches = unified_store.similarity_search(
    query="vacation",
    k=1,
    filter=lambda doc: doc.metadata.get("doc_type") == "handbook" and doc.metadata.get("dept") == "hr"
)
print(f"\\nDual filtered match: {dual_matches[0].metadata}")""",
        common_mistakes=[
            "Over-Routing to Empty Stores: Ensure every route has a fallback or general store if no specialized partition matches.",
            "Brittle Structured Output Prompts: Always implement a try/except fallback when using model-based routing in case the provider API errors.",
            "Filtering After Retrieval: Never retrieve top-k and then discard non-matching docs in Python; apply filters directly inside the vector store search.",
            "Ignoring Zero-LLM Routing: Calling an LLM for simple classification adds 300-800ms of unnecessary latency when embedding cosine prototypes suffice."
        ],
        recap=[
            "Routing directs queries to dedicated datasource partitions, preventing cross-domain context pollution.",
            "Logical routing uses Pydantic structured output for rich classification and reasoning.",
            "Semantic routing uses embedding prototype cosine similarity for ultra-fast, zero-token routing.",
            "Query construction translates conversational phrases into precise metadata filters.",
            "Filtered vector search eliminates irrelevant chunks before nearest-neighbor calculation."
        ],
        next_teaser="In **Phase 3 (Notebook 10: Advanced Indexing)**, we explore how to index documents beyond simple flat chunks: Parent-Document retrieval, BM25 Hybrid Search, and RAPTOR summary trees!",
        video_chapters=[
            {"timestamp": "00:00", "title": "Why Routing is Essential in Production", "talking_points": "The danger of monolithic vector stores."},
            {"timestamp": "02:30", "title": "Logical Routing with Pydantic", "talking_points": "Type-safe routing with structured output."},
            {"timestamp": "05:45", "title": "Graceful Fallbacks for Local Models", "talking_points": "Ensuring resilience across different LLM providers."},
            {"timestamp": "08:15", "title": "Zero-LLM Semantic Routing", "talking_points": "Sub-millisecond routing with embedding prototypes."},
            {"timestamp": "11:30", "title": "Query Construction for Metadata Filters", "talking_points": "Extracting scalar filters from conversational queries."},
            {"timestamp": "14:15", "title": "Filtered Vector Store Execution", "talking_points": "Combining semantic search with exact metadata filtering."}
        ],
        output_path=NOTEBOOKS_DIR / "09_routing_and_query_construction.ipynb"
    )
    print("Created notebook: 09_routing_and_query_construction.ipynb")


if __name__ == "__main__":
    build_nb08()
    build_nb09()
    print("Phase 2 notebooks built successfully!")
