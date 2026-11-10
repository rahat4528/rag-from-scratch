"""
Interactive Companion App for RAG From Scratch Course
Powered by Gradio and LangChain.

Usage:
    python course/app/app.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from course.utils.models import get_llm, get_embeddings, detect_provider
from course.utils.helpers import get_data_dir, cosine_sim
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
import numpy as np
import gradio as gr

# 1. Initialize Corpus and Index
data_dir = get_data_dir()
embeddings = get_embeddings()

raw_docs = []
for f in sorted(data_dir.iterdir()):
    if f.suffix.lower() in [".txt", ".md"]:
        loaded = TextLoader(str(f), encoding="utf-8").load()
        for d in loaded:
            d.metadata["source"] = f.name
        raw_docs.extend(loaded)
    elif f.suffix.lower() == ".pdf":
        loaded = PyPDFLoader(str(f)).load()
        for d in loaded:
            d.metadata["source"] = f.name
        raw_docs.extend(loaded)

splitter = RecursiveCharacterTextSplitter(chunk_size=380, chunk_overlap=40)
chunks = splitter.split_documents(raw_docs)

# BM25 and Dense Store
corpus_texts = [c.page_content for c in chunks]
bm25_index = BM25Okapi([t.lower().split() for t in corpus_texts])
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)


# 2. Query Pipeline Function
def query_rag_pipeline(user_query: str, top_k: int = 3, use_reranker: bool = True, temperature: float = 0.0):
    if not user_query.strip():
        return "Please enter a valid question.", "No sources", "No chunks retrieved."

    t0 = time.time()
    q_tokens = user_query.lower().split()
    
    # Stage 1: Retrieval
    dense_matches = vector_store.similarity_search(user_query, k=top_k * 2)
    bm25_scores = bm25_index.get_scores(q_tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
    bm25_matches = [chunks[i] for i in top_bm25_idx]
    
    # Deduplicate candidates
    candidates = {d.page_content: d for d in (dense_matches + bm25_matches)}.values()
    
    # Stage 2: Reranking
    if use_reranker:
        q_tokens_set = set(q_tokens)
        q_vec = np.array(embeddings.embed_query(user_query))
        scored = []
        for doc in candidates:
            d_tokens_set = set(doc.page_content.lower().split())
            d_vec = np.array(embeddings.embed_query(doc.page_content))
            overlap = len(q_tokens_set.intersection(d_tokens_set)) / max(len(q_tokens_set), 1)
            sem = cosine_sim(q_vec, d_vec)
            joint_score = 0.50 * sem + 0.50 * overlap
            scored.append((doc, joint_score))
        reranked = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        selected_docs = [doc for doc, _ in reranked]
    else:
        selected_docs = list(candidates)[:top_k]
        
    t_retrieval = time.time() - t0

    # Stage 3: Prompt Formatting & Generation
    llm = get_llm(temperature=temperature)
    formatted_ctx = "\n\n".join([f"[{d.metadata.get('source', 'Unknown')}] {d.page_content.strip()}" for d in selected_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert technical advisor for Novasphere Dynamics.\n"
            "Answer the user's question using ONLY the provided context snippets.\n"
            "If the answer cannot be determined from the context, state: 'I don't know based on the provided context.'\n"
            "Always cite the source document name(s) in brackets (e.g. [company_handbook.txt]).\n\n"
            "CONTEXT:\n{context}"
        )),
        ("human", "{question}")
    ])
    
    response = llm.invoke(prompt.format_messages(context=formatted_ctx, question=user_query))
    ans_text = response.content if hasattr(response, "content") else str(response)
    total_time = time.time() - t0

    # Format output sources and inspected chunks
    sources = list(set([d.metadata.get("source") for d in selected_docs if d.metadata.get("source")]))
    sources_text = ", ".join(sources) if sources else "None cited"
    
    chunks_display = f"### Pipeline Latency: {total_time*1000:.1f} ms (Retrieval: {t_retrieval*1000:.1f} ms)\n\n"
    for i, d in enumerate(selected_docs, 1):
        chunks_display += f"#### Chunk #{i} — [{d.metadata.get('source')}]\n```text\n{d.page_content.strip()}\n```\n\n"

    return ans_text.strip(), sources_text, chunks_display


# 3. Build Gradio UI
provider_name = detect_provider()

with gr.Blocks(title="Novasphere RAG Assistant") as demo:
    gr.Markdown(
        f"""# 🤖 Novasphere Dynamics: Grounded RAG Assistant
        **Interactive Companion App for the 'RAG From Scratch' Course**  
        *Active Provider:* `{provider_name.upper()}` | *Corpus:* `Novasphere Knowledge Base (4 files, 39 chunks)`
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Ask a Question",
                placeholder="e.g. How much does the Enterprise Plan cost per month?",
                lines=2
            )
            submit_btn = gr.Button("Ask Assistant", variant="primary")
            
            gr.Markdown("#### Quick Sample Questions")
            with gr.Row():
                btn_pricing = gr.Button("💰 Enterprise Pricing", size="sm")
                btn_trouble = gr.Button("🔧 ERR_QUANTUM_TIMEOUT", size="sm")
                btn_pets = gr.Button("🐕 Zurich Pet Policy", size="sm")
                btn_unans = gr.Button("❓ Unanswerable Query", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("#### Pipeline Controls")
            k_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top-K Chunks")
            rerank_toggle = gr.Checkbox(value=True, label="Enable Cross-Encoder Reranker")
            temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="LLM Temperature")

    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="Grounded Answer", lines=5)
            sources_output = gr.Textbox(label="Cited Sources", lines=1)
        with gr.Column(scale=2):
            with gr.Accordion("🔍 Inspect Retrieved Context Chunks & Latency", open=True):
                chunks_output = gr.Markdown("Submit a query to inspect retrieved context chunks.")

    # Wiring actions
    submit_btn.click(
        fn=query_rag_pipeline,
        inputs=[query_input, k_slider, rerank_toggle, temp_slider],
        outputs=[answer_output, sources_output, chunks_output]
    )
    
    # Preset buttons
    btn_pricing.click(lambda: "How much does the Enterprise Plan cost per month?", outputs=query_input)
    btn_trouble.click(lambda: "How do I resolve ERR_QUANTUM_TIMEOUT?", outputs=query_input)
    btn_pets.click(lambda: "What is the pet policy in the Zurich office?", outputs=query_input)
    btn_unans.click(lambda: "What is Novasphere Dynamics' stock ticker on NASDAQ?", outputs=query_input)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
