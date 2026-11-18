from pathlib import Path
import re
import tempfile

import ollama
import streamlit as st

from modules.pdf_extractor import process_documents
from modules.day2 import process_into_chunks
from modules.embedding_generator import generate_embeddings, model as embedding_model
from modules.vector_database import collection, store_embeddings
from modules.rag_pipeline import ask_question


# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="Intelligence Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
        .hero {
            padding: 1.35rem 1.5rem;
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 16px;
            margin-bottom: 1rem;
            background: linear-gradient(115deg, rgba(65,105,225,.10), rgba(30,144,255,.04));
        }
        .hero h1 {margin: 0 0 .35rem 0;}
        .muted {opacity: .72;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.22);
            padding: .8rem 1rem;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "notice" not in st.session_state:
    st.session_state.notice = None


# ----------------------------- Helpers -----------------------------
def get_document_names():
    """Return unique PDF names currently represented in ChromaDB."""
    records = collection.get(include=["metadatas"])
    names = set()
    for metadata in records.get("metadatas") or []:
        if metadata and metadata.get("document"):
            names.add(metadata["document"])
    return sorted(names, key=str.lower)


def get_document_chunks(document_name):
    """Fetch every stored chunk for one document, in original chunk order."""
    records = collection.get(
        where={"document": document_name},
        include=["documents", "metadatas"],
    )
    chunks = []
    ids = records.get("ids") or []
    docs = records.get("documents") or []
    metas = records.get("metadatas") or []

    for chunk_id, content, metadata in zip(ids, docs, metas):
        metadata = metadata or {}
        chunks.append({
            "id": chunk_id,
            "content": content or "",
            "metadata": metadata,
        })

    def chunk_number(item):
        match = re.search(r"_chunk_(\d+)$", item["id"])
        return int(match.group(1)) if match else 0

    return sorted(chunks, key=chunk_number)


def pages_for_chunks(chunks):
    pages = set()
    for chunk in chunks:
        raw_pages = chunk.get("metadata", {}).get("pages", "")
        for value in str(raw_pages).split(","):
            value = value.strip()
            if value.isdigit():
                pages.add(int(value))
    return sorted(pages)


def format_sources(document_name, pages):
    if not pages:
        return f"**Source:** {document_name}"
    page_text = ", ".join(str(page) for page in pages)
    return f"**Source:** `{document_name}` · **Page(s):** {page_text}"


def call_ollama(prompt):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def summarise_chunks(chunks, mode="Detailed"):
    """Map-reduce summary so a large PDF is not sent as one huge prompt."""
    if not chunks:
        return "No readable text was found in this document.", []

    # Each batch contains a manageable amount of source text.
    batch_size = 8
    partial_summaries = []
    all_pages = set()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batch_pages = pages_for_chunks(batch)
        all_pages.update(batch_pages)
        context = "\n\n".join(
            f"Pages {chunk['metadata'].get('pages', 'unknown')}:\n{chunk['content']}"
            for chunk in batch
        )
        prompt = f"""
You are summarising part of a PDF for a student.
Summarise only the supplied text. Do not add outside facts or invent details.
Preserve important names, dates, definitions, arguments, and cause-effect relationships.
Mention page references when useful.

SOURCE TEXT:
{context}

Write a clear intermediate summary in approximately 120-180 words.
"""
        partial_summaries.append(call_ollama(prompt))

    combined = "\n\n".join(
        f"Part {index + 1}: {summary}"
        for index, summary in enumerate(partial_summaries)
    )

    if mode == "Short":
        instruction = "Create a concise summary in 5-8 bullet points, focusing only on the most important ideas."
    else:
        instruction = (
            "Create a well-structured detailed summary with headings and bullet points. "
            "Cover the main ideas, important supporting details, definitions, and conclusions."
        )

    final_prompt = f"""
Combine the intermediate summaries below into one coherent summary of the PDF.
Use only the information in these summaries. Do not invent facts.
{instruction}

INTERMEDIATE SUMMARIES:
{combined}
"""
    return call_ollama(final_prompt), sorted(all_pages)


def summarise_section(document_name, section_query, mode="Detailed"):
    """Retrieve the most relevant chunks from one selected PDF and summarise them."""
    document_chunk_count = len(get_document_chunks(document_name))
    if document_chunk_count == 0:
        return "No chunks are stored for this document.", []

    query_vector = embedding_model.encode(section_query).tolist()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(8, document_chunk_count),
        where={"document": document_name},
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    selected_chunks = [
        {"id": chunk_id, "content": content or "", "metadata": metadata or {}}
        for chunk_id, content, metadata in zip(ids, docs, metas)
    ]

    if not selected_chunks:
        return "I couldn't find relevant text for that section in the selected PDF.", []

    context = "\n\n".join(
        f"Pages {chunk['metadata'].get('pages', 'unknown')}:\n{chunk['content']}"
        for chunk in selected_chunks
    )
    detail_instruction = (
        "Give a concise explanation in 4-6 bullet points."
        if mode == "Short"
        else "Explain the section clearly with a short heading and useful bullet points, preserving important details."
    )
    prompt = f"""
Summarise the requested section using only the supplied document extracts.
Do not use outside knowledge or invent information.
Requested section/topic: {section_query}
{detail_instruction}

DOCUMENT EXTRACTS:
{context}
"""
    return call_ollama(prompt), pages_for_chunks(selected_chunks)


def ingest_uploaded_pdf(uploaded_file):
    """Add one new PDF to the existing collection without duplicating its filename."""
    existing_names = get_document_names()
    if uploaded_file.name in existing_names:
        return False, (
            f"`{uploaded_file.name}` is already in the document collection. "
            "Remove the existing document first if you want to replace it."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / Path(uploaded_file.name).name
        pdf_path.write_bytes(uploaded_file.getvalue())

        pages = process_documents([str(pdf_path)])
        if not pages:
            return False, "No readable text could be extracted. The PDF may be scanned or image-only."

        chunks = process_into_chunks(pages, chunk_size=100, overlap=20)
        if not chunks:
            return False, "The PDF did not produce any text chunks to store."

        embedded_chunks = generate_embeddings(chunks)
        if not embedded_chunks:
            return False, "Embedding generation returned no chunks."

        store_embeddings(embedded_chunks)
        return True, f"Added **{uploaded_file.name}** successfully: {len(pages)} pages and {len(chunks)} chunks."


def delete_document(document_name):
    records = collection.get(
        where={"document": document_name},
        include=["metadatas"],
    )
    ids = records.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def ollama_status():
    try:
        result = ollama.list()
        models = result.get("models", [])
        names = [item.get("name") or item.get("model") or "" for item in models]
        available = any(name == "qwen2.5:3b" or name.startswith("qwen2.5:3b-") for name in names)
        return True, available
    except Exception:
        return False, False


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.title("📚 Document Assistant")
    st.caption("Your local, private document workspace")
    st.divider()

    st.subheader("System status")
    try:
        chunk_count = collection.count()
        st.success("ChromaDB connected")
        st.metric("Stored chunks", chunk_count)
    except Exception as error:
        st.error(f"ChromaDB unavailable: {error}")
        chunk_count = 0

    ollama_connected, model_available = ollama_status()
    if ollama_connected and model_available:
        st.success("Ollama connected · Qwen 2.5 3B ready")
    elif ollama_connected:
        st.warning("Ollama is running, but qwen2.5:3b is not listed")
    else:
        st.error("Ollama is not responding. Start Ollama and try again.")

    st.divider()
    st.subheader("Quick actions")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Refresh document list", use_container_width=True):
        st.rerun()


# ----------------------------- Main header -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Intelligence Document Assistant</h1>
        <div class="muted">Ask questions, explore sections, and summarise your PDFs — powered by your local RAG pipeline.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.notice:
    st.info(st.session_state.notice)
    st.session_state.notice = None

# Current collection overview
try:
    document_names = get_document_names()
    current_count = collection.count()
except Exception as error:
    st.error(f"Could not read the ChromaDB collection: {error}")
    document_names = []
    current_count = 0

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Documents", len(document_names))
metric2.metric("Stored chunks", current_count)
metric3.metric("LLM", "Qwen 2.5 · 3B")

# ----------------------------- Tabs -----------------------------
tab_chat, tab_documents, tab_summary = st.tabs(
    ["💬 Ask documents", "🗂️ Document manager", "📝 Summarise"]
)


# ----------------------------- Chat tab -----------------------------
with tab_chat:
    st.subheader("Chat with your documents")
    st.caption("Answers are grounded in the PDFs already stored in your ChromaDB collection.")

    if not document_names:
        st.info("Your collection is empty. Upload a PDF in the Document manager tab to get started.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDFs...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents and generating an answer..."):
                try:
                    answer = ask_question(question)
                except Exception as error:
                    answer = f"I couldn't complete the request. Check that ChromaDB and Ollama are running.\n\nDetails: `{error}`"
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ----------------------------- Document manager tab -----------------------------
with tab_documents:
    st.subheader("Manage your PDF collection")
    st.write("Upload a PDF to extract, chunk, embed, and store it in your existing ChromaDB collection.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Existing filenames are skipped to prevent duplicate ingestion.",
    )

    if st.button("➕ Add selected PDFs", type="primary", disabled=not uploaded_files):
        for uploaded_file in uploaded_files or []:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    success, message = ingest_uploaded_pdf(uploaded_file)
                    if success:
                        st.success(message)
                    else:
                        st.warning(message)
                except Exception as error:
                    st.error(f"Could not add {uploaded_file.name}: {error}")
        st.rerun()

    st.divider()
    st.markdown("**Documents currently in ChromaDB**")
    if document_names:
        for document_name in document_names:
            with st.container(border=True):
                left, right = st.columns([5, 1])
                with left:
                    document_chunks = get_document_chunks(document_name)
                    page_set = pages_for_chunks(document_chunks)
                    st.markdown(f"📄 **{document_name}**")
                    st.caption(f"{len(document_chunks)} chunks · {len(page_set)} pages represented")
                with right:
                    if st.button("Remove", key=f"remove_{document_name}"):
                        removed = delete_document(document_name)
                        st.session_state.notice = f"Removed {removed} chunks belonging to {document_name}."
                        st.rerun()
    else:
        st.info("No documents are currently stored.")


# ----------------------------- Summarisation tab -----------------------------
with tab_summary:
    st.subheader("Summarise a document")
    st.caption("Full-document summaries use all stored chunks. Section summaries retrieve relevant chunks from the selected PDF.")

    if not document_names:
        st.info("Upload a PDF first to enable summarisation.")
    else:
        selected_document = st.selectbox("Choose a document", document_names)
        summary_mode = st.radio("Summary length", ["Short", "Detailed"], horizontal=True)
        summary_type = st.radio(
            "What would you like to summarise?",
            ["Full document", "A section or topic"],
            horizontal=True,
        )

        section_query = ""
        if summary_type == "A section or topic":
            section_query = st.text_input(
                "Enter the section heading or topic",
                placeholder="For example: causes of the Revolt of 1857",
            )

        run_summary = st.button(
            "✨ Generate summary",
            type="primary",
            disabled=(summary_type == "A section or topic" and not section_query.strip()),
        )

        if run_summary:
            try:
                with st.spinner("Reading the document and preparing your summary..."):
                    if summary_type == "Full document":
                        all_chunks = get_document_chunks(selected_document)
                        summary, source_pages = summarise_chunks(all_chunks, summary_mode)
                    else:
                        summary, source_pages = summarise_section(
                            selected_document, section_query.strip(), summary_mode
                        )

                st.markdown("### Summary")
                st.markdown(summary)
                st.divider()
                st.markdown(format_sources(selected_document, source_pages))
            except Exception as error:
                st.error(
                    "Summary generation failed. Check that Ollama is running and the selected document is available. "
                    f"Details: `{error}`"
                )

st.caption("Runs locally on your machine. Your PDFs and embeddings remain in your local ChromaDB folder.")
