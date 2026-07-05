# RAG Customer Service

An AI customer-service assistant for an apparel store, built with **Retrieval-Augmented Generation (RAG)**. It answers questions about size recommendations, washing & care, and color selection by retrieving from a local knowledge base and letting an LLM compose the reply. A Streamlit chat UI with streaming output and persistent conversation history is included, along with a separate UI for uploading documents into the knowledge base.

## Tech Stack

- **[LangChain](https://python.langchain.com/)** — orchestration (LCEL chains, message-history management)
- **[Chroma](https://www.trychroma.com/)** — local vector database
- **Alibaba Cloud DashScope** — `text-embedding-v4` (embeddings) + `qwen3-max` (chat, via Tongyi/Qwen)
- **[Streamlit](https://streamlit.io/)** — web UI

## Architecture

```
 app_file_uploader.py                app_qa.py  (chat UI, streaming)
   (upload TXT -> KB)                      |
          |                            rag.py  (RagService: LCEL chain)
   knowledge_base.py              ┌────────┴─────────────┐
   (split + dedupe + embed)  vector_stores.py     file_history_store.py
          |                   (retriever)          (file-backed chat history)
          v                        |
     ┌──────────┐  <-- retrieve ---┘
     │  Chroma  │
     └──────────┘
```

| File | Responsibility |
|------|----------------|
| `config_data.py` | Central configuration (models, chunking, retriever k, paths) |
| `knowledge_base.py` | Split text, dedupe by MD5, embed, and write to Chroma |
| `vector_stores.py` | Wrap the Chroma retriever |
| `rag.py` | Build the RAG chain with conversation history |
| `file_history_store.py` | Persist chat history to files, keyed by session id |
| `app_qa.py` | Streamlit chat front end |
| `app_file_uploader.py` | Streamlit UI to load TXT files into the knowledge base |
| `ingest_data.py` | One-off script to (re)build the vector store from `data/*.txt` |

## Prerequisites

- Python 3.10+
- An Alibaba Cloud DashScope API key

## Setup

```bash
pip install -r requirements.txt
```

Set your DashScope API key as an environment variable:

```bash
# macOS / Linux
export DASHSCOPE_API_KEY="your-api-key"

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY = "your-api-key"
```

## Build the knowledge base

Ingest the sample data in `./data` into the local Chroma store:

```bash
python ingest_data.py
```

## Run

Chat UI:

```bash
streamlit run app_qa.py
```

Knowledge base upload UI:

```bash
streamlit run app_file_uploader.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## Notes & Known Limitations

- **Session isolation**: `session_id` is currently hard-coded to `user_001`, so all users share one conversation history. Generate a unique id per user before deploying.
- **Deduplication**: dedupe is based on the MD5 of the whole document. Editing a single character makes it look like a brand-new document, and old chunks are not removed from Chroma.
- The `data/`, `chroma_db/`, `chat_history/`, and `md5.text` artifacts are generated at runtime and excluded from version control (except the sample `data/`).
