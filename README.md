# 🤖 AI Projects

A collection of practical Artificial Intelligence and Generative AI projects focused on building real-world applications using **Python, LLMs, RAG, LangChain, vector databases, and AI APIs**.

This repository documents my hands-on work in **Generative AI, Retrieval-Augmented Generation (RAG), LLM applications, and AI engineering**.

---

## 📌 Projects

### 1. 📚 RAG-Based Document Q&A Assistant

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions based on their content.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant document sections and generate grounded responses using the Mistral API.

#### 🔑 Key Features

- 📄 **PDF Document Ingestion** — Upload and process PDF documents.
- ✂️ **Intelligent Text Chunking** — Splits documents into optimized chunks for retrieval.
- 🧠 **Semantic Embeddings** — Generates local embeddings for meaningful document search.
- 🗄️ **ChromaDB Vector Store** — Stores and retrieves document embeddings efficiently.
- 🔎 **Top-K Semantic Retrieval** — Retrieves relevant document chunks for each query.
- 🤖 **Mistral-Powered Answers** — Generates responses using retrieved document context.
- 📚 **Source Citations** — Displays document and page sources used for answers.
- ⚙️ **Configurable RAG Pipeline** — Tune chunk size, overlap, and Top-K retrieval.

#### 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| LangChain | RAG pipeline and LLM integration |
| PyPDFLoader | PDF document processing |
| RecursiveCharacterTextSplitter | Text chunking |
| ChromaDB | Vector database |
| MiniLM ONNX Embeddings | Local semantic embeddings |
| Mistral API | LLM response generation |
| Streamlit | Web application interface |

#### 🔄 Architecture

```text
PDF Documents
      ↓
PyPDFLoader
      ↓
Text Chunking
      ↓
MiniLM Embeddings
      ↓
ChromaDB
      ↓
Semantic Retrieval
      ↓
Top-K Relevant Chunks
      ↓
Mistral LLM
      ↓
Grounded Answer + Sources
