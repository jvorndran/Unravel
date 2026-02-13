# Unravel

A visual sandbox for experimenting with RAG (Retrieval-Augmented Generation) configurations.

## Overview

Unravel helps developers understand and optimize their RAG pipelines through interactive visualizations. Experiment with document parsing, chunking strategies, embedding models, and retrieval configurations—all running locally on your machine.

## Demo

![Unravel Demo](docs/demo.gif)

## Installation

### From PyPI

```bash
# Using pip
pip install unravel

# Or using uv (faster)
uv pip install unravel
```

### From Source

```bash
git clone https://github.com/unravel/unravel.git
cd unravel

# Create virtual environment
uv venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install in editable mode
uv sync
```

### Development Dependencies

For contributing to the project, install development tools:

```bash
uv sync --all-extras
```

> **Note**: This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. Install uv: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows) or see [uv installation docs](https://github.com/astral-sh/uv#installation).

## Usage

Simply run the CLI command to launch the Streamlit app:

```bash
unravel
```

This opens the app in your browser at `http://localhost:8501`.

### API Key Setup

To use LLM features (query generation with OpenAI, Anthropic, etc.), configure your API keys:

1. Run `unravel` once to create the configuration directory
2. Navigate to `~/.unravel/` (or `%USERPROFILE%\.unravel\` on Windows)
3. Edit the `.env` file and add your API key:
   ```bash
   # For OpenAI
   OPENAI_API_KEY=sk-your-key-here

   # For Anthropic
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```
4. Save the file and refresh the app

**Note:** API keys are stored securely in `.env` and never saved to session files. For local models (Ollama, LM Studio), no API key is required.

### Running as a Python Module

```bash
uv run python -m unravel
```

## Features

Unravel guides you through a structured 5-step pipeline for building and testing RAG systems:

### Step 1: Document Upload 📄
Advanced multi-format document ingestion with file upload and URL scraping:
- 📁 **File Upload**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, and Images (PNG, JPG, BMP, TIFF)
- 🌐 **URL Scraping**: Extract content from web pages with JavaScript rendering support
  - **Crawl Modes**: Single page scraping or multi-page crawling
  - **Discovery Methods**: Crawler (follows internal links), Sitemap, or Feeds (RSS/Atom)
  - **Extraction Modes**: Balanced, Favor Precision, or Favor Recall
  - **Output Formats**: Markdown, TXT, CSV, JSON, HTML, XML, XML-TEI
  - **Metadata Extraction**: Author, publication date, description, tags/categories, site name
  - **Advanced Options**: robots.txt compliance, language filtering, content options (links, images, tables, formatting)
- 🔍 **Advanced Parsing**: OCR support for scanned documents and images
- 🏗️ **Structure Preservation**: Intelligent table structure extraction and hierarchical layout understanding
- 🎯 **Content Filtering**: Selective extraction by element type (headers, footers, code blocks, etc.)
- ⚙️ **Configurable Processing**: Thread settings and OCR options for optimal performance


### Step 2: Chunk Visualization ✂️
Flexible text splitting with transparent, visual chunk inspection:
- 🔀 **Two Chunking Strategies**:
  - **Hierarchical Chunker**: One chunk per document element, preserving semantic structure
  - **Hybrid Chunker**: Token-aware splitting with configurable limits and overlap for consistency
- 🎚️ **Token-Aware Configuration**: Set maximum chunk size (default: 512 tokens) and overlap percentage
- 🏷️ **Rich Metadata**: Attach configurable metadata to each chunk (section hierarchy, element type, token count, heading text, page numbers)
- 🎴 **Visual Chunk Cards**: Display full text with section breadcrumbs, metadata badges, and overlap highlighting
- 👁️ **Context Preview**: Expandable previews showing how chunks overlap with neighbors for seamless retrieval

### Step 3: Embedding Explorer 🧭
Visualize and analyze your document embeddings:
- 🤖 **10+ Embedding Models**: Choose from sentence-transformers models (all-MiniLM, all-mpnet, paraphrase variants), multilingual, QA-optimized, and BGE embeddings
- ⚡ **Fast Startup**: Lazy model loading ensures snappy UI responsiveness
- 🚀 **GPU Acceleration**: Automatic CUDA detection and acceleration when available
- 📊 **3D UMAP Visualization**: Interactive 3D scatter plot showing embedding space with cluster analysis
- 🎨 **Color Coding Options**: Visualize by KMeans clustering to identify semantic groupings
- 🔴 **Outlier Detection**: Identify and analyze outlier chunks in the embedding space
- 🔬 **Detailed Inspection**: Hover over points to preview chunks, click to view full details
- 🔎 **Semantic Search**: Test similarity search within embeddings with visual query point projection

### Step 4: Query Testing 🔍
End-to-end RAG testing with multiple retrieval strategies:
- 🎯 **Three Retrieval Methods**:
  - **Dense (Qdrant)**: Vector similarity search using embeddings for semantic matching
  - **Sparse (BM25)**: Keyword-based search for exact term matching
  - **Hybrid**: Combines dense and sparse with configurable fusion methods (weighted sum or reciprocal rank fusion)
- 🔄 **Query Expansion**: Generate multiple query variations with LLM to improve retrieval coverage (Reciprocal Rank Fusion for result merging)
- ⚙️ **Retrieval Configuration**: Adjust Top K results and minimum similarity score thresholds (strategy-specific defaults)
- 📈 **Reranking (Optional)**: Cross-encoder reranking to re-score and improve retrieval relevance
- 🤖 **LLM Answer Generation**: Integrate OpenAI, Anthropic, or local models (Ollama, LM Studio) to generate answers from retrieved chunks
- 🎛️ **Flexible Configuration**: Customize temperature, max tokens, system prompts, and API keys
- 📋 **Detailed Results**: View ranked chunks with similarity scores, source locations, and generated answers with full transparency

### Step 5: Code Export 💾
Generate production-ready Python code capturing your exact configuration:
- 📝 **Complete Implementation Code**: Exports working Python snippets for parsing, chunking, embedding, retrieval, and reranking
- 🎯 **Exact Configuration Preservation**: Every parameter and choice is captured in the generated code
- 📦 **Dependencies Management**: Complete requirements.txt with all necessary libraries and pinned versions
- ✂️ **Copy-Paste Ready**: Integration-ready code for immediate deployment to your application
- ✨ **Supports All Features**: Includes code for parsing options, chunking strategies, embedding models, retrieval methods, and LLM configuration

## Storage

All data is stored locally in `~/.unravel/`:

```
~/.unravel/
├── documents/          # Uploaded raw documents
├── chunks/             # Processed chunk data
├── embeddings/         # Cached embeddings
├── indices/            # Vector indices (Qdrant storage)
├── session_state.json  # UI state persistence
├── llm_config.json     # LLM configuration
└── rag_config.json     # RAG pipeline settings
```

No data is transmitted to external servers except when using LLM APIs for query generation.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/unravel/unravel.git
cd unravel

# Create virtual environment
uv venv

# Activate virtual environment (Windows)
.venv\Scripts\activate
# Or macOS/Linux: source .venv/bin/activate

# Install in development mode with all dependencies
uv sync --all-extras
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black unravel
uv run ruff check unravel
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

See [CLAUDE.md](CLAUDE.md) for coding standards and development philosophy.

## License

MIT License - see LICENSE file for details.

