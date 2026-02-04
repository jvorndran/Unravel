# RAG Lens

A visual sandbox for experimenting with RAG (Retrieval-Augmented Generation) configurations.

## Overview

RAG Lens helps developers understand and optimize their RAG pipelines through interactive visualizations. Experiment with document parsing, chunking strategies, embedding models, and retrieval configurations—all running locally on your machine.

## Installation

### From PyPI

```bash
pip install rag-lens
```

### From Source

```bash
git clone https://github.com/rag-lens/rag-lens.git
cd rag-lens
pip install -e .
```

### Development Dependencies

For contributing to the project, install development tools:

```bash
pip install rag-lens[dev]
```

### GPU Acceleration

For faster document parsing and embedding generation with CUDA-enabled GPUs, install PyTorch with CUDA support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions.

## Usage

Simply run the CLI command to launch the Streamlit app:

```bash
rag-lens
```

This opens the app in your browser at `http://localhost:8501`.

### API Key Setup

To use LLM features (query generation with OpenAI, Anthropic, etc.), configure your API keys:

1. Run `rag-lens` once to create the configuration directory
2. Navigate to `~/.rag-lens/` (or `%USERPROFILE%\.rag-lens\` on Windows)
3. Edit the `.env` file and add your API key:
   ```bash
   # For OpenAI
   OPENAI_API_KEY=sk-your-key-here

   # For Anthropic
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```
4. Save the file and refresh the app

**Note:** API keys are stored securely in `.env` and never saved to session files. For local models (Ollama, LM Studio), no API key is required.

### CLI Options

```bash
rag-lens --help

Options:
  -p, --port INTEGER  Port to run the Streamlit app on (default: 8501)
  -h, --host TEXT     Host to bind the Streamlit app to (default: localhost)
  --version           Show the version and exit.
  --help              Show this message and exit.
```

### Running as a Python Module

```bash
python -m rag_lens
```

## Features

RAG Lens guides you through a structured 5-step pipeline for building and testing RAG systems:

### Step 1: Document Upload
Advanced multi-format document ingestion powered by Docling:
- **Supported Formats**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, and Images (PNG, JPG, BMP, TIFF)
- **Advanced Parsing**: OCR support for scanned documents and images
- **Structure Preservation**: Intelligent table structure extraction and hierarchical layout understanding
- **Content Filtering**: Selective extraction by element type (headers, footers, code blocks, etc.)
- **Output Formats**: Generate parsed content as Markdown, HTML, DocTags, or JSON
- **Configurable Processing**: Thread settings and OCR options for optimal performance


### Step 2: Chunk Visualization
Flexible text splitting with transparent, visual chunk inspection:
- **Two Chunking Strategies**:
  - **Hierarchical Chunker**: One chunk per document element, preserving semantic structure
  - **Hybrid Chunker**: Token-aware splitting with configurable limits and overlap for consistency
- **Token-Aware Configuration**: Set maximum chunk size (default: 512 tokens) and overlap percentage
- **Rich Metadata**: Attach configurable metadata to each chunk (section hierarchy, element type, token count, heading text, page numbers)
- **Visual Chunk Cards**: Display full text with section breadcrumbs, metadata badges, and overlap highlighting
- **Context Preview**: Expandable previews showing how chunks overlap with neighbors for seamless retrieval

### Step 3: Embedding Explorer
Visualize and analyze your document embeddings:
- **10+ Embedding Models**: Choose from sentence-transformers models (all-MiniLM, all-mpnet, paraphrase variants), multilingual, QA-optimized, and BGE embeddings
- **Fast Startup**: Lazy model loading ensures snappy UI responsiveness
- **GPU Acceleration**: Automatic CUDA detection and acceleration when available
- **2D UMAP Visualization**: Interactive scatter plot showing embedding space with cluster analysis
- **Color Coding Options**: Visualize by document or KMeans clustering to identify semantic groupings
- **Outlier Detection**: Identify and analyze outlier chunks that may have unusual embeddings
- **Detailed Inspection**: Hover over points to preview chunks, click to view full details

### Step 4: Query Testing
End-to-end RAG testing with multiple retrieval strategies:
- **Three Retrieval Methods**:
  - **Dense (FAISS)**: Vector similarity search using embeddings for semantic matching
  - **Sparse (BM25)**: Keyword-based search for exact term matching
  - **Hybrid**: Combines dense and sparse with configurable fusion methods (weighted sum or reciprocal rank fusion)
- **Reranking (Optional)**: Cross-encoder reranking to re-score and improve retrieval relevance
- **LLM Answer Generation**: Integrate OpenAI, Anthropic, or local models (Ollama, LM Studio) to generate answers from retrieved chunks
- **Flexible Configuration**: Customize temperature, max tokens, system prompts, and API keys
- **Detailed Results**: View ranked chunks with similarity scores, source locations, and generated answers with full transparency

### Step 5: Code Export
Generate production-ready Python code capturing your exact configuration:
- **Complete Implementation Code**: Exports working Python snippets for parsing, chunking, embedding, retrieval, and reranking
- **Exact Configuration Preservation**: Every parameter and choice is captured in the generated code
- **Dependencies Management**: Complete requirements.txt with all necessary libraries and pinned versions
- **Copy-Paste Ready**: Integration-ready code for immediate deployment to your application
- **Supports All Features**: Includes code for parsing options, chunking strategies, embedding models, retrieval methods, and LLM configuration

## Storage

All data is stored locally in `~/.rag-lens/`:

```
~/.rag-lens/
├── documents/          # Uploaded raw documents
├── chunks/             # Processed chunk data
├── embeddings/         # Cached embeddings
├── indices/            # FAISS vector indices
├── session_state.json  # UI state persistence
├── llm_config.json     # LLM configuration
└── rag_config.json     # RAG pipeline settings
```

No data is transmitted to external servers except when using LLM APIs for query generation.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/rag-lens/rag-lens.git
cd rag-lens

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black rag_lens
ruff check rag_lens
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

See [CLAUDE.md](CLAUDE.md) for coding standards and development philosophy.

## License

MIT License - see LICENSE file for details.

