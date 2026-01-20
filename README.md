# RAG Visualizer

A visual sandbox for experimenting with RAG (Retrieval-Augmented Generation) configurations.

## Overview

RAG Visualizer helps developers understand and optimize their RAG pipelines through interactive visualizations. Experiment with document parsing, chunking strategies, embedding models, and retrieval configurations—all running locally on your machine.

## Requirements

- Python 3.9, 3.10, 3.11, or 3.12
- 4GB+ RAM recommended (for embedding models)
- Optional: CUDA-compatible GPU for acceleration

## Installation

### From PyPI

```bash
pip install rag-visualizer
```

### From Source

```bash
git clone https://github.com/rag-visualizer/rag-visualizer.git
cd rag-visualizer
pip install -e .
```

### Optional Dependencies

Install LLM support for RAG query generation:

```bash
pip install rag-visualizer[llm]
```

Install NLP enhancements:

```bash
pip install rag-visualizer[nlp]
```

Install all optional dependencies:

```bash
pip install rag-visualizer[all]
```

### GPU Acceleration

For faster document parsing and embedding generation with CUDA-enabled GPUs, install PyTorch with CUDA support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions.

## Usage

Simply run the CLI command to launch the Streamlit app:

```bash
rag-visualizer
```

This opens the app in your browser at `http://localhost:8501`.

### CLI Options

```bash
rag-visualizer --help

Options:
  -p, --port INTEGER  Port to run the Streamlit app on (default: 8501)
  -h, --host TEXT     Host to bind the Streamlit app to (default: localhost)
  --version           Show the version and exit.
  --help              Show this message and exit.
```

### Running as a Python Module

```bash
python -m rag_visualizer
```

## Features

### Document Upload
- Multi-format support: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, Images
- Powered by Docling for advanced document parsing
- OCR support for scanned documents and images
- Table structure extraction and preservation
- Content filtering by document element type
- Configurable output formats (Markdown, HTML, DocTags, JSON)

![Document Upload](docs/images/upload.png)

### Chunk Visualization
- Multiple chunking strategies (Hierarchical, Hybrid)
- Token-aware chunking with configurable limits and overlap
- Visual chunk cards with metadata badges
- Section hierarchy breadcrumbs for context
- Overlap highlighting
- Expandable context previews
- Configurable metadata fields

![Chunk Visualization](docs/images/chunks.png)

### Embedding Explorer
- Multiple embedding models via sentence-transformers
- 2D UMAP visualization of embedding space
- Interactive Plotly scatter plots
- Outlier detection and analysis
- Color coding by document or cluster
- Click-to-inspect chunk details

![Embedding Explorer](docs/images/embeddings.png)

### Query Testing
- Semantic search with adjustable top-K results
- Cosine similarity scoring
- Retrieved chunk highlighting
- Query visualization on embedding plot
- LLM integration for RAG answer generation
- Support for OpenAI, Anthropic, and OpenAI-compatible local models

![Query Testing](docs/images/query.png)

### Code Export
- Production-ready Python code generation
- Exact configuration preservation
- Complete requirements.txt with dependencies
- Ready-to-run scripts for deployment

## Storage

All data is stored locally in `~/.rag-visualizer/`:

```
~/.rag-visualizer/
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
git clone https://github.com/rag-visualizer/rag-visualizer.git
cd rag-visualizer

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black rag_visualizer
ruff check rag_visualizer
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI Framework | Streamlit | Web application interface |
| UI Components | streamlit-shadcn-ui | Modern UI components |
| Visualization | Plotly | Interactive charts |
| Document Parsing | Docling | Universal document conversion |
| Embeddings | sentence-transformers | Text embeddings |
| Vector Search | FAISS | Similarity search and indexing |
| Dimensionality Reduction | UMAP | 2D embedding visualization |
| Tokenization | tiktoken | Token counting |
| CLI | Click | Command-line interface |
| LLM Integration | OpenAI, Anthropic SDKs | RAG query generation (optional) |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

See [CLAUDE.md](CLAUDE.md) for coding standards and development philosophy.

## License

MIT License - see LICENSE file for details.

