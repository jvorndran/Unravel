# Unravel - Product Requirements Document

## 1. Executive Summary

Unravel is an interactive visual sandbox for experimenting with Retrieval-Augmented Generation (RAG) configurations. The application provides developers and AI practitioners with a local tool to understand how documents are processed, chunked, embedded, and retrieved in RAG pipelines.

When building RAG systems, developers struggle to understand how their documents are being chunked, how embeddings cluster in vector space, and which chunks get retrieved for different queries. Unravel addresses these pain points by providing interactive visualizations and configuration options that make the RAG pipeline transparent and experimentable.

**MVP Goal:** Deliver a fully functional local application that allows users to upload documents, visualize chunking strategies, explore embedding spaces, test retrieval queries, and export production-ready code.

---

## 2. Mission

### Mission Statement

Empower developers to build better RAG systems by making every step of the pipeline visible, configurable, and understandable.

### Core Principles

1. **Transparency** - Every step of the RAG pipeline should be visible and inspectable
2. **Experimentation** - Users should be able to quickly iterate on configurations without code changes
3. **Local-First** - All processing happens locally; no data leaves the user's machine
4. **Production-Ready** - Configurations can be exported as working code for production use
5. **Simplicity** - Complex concepts presented through clean, intuitive interfaces

---

## 3. Target Users

### Primary Persona: AI/ML Developer

- **Role:** Software engineer building RAG-based applications
- **Technical Level:** Intermediate to advanced Python developer
- **Experience:** Familiar with embeddings, vector databases, and LLMs
- **Environment:** Local development machine with Python 3.9+

**Key Needs:**
- Understand how different chunking strategies affect retrieval quality
- Visualize embedding distributions to identify clustering patterns
- Test queries before deploying to production
- Export working code to integrate into existing projects

**Pain Points:**
- Black-box nature of RAG pipelines makes debugging difficult
- Trial-and-error approach to finding optimal chunk sizes
- No visibility into why certain chunks are retrieved over others
- Time-consuming to experiment with different configurations

### Secondary Persona: Data Scientist / Researcher

- **Role:** Researcher exploring document understanding and retrieval
- **Technical Level:** Strong Python skills, ML background
- **Use Case:** Evaluating RAG approaches for research projects

**Key Needs:**
- Compare different embedding models on the same corpus
- Analyze chunk distributions and overlap patterns
- Export results for further analysis

---

## 4. MVP Scope

### In Scope

**Core Functionality**
- ✅ Document upload and parsing (PDF, DOCX, PPTX, XLSX, HTML, Markdown, Text, Images)
- ✅ Multiple chunking strategies (Hierarchical, Hybrid)
- ✅ Configurable chunking parameters (max tokens, overlap, tokenizer)
- ✅ Multiple embedding models (sentence-transformers)
- ✅ 2D UMAP visualization of embedding space
- ✅ Query testing with multiple retrieval strategies (Dense, Sparse, Hybrid)
- ✅ Optional cross-encoder reranking for improved results
- ✅ Code export for production deployment

**Document Processing**
- ✅ Docling-based parsing with OCR support
- ✅ Table structure extraction
- ✅ Content filtering by document element type
- ✅ Image extraction from PDFs
- ✅ Multiple output formats (Markdown, HTML, DocTags, JSON)

**User Interface**
- ✅ Step-by-step pipeline navigation (Upload, Chunks, Embeddings, Query, Export)
- ✅ Sidebar configuration panel with RAG and LLM settings
- ✅ Interactive chunk visualization with expandable context preview
- ✅ Embedding scatter plot with hover details

**LLM Integration**
- ✅ OpenAI API support
- ✅ Anthropic API support
- ✅ OpenAI-compatible local models (Ollama, LM Studio)
- ✅ Configurable temperature, max tokens, system prompts

**Persistence**
- ✅ Local storage for documents, chunks, embeddings, and indices
- ✅ Session state persistence across browser refreshes
- ✅ Configuration save/restore

### Out of Scope

**Deferred Features**
- ❌ Multi-document comparison side-by-side
- ❌ Batch processing of document folders
- ❌ Cloud storage integration (S3, GCS, Azure Blob)
- ❌ User authentication and multi-tenancy
- ❌ API endpoint exposure for programmatic access
- ❌ Evaluation metrics (precision, recall, MRR)
- ❌ A/B testing framework for configurations
- ❌ Integration with production vector databases (Pinecone, Weaviate, Qdrant)
- ❌ Collaborative features (sharing, commenting)
- ❌ Custom embedding model fine-tuning

---

## 5. User Stories

### Document Upload & Parsing

**US-1:** As a developer, I want to upload documents in various formats, so that I can process any document type my production system will handle.

*Example:* User drags a PDF research paper into the upload area. The system parses it using Docling, extracts text with table structures preserved, and displays a preview.

**US-2:** As a developer, I want to configure parsing options like OCR and content filtering, so that I can optimize for my specific document types.

*Example:* User enables OCR for scanned PDFs and filters out page headers/footers to reduce noise in the extracted text.

### Chunking & Visualization

**US-3:** As a developer, I want to see how my documents are split into chunks, so that I can understand the boundaries and ensure semantic coherence.

*Example:* User views chunks displayed inline with colored backgrounds, metadata badges, and section breadcrumbs. Each chunk can be expanded to reveal the extra context that will be prepended during embedding.

**US-4:** As a developer, I want to compare different chunking strategies, so that I can find the optimal approach for my use case.

*Example:* User switches between Hierarchical and Hybrid chunking strategies, observing how chunk boundaries change and how many chunks are produced.

### Embeddings & Visualization

**US-5:** As a developer, I want to visualize my document embeddings in 2D space, so that I can identify clusters and understand semantic relationships.

*Example:* User generates embeddings using all-MiniLM-L6-v2 and views a UMAP scatter plot. Related chunks cluster together, revealing document structure.

**US-6:** As a developer, I want to try different embedding models, so that I can compare quality and performance trade-offs.

*Example:* User switches from MiniLM (fast, 384 dims) to MPNet (higher quality, 768 dims) and regenerates embeddings to compare clustering.

### Query & Retrieval

**US-7:** As a developer, I want to test retrieval queries against my indexed documents, so that I can validate that relevant chunks are being returned.

*Example:* User enters "What are the key findings?" and sees the top-5 most similar chunks with their similarity scores and source locations.

**US-8:** As a developer, I want to use an LLM to generate answers from retrieved chunks, so that I can test end-to-end RAG quality.

*Example:* User configures GPT-4 API key, asks a question, and receives a synthesized answer with citations to specific chunks.

**US-8.1:** As a developer, I want to experiment with different retrieval strategies (dense, sparse, hybrid), so that I can optimize retrieval quality for my use case.

*Example:* User switches between Dense (Qdrant), Sparse (BM25), and Hybrid retrieval, comparing which strategy returns the most relevant chunks for technical documentation vs. general narrative text.

**US-8.2:** As a developer, I want to enable reranking to improve my retrieval results, so that I can maximize answer quality.

*Example:* User enables FlashRank reranking and observes improved relevance in the top-5 results, with less relevant chunks filtered out.

### Export & Integration

**US-9:** As a developer, I want to export my configuration as working Python code, so that I can integrate it into my production application.

*Example:* User clicks "Export" and downloads a Python script with their exact parsing, chunking, embedding, and retrieval configuration ready to run.

---

## 6. Core Architecture & Patterns

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI Layer                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │ Upload  │ │ Chunks  │ │Embedding│ │  Query  │ │ Export │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └───┬────┘ │
└───────┼──────────┼──────────┼──────────┼───────────┼────────┘
        │          │          │          │           │
┌───────▼──────────▼──────────▼──────────▼───────────▼────────┐
│                      Services Layer                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │ Parsers  │ │ Chunking │ │Embedders │ │ Vector Store     ││
│  │ (Docling)│ │(Docling) │ │(ST)      │ │ (Qdrant)         ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │ Retrieval│ │   LLM    │ │  Export  │ │     Storage      ││
│  │ (Dense/  │ │ Service  │ │Generator │ │(~/.unravel││
│  │  Sparse/ │ │          │ │          │ │                  ││
│  │  Hybrid) │ │          │ │          │ │                  ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
unravel/
├── __init__.py
├── __main__.py           # Module entry point
├── app.py                # Main Streamlit application
├── cli.py                # Click CLI interface
├── services/
│   ├── __init__.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── core.py       # Chunking orchestration
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py   # Provider interface
│   │       └── docling_provider.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── core.py       # Retrieval provider registry
│   │   ├── reranking.py  # FlashRank reranking
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py   # Provider interface
│   │       ├── dense.py  # Qdrant retriever
│   │       ├── sparse.py # BM25 retriever
│   │       └── hybrid.py # Combined retriever
│   ├── embedders.py      # Embedding generation
│   ├── export/
│   │   ├── __init__.py
│   │   ├── generator.py  # Code generation
│   │   └── templates.py  # Code templates
│   ├── llm.py            # LLM integration
│   ├── storage.py        # Local persistence
│   └── vector_store.py   # Qdrant operations
├── ui/
│   ├── sidebar.py        # Configuration sidebar
│   └── steps/
│       ├── __init__.py
│       ├── upload.py     # Document upload step
│       ├── chunks.py     # Chunk visualization step
│       ├── embeddings.py # Embedding visualization step
│       ├── query.py      # Query testing step
│       └── export.py     # Code export step
└── utils/
    ├── __init__.py
    ├── parsers.py        # Document parsing
    ├── ui.py             # UI utilities
    └── visualization.py  # Plotting utilities
```

### Key Design Patterns

**Provider Pattern (Chunking)**
```python
class ChunkingProvider(Protocol):
    name: str
    def get_available_splitters(self) -> list[SplitterInfo]: ...
    def chunk(self, splitter: str, text: str, **params) -> list[Chunk]: ...
```

**Provider Pattern (Retrieval)**
```python
class RetrieverProvider(Protocol):
    name: str
    def get_available_retrievers(self) -> list[RetrieverInfo]: ...
    def search(self, query: str, k: int, vector_store, embedder, **params) -> list[SearchResult]: ...
    def preprocess(self, vector_store, **params) -> dict[str, Any]: ...
```

**Lazy Loading (Embeddings)**
```python
class Embedder:
    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
```

**Streamlit Caching**
```python
@st.cache_data(show_spinner="Generating chunks...")
def _get_chunks_cached(provider, splitter, text, **params): ...

@st.cache_resource
def get_embedder(model_name): ...
```

**Fragment-Based Rendering**
```python
@st.fragment
def render_main_content():
    # Isolated rerun scope - sidebar doesn't re-render on tab switches
    render_step_nav()
    # ... step content
```

---

## 7. Features

### 7.1 Document Upload

**Purpose:** Accept and parse documents in multiple formats

**Supported Formats:**
| Format | Extension | Parser |
|--------|-----------|--------|
| PDF | .pdf | Docling |
| Word | .docx | Docling |
| PowerPoint | .pptx | Docling |
| Excel | .xlsx | Docling |
| HTML | .html, .htm | Docling |
| Markdown | .md | Native |
| Plain Text | .txt | Native |
| Images | .png, .jpg, .jpeg, .bmp, .tiff | Docling OCR |

**Parsing Options:**
- Output format: Markdown, HTML, DocTags, JSON
- OCR: Enable for scanned documents
- Table extraction: Preserve table structure
- Content filtering: Exclude headers, footers, etc.
- Image extraction: Extract embedded images
- Character limit: Cap parsed content length

### 7.2 Text Chunking

**Purpose:** Split documents into semantic chunks for embedding

**Chunking Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| Hierarchical | One chunk per document element | Structured documents with clear sections |
| Hybrid | Token-aware with structure preservation | RAG with embedding model token limits |

**Parameters:**
- Max tokens (default: 512)
- Chunk overlap (default: 50 tokens)
- Tokenizer (cl100k_base, other tiktoken encodings)
- Merge peers (Hierarchical only)
- Include metadata (multiselect)

**Chunk Metadata (Configurable):**

Users can select which metadata fields to include in each chunk via multiselect:

| Metadata Field | Description |
|----------------|-------------|
| Section Hierarchy | List of parent headings for context |
| Element Type | Document element type(s) in the chunk |
| Token Count | Actual token count using tiktoken |
| Heading Text | Current section heading text |
| Start Index | Character start position in source |
| End Index | Character end position in source |

Default metadata: Section Hierarchy, Element Type

**Chunk Visualization:**
- Full chunk text displayed inline with colored backgrounds
- Metadata badges (page number, element type, character count)
- Section hierarchy breadcrumb above chunk text
- Overlap text highlighted with dashed orange border
- Expandable context preview (click arrow at bottom of chunk):
  - Shows extra context prepended for embedding (section path, element type label)
  - Displays "No extra context added" when chunk is used as-is

### 7.3 Embedding Generation

**Purpose:** Convert text chunks to dense vector representations

**Available Models:**
| Model | Dimensions | Description |
|-------|------------|-------------|
| all-MiniLM-L6-v2 | 384 | Fast, lightweight (22M params) |
| all-mpnet-base-v2 | 768 | Higher quality (110M params) |
| paraphrase-MiniLM-L3-v2 | 384 | Fastest, smallest (17M params) |
| multi-qa-MiniLM-L6-cos-v1 | 384 | Optimized for QA retrieval |

**Features:**
- L2 normalization for cosine similarity
- Batch processing with configurable batch size
- GPU acceleration (auto-detected)

### 7.4 Embedding Visualization

**Purpose:** Visualize high-dimensional embeddings in 2D space

**Visualization:**
- UMAP dimensionality reduction (2D projection)
- Interactive Plotly scatter plot
- Color coding by document or cluster
- Hover tooltips with chunk preview
- Click to select and inspect chunks

### 7.5 Query & Retrieval

**Purpose:** Test retrieval quality with multiple search strategies

**Retrieval Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| Dense (Qdrant) | Vector similarity search using embeddings | Semantic search, finding conceptually similar content |
| Sparse (BM25) | Keyword-based search using BM25 algorithm | Exact keyword matches, technical terms |
| Hybrid | Combines dense + sparse with score fusion | Best of both worlds, robust retrieval |

**Hybrid Search Configuration:**
- **Dense Weight:** Configurable 0.0-1.0 (default 0.7) - controls weighting between vector and keyword search
- **Fusion Methods:**
  - **Weighted Sum:** Combines normalized scores with configurable weights
  - **Reciprocal Rank Fusion (RRF):** Rank-based fusion for combining result lists

**Reranking:**
- Cross-encoder reranking using FlashRank
- Improves result quality by reordering with more accurate models
- Configurable models:
  - ms-marco-MiniLM-L-12-v2 (default)
  - ms-marco-TinyBERT-L-2-v2
- Configurable top-N results to keep after reranking

**Core Features:**
- Free-text query input
- Configurable top-K results (1-20)
- Similarity/relevance scoring
- Source chunk highlighting
- Query embedding visualization on plot
- Automatic fallback to dense retrieval if BM25 index unavailable

**LLM Integration:**
- RAG-augmented answer generation
- Context injection from retrieved chunks
- Configurable system prompts
- Temperature and max token controls

### 7.6 Code Export

**Purpose:** Generate production-ready Python code

**Export Contents:**
- Document parsing code with exact configuration
- Chunking setup with selected strategy
- Embedding generation with chosen model
- Qdrant index creation and querying
- Complete requirements.txt

---

## 8. Technology Stack

### Core Dependencies

| Component | Package | Version | Purpose |
|-----------|---------|---------|---------|
| UI Framework | streamlit | >=1.28 | Web application interface |
| UI Components | streamlit-shadcn-ui | >=0.8.0 | Modern UI components |
| Visualization | plotly | >=5.18 | Interactive charts |
| Document Parsing | docling | >=1.0 | Universal document conversion |
| Embeddings | sentence-transformers | >=2.2 | Text embeddings |
| Vector Search | faiss-cpu | >=1.7 | Dense similarity search |
| Sparse Retrieval | rank-bm25 | >=0.2.2 | BM25 keyword search |
| Dim. Reduction | umap-learn | >=0.5 | Embedding visualization |
| Tokenization | tiktoken | >=0.5 | Token counting |
| CLI | click | >=8.1 | Command-line interface |
| DOCX Support | python-docx | >=1.1 | Word document parsing |
| Markdown | markdown | >=3.5 | Markdown processing |

### Optional Dependencies

**LLM Integration:**
```toml
[project.optional-dependencies]
llm = [
    "openai>=1.0",
    "anthropic>=0.18",
]
```

**Reranking:**
```toml
reranking = [
    "flashrank>=0.2.0",
]
```

**NLP Enhancements:**
```toml
nlp = [
    "nltk>=3.8",
    "spacy>=3.7",
]
```

**Development:**
```toml
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
]
```

### System Requirements

- **Python:** 3.9, 3.10, 3.11, 3.12
- **OS:** Windows, macOS, Linux
- **Memory:** 4GB+ recommended (embedding models)
- **GPU:** Optional (CUDA for acceleration)

---

## 9. Security & Configuration

### Configuration Management

**Environment Variables:**
| Variable | Purpose |
|----------|---------|
| OPENAI_API_KEY | OpenAI API authentication |
| ANTHROPIC_API_KEY | Anthropic API authentication |

**Local Storage:**
```
~/.unravel/
├── documents/     # Uploaded raw documents
├── chunks/        # Processed chunk data
├── embeddings/    # Cached embeddings
├── indices/       # Vector indices (Qdrant storage)
├── session/
│   ├── session_state.json
│   ├── bm25_index.pkl        # BM25 sparse index
│   ├── reduced_embeddings.npy
│   └── current_vector_store/
├── llm_config.json
└── rag_config.json
```

### Security Scope

**In Scope:**
- ✅ Local-only data processing (no cloud transmission)
- ✅ API key storage in local config files
- ✅ Environment variable support for sensitive keys
- ✅ No telemetry or usage tracking

**Out of Scope:**
- ❌ User authentication
- ❌ Encryption at rest
- ❌ Audit logging
- ❌ Role-based access control

### Deployment

**Local Installation:**
```bash
pip install unravel
unravel --port 8501
```

**Development:**
```bash
git clone https://github.com/unravel/unravel.git
cd unravel
pip install -e ".[dev]"
```

---

## 10. Success Criteria

### MVP Success Definition

The MVP is successful when a developer can:
1. Upload a document and see it parsed correctly
2. View chunks with clear boundaries and metadata
3. Generate and visualize embeddings
4. Run a query and see relevant chunks retrieved
5. Export working Python code

### Functional Requirements

**Document Processing:**
- ✅ Parse PDF documents with text and table extraction
- ✅ Parse DOCX, PPTX, XLSX, HTML, Markdown, Text files
- ✅ Handle documents up to 1MB / 40,000 characters
- ✅ Display parsing progress and errors clearly

**Chunking:**
- ✅ Support at least 2 chunking strategies
- ✅ Allow configuration of chunk size and overlap
- ✅ Display chunk count, sizes, and boundaries
- ✅ Highlight chunk source in original document

**Embeddings:**
- ✅ Support at least 4 embedding models
- ✅ Generate embeddings in under 30 seconds for typical documents
- ✅ Display 2D visualization with interactive exploration
- ✅ Cache embeddings to avoid regeneration

**Query:**
- ✅ Return top-K results with similarity scores
- ✅ Response time under 1 second for indexed documents
- ✅ Optional LLM-generated answers

**Export:**
- ✅ Generate syntactically correct Python code
- ✅ Include all configuration parameters
- ✅ Provide requirements.txt with pinned versions

### Quality Indicators

- **Performance:** Page load under 3 seconds, query response under 1 second
- **Reliability:** No crashes on supported file formats
- **Usability:** New user can complete full pipeline in under 5 minutes

---

## 11. Implementation Phases

### Phase 1: Core Pipeline (Complete)

**Goal:** Establish end-to-end RAG pipeline with basic functionality

**Deliverables:**
- ✅ Document upload and PDF parsing
- ✅ Basic text chunking
- ✅ Embedding generation with sentence-transformers
- ✅ Qdrant vector indexing
- ✅ Simple query interface

**Validation:** User can upload PDF, chunk it, embed it, and query it.

### Phase 2: Enhanced Processing (Complete)

**Goal:** Expand document support and improve chunking quality

**Deliverables:**
- ✅ Docling integration for advanced PDF parsing
- ✅ Multi-format support (DOCX, PPTX, XLSX, HTML, Images)
- ✅ Hierarchical and Hybrid chunking strategies
- ✅ Content filtering options
- ✅ Image extraction capability

**Validation:** User can process diverse document types with structure preservation.

### Phase 3: Visualization & UX (Complete)

**Goal:** Make the pipeline transparent and interactive

**Deliverables:**
- ✅ UMAP embedding visualization
- ✅ Interactive chunk explorer
- ✅ Step-by-step navigation UI
- ✅ Configuration sidebar with save/restore
- ✅ Session persistence

**Validation:** User can visually understand their document's embedding space.

### Phase 4: Integration & Export (Complete)

**Goal:** Enable production deployment

**Deliverables:**
- ✅ LLM integration (OpenAI, Anthropic, local)
- ✅ Code export with templates
- ✅ Multiple output format support
- ✅ CLI interface

**Validation:** User can export working code and run it independently.

### Phase 5: Advanced Retrieval (Complete)

**Goal:** Enhance retrieval quality with multiple strategies

**Deliverables:**
- ✅ Dense retrieval (Qdrant vector similarity)
- ✅ Sparse retrieval (BM25 keyword search)
- ✅ Hybrid retrieval (combined dense + sparse with score fusion)
- ✅ Weighted sum and Reciprocal Rank Fusion (RRF) methods
- ✅ Optional cross-encoder reranking with FlashRank
- ✅ BM25 index persistence
- ✅ Configurable retrieval parameters in sidebar

**Validation:** User can compare retrieval strategies and improve result quality with reranking.

---

## 12. Future Considerations

### Post-MVP Enhancements

**Evaluation & Metrics:**
- Retrieval quality metrics (precision, recall, MRR)
- Chunk quality scoring
- Automated configuration recommendations

**Advanced Retrieval:**
- Multi-query retrieval strategies
- Hypothetical Document Embeddings (HyDE)
- Parent-document retrieval
- Ensemble retrieval combining multiple strategies
- Query expansion and rewriting
- Additional reranking models (Cohere, cross-encoder variants)

**Advanced Embedding:**
- OpenAI embeddings (text-embedding-3-small/large)
- Cohere embeddings
- Custom model loading from Hugging Face

**Production Integration:**
- Direct export to Pinecone, Weaviate, Qdrant
- Docker container deployment
- REST API exposure

**Collaboration:**
- Shareable configuration links
- Team workspaces
- Annotation and commenting

### Integration Opportunities

- **LangChain/LlamaIndex:** Export as chain/index configuration
- **MLflow:** Experiment tracking integration
- **Weights & Biases:** Embedding visualization export
- **GitHub:** Direct repository export

---

## 13. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Large document OOM** | High | Medium | Character limit cap, streaming parsing, clear memory warnings |
| **Slow embedding generation** | Medium | Medium | Progress indicators, caching, GPU auto-detection |
| **Docling parsing failures** | High | Low | Graceful error handling, fallback to raw text extraction |
| **Browser tab crashes** | High | Low | Fragment-based rendering, chunked data loading |
| **API key exposure** | High | Low | Environment variable support, local-only storage, no telemetry |

---

## 14. Appendix

### Related Documents

| Document | Location | Purpose |
|----------|----------|---------|
| README | `/README.md` | User-facing documentation |
| Development Guide | `/CLAUDE.md` | Coding standards and principles |
| Storage Guide | `/docs/STORAGE_GUIDE.md` | Local storage architecture |
| Export Features | `/docs/EXPORT_FEATURE.md` | Code export documentation |
| Parsing Features | `/docs/PARSING_FEATURES.md` | Document parsing details |

### Key Dependencies

| Dependency | Documentation |
|------------|---------------|
| Docling | https://docling-project.github.io/docling/ |
| Streamlit | https://docs.streamlit.io/ |
| sentence-transformers | https://www.sbert.net/ |
| Qdrant | https://qdrant.tech/ |
| UMAP | https://umap-learn.readthedocs.io/ |

### Repository Structure

```
RAG-Visualizer/
├── unravel/       # Main package
├── tests/                # Test suite
├── docs/                 # Documentation
├── .streamlit/           # Streamlit configuration
├── pyproject.toml        # Project metadata
├── README.md             # User documentation
└── CLAUDE.md             # Development guidelines
```
