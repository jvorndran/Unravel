# Docling-Only Refactoring Plan

This document outlines the plan to refocus the RAG-Visualizer project to exclusively use Docling for document processing and chunking.

---

## Overview

### Current State

| Area | Current Implementation |
|------|------------------------|
| PDF Parsing | pypdf, Docling, LlamaParse (3 options) |
| File Formats | PDF, DOCX, Markdown, Text |
| Chunking | 13 LangChain text splitters |
| Output Formats | Markdown, Original, Plain Text |

### Target State (Docling-Only)

| Area | Target Implementation |
|------|----------------------|
| Document Parsing | Docling only |
| File Formats | PDF, DOCX, PPTX, XLSX, HTML, Images |
| Chunking | HierarchicalChunker, HybridChunker |
| Output Formats | Markdown, HTML, DocTags, JSON |

---

## Phase 1: Remove Non-Docling Parsers

### 1.1 Delete Parser Functions

**File:** `rag_visualizer/utils/parsers.py`

| Function | Lines | Action |
|----------|-------|--------|
| `parse_pdf_pypdf()` | 107-150 | Delete |
| `parse_pdf_llamaparse()` | 390-437 | Delete |
| `parse_pdf()` dispatcher | 439-463 | Simplify to call docling directly |

**Before:**
```python
def parse_pdf(content: bytes, params: dict) -> tuple[str, list[ExtractedImage]]:
    engine = params.get("pdf_parser", "pypdf")
    if engine == "docling":
        return parse_pdf_docling(content, params)
    elif engine == "llamaparse":
        return parse_pdf_llamaparse(content, params), []
    else:
        return parse_pdf_pypdf(content, params), []
```

**After:**
```python
def parse_pdf(content: bytes, params: dict) -> tuple[str, list[ExtractedImage]]:
    return parse_pdf_docling(content, params)
```

### 1.2 Remove Import Statements

**File:** `rag_visualizer/utils/parsers.py` (lines 13-39)

Remove try/except blocks for:
- `pypdf`
- `llama_parse`

Keep:
- `python-docx`
- `markdown`
- `docling`

### 1.3 Update Dependencies

**File:** `pyproject.toml`

```toml
# Remove these dependencies:
"pypdf>=3.17",
"llama-parse>=0.4",

# Keep:
"docling>=1.0",
"python-docx>=1.1",
"markdown>=3.5",
```

### 1.4 Delete Obsolete Test File

**Delete:** `tests/test_sidebar_pdf_parser_selection.py` (660 lines)

This file tests multi-parser dropdown behavior which will no longer exist.

---

## Phase 2: Simplify Sidebar UI

### 2.1 Remove Parser Selection Dropdown

**File:** `rag_visualizer/ui/sidebar.py`

| Lines | Content | Action |
|-------|---------|--------|
| 130-152 | PDF parser dropdown | Delete |
| 134-145 | Parser display/value mapping dicts | Delete |
| 143-146 | Widget state sync for pdf_parser | Delete |

### 2.2 Remove Parser-Specific Option Sections

| Lines | Content | Action |
|-------|---------|--------|
| 200-227 | LlamaParse API key input | Delete |
| 400-407 | Fallback values for non-docling parsers | Delete |
| 409-426 | pypdf options conditional block | Delete |

### 2.3 Simplify Docling Options

Keep the Docling-specific options but remove conditional rendering:
- Compute device selection (auto/cpu/cuda/mps)
- OCR toggle
- Table structure extraction toggle
- Worker thread count
- Content filtering (DocItemLabel checkboxes)
- Image extraction toggle

**Estimated removal:** ~300 lines of conditional UI code

---

## Phase 3: Update Configuration Defaults

### 3.1 Clean Up Parsing Params

**Files to update:**
- `rag_visualizer/ui/sidebar.py` (lines 64-78)
- `rag_visualizer/ui/steps/upload.py` (lines 25-29)
- `rag_visualizer/app.py` (lines 76-77)

**Remove these keys:**
```python
"pdf_parser": "pypdf",
"llamaparse_api_key": "",
"preserve_structure": True,   # pypdf-only
"extract_tables": True,       # pypdf-only
```

**Final parsing params structure:**
```python
parsing_params = {
    # Docling options
    "docling_enable_ocr": False,
    "docling_table_structure": True,
    "docling_threads": 4,
    "docling_filter_labels": ["PAGE_HEADER", "PAGE_FOOTER"],
    "docling_extract_images": False,
    "docling_enable_captioning": False,
    "docling_device": "auto",
    # Output options
    "output_format": "markdown",
    "normalize_whitespace": True,
    "remove_special_chars": False,
    "max_characters": 40_000,
}
```

---

## Phase 4: Simplify Export Code Generation

### 4.1 Remove Parser Templates

**File:** `rag_visualizer/services/export/templates.py`

| Template | Action |
|----------|--------|
| `PARSING_PYPDF` | Delete (~48 lines) |
| `PARSING_LLAMAPARSE` | Delete (~41 lines) |
| `PARSING_DOCLING` | Keep |

### 4.2 Simplify Generator

**File:** `rag_visualizer/services/export/generator.py`

| Lines | Content | Action |
|-------|---------|--------|
| 36-40 | `PARSER_DEPENDENCIES` dict | Keep only docling entry |
| 96-140 | `generate_parsing_code()` if/elif chain | Simplify to direct docling call |
| 251-257 | Parser dependency resolution | Simplify |
| 276-280 | Parser-based PDF detection | Remove conditional |

---

## Phase 5: Replace LangChain Chunking with Docling Chunkers

### 5.1 Remove LangChain Chunking Provider

**Delete:** `rag_visualizer/services/chunking/providers/langchain_provider.py`

**Modify:** `rag_visualizer/services/chunking/core.py`
- Remove provider registry pattern (only one provider now)
- Or keep pattern but register only Docling provider

### 5.2 Implement Docling Chunkers

**Create:** `rag_visualizer/services/chunking/providers/docling_provider.py`

```python
from docling.chunking import HybridChunker
from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

CHUNKING_STRATEGIES = {
    "hierarchical": {
        "name": "Hierarchical",
        "description": "Structure-aware chunking that creates one chunk per document element",
        "class": HierarchicalChunker,
    },
    "hybrid": {
        "name": "Hybrid",
        "description": "Tokenization-aware refinement on top of hierarchical chunking",
        "class": HybridChunker,
    },
}
```

### 5.3 Update Chunking UI

**File:** `rag_visualizer/ui/sidebar.py`

Replace 13-strategy dropdown with 2 options:

```
Chunking Strategy
├── Strategy: [Hierarchical | Hybrid]
├── Tokenizer: [HuggingFace model selector | OpenAI]
├── Max Tokens: [slider, default 512]
└── Merge Peers: [toggle, for Hybrid only]
```

### 5.4 Update Dependencies

**File:** `pyproject.toml`

```toml
# Remove:
"langchain-text-splitters>=0.3",

# Add/ensure:
"docling-core[chunking]>=1.0",  # For HuggingFace tokenizers
```

---

## Phase 6: Expand File Format Support

### 6.1 Add New Format Parsers

**File:** `rag_visualizer/utils/parsers.py`

Add support for Docling-supported formats:

| Format | Extension | Implementation |
|--------|-----------|----------------|
| PowerPoint | .pptx | Use Docling DocumentConverter |
| Excel | .xlsx | Use Docling DocumentConverter |
| HTML | .html | Use Docling DocumentConverter |
| Images | .png, .jpg | Use Docling DocumentConverter |

### 6.2 Update File Upload UI

**File:** `rag_visualizer/ui/steps/upload.py`

Update accepted file types:
```python
SUPPORTED_FORMATS = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt", ".png", ".jpg", ".jpeg"]
```

---

## Phase 7: Add Docling Export Formats

### 7.1 Update Output Format Options

**File:** `rag_visualizer/ui/sidebar.py`

```python
OUTPUT_FORMATS = {
    "markdown": "Markdown",
    "html": "HTML",
    "doctags": "DocTags",
    "json": "JSON (Lossless)",
}
```

### 7.2 Implement Export Methods

**File:** `rag_visualizer/utils/parsers.py`

```python
def export_document(doc: DoclingDocument, format: str) -> str:
    if format == "markdown":
        return doc.export_to_markdown()
    elif format == "html":
        return doc.export_to_html()
    elif format == "doctags":
        return doc.export_to_document_tokens()
    elif format == "json":
        return doc.model_dump_json()
```

---

## Phase 8: Enhance Chunk Metadata Display

### 8.1 Update Chunk Visualization

**File:** `rag_visualizer/ui/steps/chunks.py`

Display Docling-specific metadata:
- Section hierarchy (headers/captions)
- Source page numbers
- Item types (paragraph, table, figure, etc.)
- Bounding box information (optional)

### 8.2 Add Contextualize Preview

Show the output of `chunker.contextualize(chunk)` which produces metadata-enriched text for embedding.

---

## Summary: Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Remove pypdf, llama-parse, langchain-text-splitters; add docling-core[chunking] |
| `rag_visualizer/utils/parsers.py` | Delete pypdf/llamaparse functions, add new format support |
| `rag_visualizer/ui/sidebar.py` | Remove ~300 lines of conditional UI, simplify chunking options |
| `rag_visualizer/ui/steps/upload.py` | Update supported formats |
| `rag_visualizer/ui/steps/chunks.py` | Update chunk metadata display |
| `rag_visualizer/services/chunking/` | Replace LangChain with Docling chunkers |
| `rag_visualizer/services/export/templates.py` | Remove pypdf/llamaparse templates |
| `rag_visualizer/services/export/generator.py` | Simplify to Docling-only |
| `rag_visualizer/app.py` | Update default config |

## Summary: Files to Delete

| File | Reason |
|------|--------|
| `tests/test_sidebar_pdf_parser_selection.py` | Tests obsolete multi-parser behavior |
| `rag_visualizer/services/chunking/providers/langchain_provider.py` | Replaced by Docling chunkers |

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| PDF Parser Options | 3 | 1 |
| Chunking Strategies | 13 | 2 |
| Supported File Formats | 4 | 9 |
| Export Formats | 3 | 4 |
| Lines of Code (removed) | - | ~1,260 |
| Dependencies (removed) | - | 3 |

---

## References

- [Docling Documentation](https://docling-project.github.io/docling/)
- [Docling Chunking Concepts](https://docling-project.github.io/docling/concepts/chunking/)
- [HybridChunker Example](https://docling-project.github.io/docling/examples/hybrid_chunking/)
- [IBM Granite-Docling Announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)
