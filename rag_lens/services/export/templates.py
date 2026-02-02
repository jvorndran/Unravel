"""Code templates for pipeline export.

Each template is a string with placeholders that get filled in by the generator.
"""

# =============================================================================
# PARSING TEMPLATES
# =============================================================================

PARSING_DOCLING = '''"""Document Parsing with Docling

Advanced PDF parsing with optional OCR and table extraction.

Configuration:
- Enable OCR: {enable_ocr}
- Enable Table Structure: {enable_tables}
- Threads: {num_threads}
"""

import os
import tempfile
from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline, ThreadedPdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend


def parse_pdf(file_path: str) -> str:
    """Parse PDF file using Docling.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text in markdown format
    """
    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            num_threads={num_threads},
            device=AcceleratorDevice.AUTO,
        ),
        do_ocr={enable_ocr},
        do_table_structure={enable_tables},
        generate_page_images=False,
        generate_table_images=False,
        generate_picture_images=False,
    )

    format_option = FormatOption(
        pipeline_options=pipeline_options,
        backend=DoclingParseV4DocumentBackend,
        pipeline_cls=StandardPdfPipeline,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={{InputFormat.PDF: format_option}},
    )

    result = converter.convert(file_path)
    return result.document.export_to_markdown()


# Example usage:
# text = parse_pdf("document.pdf")
# print(f"Extracted {{len(text)}} characters")
'''

PARSING_DOCX = '''"""Document Parsing for DOCX files

Configuration:
- Output Format: {output_format}
"""

from docx import Document


def parse_docx(file_path: str) -> str:
    """Parse DOCX file and return extracted text.

    Args:
        file_path: Path to the DOCX file

    Returns:
        Extracted text content
    """
    doc = Document(file_path)
    text_parts = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    return "\\n\\n".join(text_parts)


# Example usage:
# text = parse_docx("document.docx")
# print(f"Extracted {{len(text)}} characters")
'''

PARSING_TEXT = '''"""Document Parsing for Text/Markdown files
"""


def parse_text(file_path: str) -> str:
    """Parse text file and return content.

    Args:
        file_path: Path to the text file

    Returns:
        File content as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# Example usage:
# text = parse_text("document.txt")
# print(f"Read {{len(text)}} characters")
'''

# =============================================================================
# CHUNKING TEMPLATES
# =============================================================================

CHUNKING_HIERARCHICAL = '''"""Text Chunking with Hierarchical Strategy

Structure-aware chunking that creates one chunk per document element (paragraph,
header, list, code block). Best for preserving document structure.

Uses Docling's native HierarchicalChunker for accurate structure detection.

Configuration:
- Merge List Items: {merge_small_chunks}
"""

import tempfile
from pathlib import Path

from docling.chunking import HierarchicalChunker
from docling.document_converter import DocumentConverter


def chunk_text(text: str) -> list[str]:
    """Split text into hierarchical chunks using Docling's HierarchicalChunker.

    Args:
        text: Source text to split

    Returns:
        List of text chunks
    """
    # Convert text to DoclingDocument via temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(text)
        temp_path = f.name

    try:
        # Parse text into structured document
        converter = DocumentConverter()
        result = converter.convert(Path(temp_path))
        doc = result.document

        # Chunk using hierarchical strategy
        chunker = HierarchicalChunker(merge_list_items={merge_small_chunks})
        chunks = list(chunker.chunk(doc))

        # Extract text from chunk objects
        return [chunk.text for chunk in chunks]

    finally:
        # Clean up temp file
        try:
            Path(temp_path).unlink()
        except Exception:
            pass


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
# for i, chunk in enumerate(chunks[:3]):
#     print(f"Chunk {{i+1}}: {{chunk[:100]}}...")
'''

CHUNKING_HYBRID = '''"""Text Chunking with Hybrid Strategy

Token-aware chunking that respects document structure while maintaining token limits.
Best for embedding models with fixed context windows.

Uses Docling's native HybridChunker for token-aware splitting.

Configuration:
- Max Tokens: {max_tokens}
- Chunk Overlap: {chunk_overlap} tokens
- Tokenizer: {tokenizer}
"""

import tempfile
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter


def chunk_text(text: str) -> list[str]:
    """Split text into token-aware chunks using Docling's HybridChunker.

    Args:
        text: Source text to split

    Returns:
        List of text chunks
    """
    # Convert text to DoclingDocument via temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(text)
        temp_path = f.name

    try:
        # Parse text into structured document
        converter = DocumentConverter()
        result = converter.convert(Path(temp_path))
        doc = result.document

        # Chunk using hybrid strategy
        chunker = HybridChunker(
            max_tokens={max_tokens},
            tokenizer="{tokenizer}",
        )
        chunks = list(chunker.chunk(doc))

        # Extract text from chunk objects
        return [chunk.text for chunk in chunks]

    finally:
        # Clean up temp file
        try:
            Path(temp_path).unlink()
        except Exception:
            pass


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
# for i, chunk in enumerate(chunks[:3]):
#     print(f"Chunk {{i+1}}: {{chunk[:100]}}...")
'''

CHUNKING_RECURSIVE = '''"""Text Chunking with RecursiveCharacterTextSplitter

Best for most documents. Splits by paragraphs, then sentences, then words
to maintain semantic coherence.

Configuration:
- Chunk Size: {chunk_size} characters
- Chunk Overlap: {chunk_overlap} characters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Source text to split

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
        length_function=len,
        add_start_index=True,
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
# for i, chunk in enumerate(chunks[:3]):
#     print(f"Chunk {{i+1}}: {{chunk[:100]}}...")
'''

CHUNKING_CHARACTER = '''"""Text Chunking with CharacterTextSplitter

Simple split on a separator. Use for well-structured text with consistent delimiters.

Configuration:
- Chunk Size: {chunk_size} characters
- Chunk Overlap: {chunk_overlap} characters
- Separator: {separator_display}
"""

from langchain_text_splitters import CharacterTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split text into chunks on separator.

    Args:
        text: Source text to split

    Returns:
        List of text chunks
    """
    splitter = CharacterTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
        separator={separator},
        length_function=len,
        add_start_index=True,
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_TOKEN = '''"""Text Chunking with TokenTextSplitter

Splits by tokens (not characters). Use when you need precise token counts
for LLM context windows.

Configuration:
- Chunk Size: {chunk_size} tokens
- Chunk Overlap: {chunk_overlap} tokens
- Encoding: {encoding_name}
"""

from langchain_text_splitters import TokenTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split text into chunks by token count.

    Args:
        text: Source text to split

    Returns:
        List of text chunks
    """
    splitter = TokenTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
        encoding_name="{encoding_name}",
        add_start_index=True,
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_MARKDOWN = '''"""Text Chunking with MarkdownTextSplitter

Use for Markdown files. Keeps headings with their content for better semantic chunks.

Configuration:
- Chunk Size: {chunk_size} characters
- Chunk Overlap: {chunk_overlap} characters
"""

from langchain_text_splitters import MarkdownTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split markdown text into semantic chunks.

    Args:
        text: Markdown text to split

    Returns:
        List of text chunks
    """
    splitter = MarkdownTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
        add_start_index=True,
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(markdown_text)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_LATEX = '''"""Text Chunking with LatexTextSplitter

Use for LaTeX/academic documents. Respects sections, equations, and environments.

Configuration:
- Chunk Size: {chunk_size} characters
- Chunk Overlap: {chunk_overlap} characters
"""

from langchain_text_splitters import LatexTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split LaTeX text into semantic chunks.

    Args:
        text: LaTeX text to split

    Returns:
        List of text chunks
    """
    splitter = LatexTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
        add_start_index=True,
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(latex_text)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_PYTHON = '''"""Text Chunking with PythonCodeTextSplitter

Use for Python code. Splits at function/class boundaries to keep code units intact.

Configuration:
- Chunk Size: {chunk_size} characters
- Chunk Overlap: {chunk_overlap} characters
"""

from langchain_text_splitters import PythonCodeTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split Python code into semantic chunks.

    Args:
        text: Python code to split

    Returns:
        List of code chunks
    """
    splitter = PythonCodeTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
        add_start_index=True,
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(python_code)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_HTML = '''"""Text Chunking with HTMLHeaderTextSplitter

Use for HTML pages. Creates chunks based on heading hierarchy (h1, h2, h3).
"""

from langchain_text_splitters import HTMLHeaderTextSplitter


def chunk_text(html: str) -> list[str]:
    """Split HTML into chunks based on headers.

    Args:
        html: HTML content to split

    Returns:
        List of text chunks
    """
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]

    splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    documents = splitter.split_text(html)

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(html_content)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_JSON = '''"""Text Chunking with RecursiveJsonSplitter

Use for JSON data. Keeps nested objects together while respecting size limits.

Configuration:
- Max Chunk Size: {max_chunk_size} characters
- Min Chunk Size: {min_chunk_size} characters
"""

import json
from langchain_text_splitters import RecursiveJsonSplitter


def chunk_json(json_text: str) -> list[str]:
    """Split JSON into semantic chunks.

    Args:
        json_text: JSON string to split

    Returns:
        List of JSON string chunks
    """
    splitter = RecursiveJsonSplitter(
        max_chunk_size={max_chunk_size},
        min_chunk_size={min_chunk_size},
    )

    json_data = json.loads(json_text)
    json_chunks = splitter.split_json(json_data)

    return [json.dumps(chunk, indent=2) for chunk in json_chunks]


# Example usage:
# chunks = chunk_json(json_string)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_SENTENCE_TRANSFORMERS = '''"""Text Chunking with SentenceTransformersTokenTextSplitter

Aligns with embedding model tokenization. Use for optimal embedding boundaries.

Configuration:
- Tokens Per Chunk: {tokens_per_chunk}
- Chunk Overlap: {chunk_overlap} tokens
- Model: {model_name}
"""

from langchain_text_splitters import SentenceTransformersTokenTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split text aligned to embedding model tokens.

    Args:
        text: Text to split

    Returns:
        List of text chunks
    """
    splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk={tokens_per_chunk},
        chunk_overlap={chunk_overlap},
        model_name="{model_name}",
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_NLTK = '''"""Text Chunking with NLTKTextSplitter

Uses NLTK for sentence detection. Good for natural language with proper punctuation.

Configuration:
- Chunk Size: {chunk_size} characters
- Chunk Overlap: {chunk_overlap} characters
"""

from langchain_text_splitters import NLTKTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split text using NLTK sentence boundaries.

    Args:
        text: Text to split

    Returns:
        List of text chunks
    """
    splitter = NLTKTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
'''

CHUNKING_SPACY = '''"""Text Chunking with SpacyTextSplitter

Uses spaCy NLP for intelligent text segmentation. Best for complex natural language.

Configuration:
- Chunk Size: {chunk_size} characters
- Chunk Overlap: {chunk_overlap} characters
- Pipeline: {pipeline}
"""

from langchain_text_splitters import SpacyTextSplitter


def chunk_text(text: str) -> list[str]:
    """Split text using spaCy NLP pipeline.

    Args:
        text: Text to split

    Returns:
        List of text chunks
    """
    splitter = SpacyTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
        pipeline="{pipeline}",
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
'''

# Generic fallback
CHUNKING_GENERIC = '''"""Text Chunking with {splitter_name}

Configuration:
{config_comment}
"""

from langchain_text_splitters import {splitter_name}


def chunk_text(text: str) -> list[str]:
    """Split text into chunks.

    Args:
        text: Text to split

    Returns:
        List of text chunks
    """
    splitter = {splitter_name}(
{params_code}
    )

    documents = splitter.create_documents([text])

    return [doc.page_content for doc in documents]


# Example usage:
# chunks = chunk_text(text)
# print(f"Created {{len(chunks)}} chunks")
'''

# =============================================================================
# EMBEDDING TEMPLATES
# =============================================================================

EMBEDDING_TEMPLATE = '''"""Embedding Generation with {backend_display}

Backend: {backend}
Model: {model_name}
Dimension: {dimension}
Description: {description}
"""

{import_statement}
import numpy as np

# Initialize the embedding model
{model_init}


def generate_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process at once

    Returns:
        numpy array of shape (len(texts), {dimension})
    """
{encode_texts_code}

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Generate embedding for a single query.

    Args:
        query: Query text to embed

    Returns:
        numpy array of shape ({dimension},)
    """
{encode_query_code}


# Example usage:
# texts = ["First chunk", "Second chunk", "Third chunk"]
# embeddings = generate_embeddings(texts)
# print(f"Generated embeddings with shape: {{embeddings.shape}}")
#
# # Similarity search example
# query_embedding = embed_query("search query")
# similarities = np.dot(embeddings, query_embedding)  # Cosine similarity (normalized)
# print(f"Similarities: {{similarities}}")
'''

# Backend-specific snippets
EMBEDDING_IMPORTS = {
    "sentence-transformers": "from sentence_transformers import SentenceTransformer",
    "flagembedding": "from FlagEmbedding import FlagModel",
}

EMBEDDING_MODEL_INIT = {
    "sentence-transformers": 'model = SentenceTransformer("{model_name}")',
    "flagembedding": 'model = FlagModel("{model_name}", use_fp16=True)',
}

EMBEDDING_ENCODE_TEXTS = {
    "sentence-transformers": """    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Recommended for cosine similarity
        convert_to_numpy=True,
    )""",
    "flagembedding": """    embeddings = model.encode(texts, batch_size=batch_size)

    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype(np.float32)""",
}

EMBEDDING_ENCODE_QUERY = {
    "sentence-transformers": """    return model.encode(query, normalize_embeddings=True, convert_to_numpy=True)""",
    "flagembedding": """    embedding = model.encode_queries([query])
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding[0].astype(np.float32)""",
}

BACKEND_DISPLAY_NAMES = {
    "sentence-transformers": "SentenceTransformers",
    "flagembedding": "FlagEmbedding (BGE)",
}

# =============================================================================
# TEMPLATE REGISTRIES
# =============================================================================

PARSING_TEMPLATES = {
    "docling": PARSING_DOCLING,
}

CHUNKING_TEMPLATES = {
    # Docling chunkers (primary)
    "HierarchicalChunker": CHUNKING_HIERARCHICAL,
    "HybridChunker": CHUNKING_HYBRID,
    # Legacy LangChain splitters (for backwards compatibility in exports)
    "RecursiveCharacterTextSplitter": CHUNKING_RECURSIVE,
    "CharacterTextSplitter": CHUNKING_CHARACTER,
    "TokenTextSplitter": CHUNKING_TOKEN,
    "MarkdownTextSplitter": CHUNKING_MARKDOWN,
    "LatexTextSplitter": CHUNKING_LATEX,
    "PythonCodeTextSplitter": CHUNKING_PYTHON,
    "HTMLHeaderTextSplitter": CHUNKING_HTML,
    "RecursiveJsonSplitter": CHUNKING_JSON,
    "SentenceTransformersTokenTextSplitter": CHUNKING_SENTENCE_TRANSFORMERS,
    "NLTKTextSplitter": CHUNKING_NLTK,
    "SpacyTextSplitter": CHUNKING_SPACY,
}

# =============================================================================
# RETRIEVAL TEMPLATES
# =============================================================================

RETRIEVAL_DENSE = '''"""Dense Retrieval using Cosine Similarity

Retrieve relevant chunks using semantic similarity between query and chunk embeddings.

Configuration:
- Top K: {top_k}
"""

import numpy as np


def retrieve_dense(query: str, chunks: list[str], embeddings: np.ndarray, top_k: int = {top_k}) -> list[tuple[str, float]]:
    """Retrieve top-k most similar chunks using dense retrieval.

    Args:
        query: Query string
        chunks: List of text chunks
        embeddings: Chunk embeddings (normalized)
        top_k: Number of results to return

    Returns:
        List of (chunk, score) tuples sorted by relevance
    """
    # Generate query embedding
    query_embedding = embed_query(query)

    # Compute cosine similarity (embeddings are already normalized)
    similarities = embeddings @ query_embedding

    # Get top-k indices
    top_indices = similarities.argsort()[::-1][:top_k]

    # Return chunks with scores
    return [(chunks[i], float(similarities[i])) for i in top_indices]


# Example usage:
# results = retrieve_dense("What is the main topic?", chunks, embeddings, top_k=5)
# for chunk, score in results:
#     print(f"Score: {{score:.4f}} - {{chunk[:100]}}...")
'''

RETRIEVAL_SPARSE = '''"""Sparse Retrieval using BM25

Retrieve relevant chunks using keyword-based BM25 algorithm.

Configuration:
- Top K: {top_k}
"""

from rank_bm25 import BM25Okapi


def retrieve_sparse(query: str, chunks: list[str], top_k: int = {top_k}) -> list[tuple[str, float]]:
    """Retrieve top-k most similar chunks using BM25.

    Args:
        query: Query string
        chunks: List of text chunks
        top_k: Number of results to return

    Returns:
        List of (chunk, score) tuples sorted by relevance
    """
    # Tokenize chunks (simple whitespace split)
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_chunks)

    # Tokenize query
    query_tokens = query.lower().split()

    # Get BM25 scores
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices
    top_indices = scores.argsort()[::-1][:top_k]

    # Return chunks with scores
    return [(chunks[i], float(scores[i])) for i in top_indices]


# Example usage:
# results = retrieve_sparse("What is the main topic?", chunks, top_k=5)
# for chunk, score in results:
#     print(f"Score: {{score:.4f}} - {{chunk[:100]}}...")
'''

RETRIEVAL_HYBRID = '''"""Hybrid Retrieval using Dense + Sparse Fusion

Combine dense (semantic) and sparse (keyword) retrieval for better results.

Configuration:
- Dense Weight: {dense_weight}
- Sparse Weight: {sparse_weight}
- Fusion Method: {fusion_method}
- Top K: {top_k}
"""

import numpy as np
from rank_bm25 import BM25Okapi


def retrieve_hybrid(
    query: str,
    chunks: list[str],
    embeddings: np.ndarray,
    top_k: int = {top_k},
    dense_weight: float = {dense_weight},
    sparse_weight: float = {sparse_weight},
    fusion_method: str = "{fusion_method}",
) -> list[tuple[str, float]]:
    """Retrieve top-k chunks using hybrid dense + sparse retrieval.

    Args:
        query: Query string
        chunks: List of text chunks
        embeddings: Chunk embeddings (normalized)
        top_k: Number of results to return
        dense_weight: Weight for dense scores
        sparse_weight: Weight for sparse scores
        fusion_method: 'weighted_sum' or 'rrf' (reciprocal rank fusion)

    Returns:
        List of (chunk, score) tuples sorted by relevance
    """
    # Dense retrieval
    query_embedding = embed_query(query)
    dense_scores = embeddings @ query_embedding

    # Sparse retrieval (BM25)
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = query.lower().split()
    sparse_scores = bm25.get_scores(query_tokens)

    # Fusion
    if fusion_method == "weighted_sum":
        # Normalize scores to [0, 1] and combine
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)
        sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-9)
        combined_scores = dense_weight * dense_norm + sparse_weight * sparse_norm
    elif fusion_method == "rrf":
        # Reciprocal rank fusion
        k = 60  # RRF constant
        dense_ranks = dense_scores.argsort()[::-1]
        sparse_ranks = sparse_scores.argsort()[::-1]

        # Create rank maps
        dense_rank_map = {{idx: rank for rank, idx in enumerate(dense_ranks)}}
        sparse_rank_map = {{idx: rank for rank, idx in enumerate(sparse_ranks)}}

        # Compute RRF scores
        combined_scores = np.array([
            dense_weight / (k + dense_rank_map[i]) + sparse_weight / (k + sparse_rank_map[i])
            for i in range(len(chunks))
        ])
    else:
        raise ValueError(f"Unknown fusion method: {{fusion_method}}")

    # Get top-k indices
    top_indices = combined_scores.argsort()[::-1][:top_k]

    # Return chunks with combined scores
    return [(chunks[i], float(combined_scores[i])) for i in top_indices]


# Example usage:
# results = retrieve_hybrid("What is the main topic?", chunks, embeddings, top_k=5)
# for chunk, score in results:
#     print(f"Score: {{score:.4f}} - {{chunk[:100]}}...")
'''

RETRIEVAL_TEMPLATES = {
    "DenseRetriever": RETRIEVAL_DENSE,
    "SparseRetriever": RETRIEVAL_SPARSE,
    "HybridRetriever": RETRIEVAL_HYBRID,
}

# =============================================================================
# RERANKING TEMPLATE
# =============================================================================

RERANKING_TEMPLATE = '''"""Reranking with {model_name}

Rerank retrieved chunks using a cross-encoder model for improved relevance.

Library: {library}
Description: {description}

Configuration:
- Model: {model_name}
- Top N: {top_n}
"""

{import_statement}


{init_code}


def rerank(query: str, chunks: list[str], top_n: int = {top_n}) -> list[tuple[str, float]]:
    """Rerank chunks using cross-encoder model.

    Args:
        query: Query string
        chunks: List of text chunks to rerank
        top_n: Number of top results to return after reranking

    Returns:
        List of (chunk, score) tuples sorted by reranked relevance
    """
{rerank_code}


# Example usage:
# # First retrieve candidates (e.g., top 20 from dense/sparse/hybrid retrieval)
# candidates = retrieve_hybrid(query, chunks, embeddings, top_k=20)
# candidate_chunks = [chunk for chunk, _ in candidates]
#
# # Then rerank to get best top_n results
# reranked = rerank(query, candidate_chunks, top_n=5)
# for chunk, score in reranked:
#     print(f"Score: {{score:.4f}} - {{chunk[:100]}}...")
'''

# =============================================================================
# LLM TEMPLATE
# =============================================================================

LLM_TEMPLATE = '''"""RAG Response Generation with {provider}

Generate responses using retrieved context and LLM.

Provider: {provider}
Model: {model}

Configuration:
- Temperature: {temperature}
- Max Tokens: {max_tokens}
{base_url_display}
"""

{import_statement}


{client_init}


def generate_rag_response(
    query: str,
    context_chunks: list[str],
    system_prompt: str = """{system_prompt}""",
    temperature: float = {temperature},
    max_tokens: int = {max_tokens},
) -> str:
    """Generate RAG response using retrieved context.

    Args:
        query: User query
        context_chunks: List of relevant text chunks
        system_prompt: System prompt for the LLM
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated response string
    """
    # Format context
    context = "\\n\\n".join([f"[{{i+1}}] {{chunk}}" for i, chunk in enumerate(context_chunks)])

    # Create user message
    user_message = f"""Context:
{{context}}

Question: {{query}}

Please answer the question based on the provided context."""

{generation_code}


# Example usage:
# # Retrieve relevant chunks
# results = retrieve_hybrid(query, chunks, embeddings, top_k=20)
# candidate_chunks = [chunk for chunk, _ in results]
#
# # Optional: Rerank
# reranked = rerank(query, candidate_chunks, top_n=5)
# top_chunks = [chunk for chunk, _ in reranked]
#
# # Generate response
# response = generate_rag_response(query, top_chunks)
# print(f"Answer: {{response}}")
'''
