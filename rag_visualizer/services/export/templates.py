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
from docling.datamodel.accelerator_options import AcceleratorOptions
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
            device="auto",
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
    text = result.document.export_to_markdown()
{post_processing}
    return text


# Example usage:
# text = parse_pdf("document.pdf")
# print(f"Extracted {{len(text)}} characters")
'''

PARSING_DOCX = '''"""Document Parsing for DOCX files

Configuration:
- Output Format: {output_format}
- Normalize Whitespace: {normalize_whitespace}
- Remove Special Characters: {remove_special_chars}
"""

from docx import Document
import re


def parse_docx(file_path: str) -> str:
    """Parse DOCX file and return extracted text.

    Args:
        file_path: Path to the DOCX file

    Returns:
        Extracted text content with markdown headings
    """
    doc = Document(file_path)
    text_parts = []

    for paragraph in doc.paragraphs:
        if not paragraph.text.strip():
            continue

        style_name = paragraph.style.name if paragraph.style else ""

        # Convert headings to markdown
        if "Heading 1" in style_name:
            text_parts.append(f"# {{paragraph.text}}")
        elif "Heading 2" in style_name:
            text_parts.append(f"## {{paragraph.text}}")
        elif "Heading 3" in style_name:
            text_parts.append(f"### {{paragraph.text}}")
        else:
            text_parts.append(paragraph.text)

    text = "\\n\\n".join(text_parts)
{post_processing}
    return text


# Example usage:
# text = parse_docx("document.docx")
# print(f"Extracted {{len(text)}} characters")
'''

PARSING_TEXT = '''"""Document Parsing for Text/Markdown files

Configuration:
- Normalize Whitespace: {normalize_whitespace}
- Remove Special Characters: {remove_special_chars}
"""

import re


def parse_text(file_path: str) -> str:
    """Parse text file and return content.

    Args:
        file_path: Path to the text file

    Returns:
        File content as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
{post_processing}
    return text


# Example usage:
# text = parse_text("document.txt")
# print(f"Read {{len(text)}} characters")
'''

# =============================================================================
# CHUNKING TEMPLATES
# =============================================================================

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
# EMBEDDING TEMPLATE
# =============================================================================

EMBEDDING_TEMPLATE = '''"""Embedding Generation with SentenceTransformers

Model: {model_name}
Dimension: {dimension}
Description: {description}
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the embedding model
model = SentenceTransformer("{model_name}")


def generate_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process at once

    Returns:
        numpy array of shape (len(texts), {dimension})
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Recommended for cosine similarity
        convert_to_numpy=True,
    )

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Generate embedding for a single query.

    Args:
        query: Query text to embed

    Returns:
        numpy array of shape ({dimension},)
    """
    return model.encode(query, normalize_embeddings=True, convert_to_numpy=True)


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

# =============================================================================
# TEMPLATE REGISTRIES
# =============================================================================

PARSING_TEMPLATES = {
    "docling": PARSING_DOCLING,
}

CHUNKING_TEMPLATES = {
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
