"""Docling-style chunking provider using native Docling chunkers.

Provides structure-aware and token-aware chunking strategies using Docling's
native HierarchicalChunker and HybridChunker.
"""

import re
from bisect import bisect_right
from typing import Any

import tiktoken

from docling.chunking import HierarchicalChunker, HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from ..core import Chunk
from .base import ChunkingProvider, ParameterInfo, SplitterInfo


# Centralized metadata options for all chunking strategies
# Maps display name -> internal key
METADATA_OPTIONS = {
    "Section Hierarchy": "section_hierarchy",
    "Element Type": "element_type",
    "Token Count": "token_count",
    "Heading Text": "heading_text",
    "Page Numbers": "page_no",
}

# Default metadata to include
DEFAULT_METADATA = ["Section Hierarchy", "Element Type"]

# All available display options (for UI)
METADATA_DISPLAY_OPTIONS = list(METADATA_OPTIONS.keys())


def _count_tokens(text: str, tokenizer_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback: estimate ~4 chars per token
        return len(text) // 4


def _get_token_offsets(
    text: str, tokenizer_name: str
) -> tuple[tiktoken.Encoding, list[int], list[int]]:
    """Return tokenizer, tokens, and token start offsets in characters."""
    enc = tiktoken.get_encoding(tokenizer_name)
    tokens = enc.encode(text)
    offsets: list[int] = []
    cursor = 0
    for token in tokens:
        offsets.append(cursor)
        cursor += len(enc.decode([token]))
    return enc, tokens, offsets


def _chunk_raw_hierarchical(
    text: str,
    merge_small_chunks: bool,
    min_chunk_size: int,
    include_metadata: list[str],
    heading_index: tuple[list[int], list[list[str]], list[str]] | None,
) -> list[Chunk]:
    """Chunk raw text by paragraph boundaries while preserving markdown."""
    if not text:
        return []

    blocks: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n{2,}", text):
        end = match.end()
        if end > start:
            block_text = text[start:end]
            if block_text.strip():
                blocks.append((start, end))
        start = end
    if start < len(text):
        block_text = text[start:]
        if block_text.strip():
            blocks.append((start, len(text)))

    if not blocks:
        return [
            Chunk(
                text=text,
                start_index=0,
                end_index=len(text),
                metadata={
                    "strategy": "Hierarchical",
                    "provider": "Docling",
                    "chunk_index": 0,
                    "size": len(text),
                },
            )
        ]

    chunks: list[Chunk] = []
    current_start = blocks[0][0]
    current_end = blocks[0][1]

    for block_start, block_end in blocks[1:]:
        current_size = current_end - current_start
        if merge_small_chunks and current_size < min_chunk_size:
            current_end = block_end
            continue

        chunks.append(
            _build_raw_chunk(
                text=text,
                start=current_start,
                end=current_end,
                chunk_index=len(chunks),
                strategy="Hierarchical",
                include_metadata=include_metadata,
                heading_index=heading_index,
            )
        )
        current_start = block_start
        current_end = block_end

    if current_end > current_start:
        chunks.append(
            _build_raw_chunk(
                text=text,
                start=current_start,
                end=current_end,
                chunk_index=len(chunks),
                strategy="Hierarchical",
                include_metadata=include_metadata,
                heading_index=heading_index,
            )
        )

    return chunks


def _chunk_raw_hybrid(
    text: str,
    tokenizer_name: str,
    max_tokens: int,
    chunk_overlap: int,
    include_metadata: list[str],
    heading_index: tuple[list[int], list[list[str]], list[str]] | None,
) -> list[Chunk]:
    """Chunk raw text by tokens while preserving original formatting."""
    if not text:
        return []
    safe_max_tokens = max(1, int(max_tokens or 1))
    safe_overlap = max(0, int(chunk_overlap or 0))
    if safe_overlap >= safe_max_tokens:
        safe_overlap = max(0, safe_max_tokens - 1)

    enc, tokens, offsets = _get_token_offsets(text, tokenizer_name)
    if not tokens:
        return [
            Chunk(
                text=text,
                start_index=0,
                end_index=len(text),
                metadata={
                    "strategy": "Hybrid",
                    "provider": "Docling",
                    "chunk_index": 0,
                    "size": len(text),
                    "token_count": 0,
                },
            )
        ]

    chunks: list[Chunk] = []
    start_token = 0
    while start_token < len(tokens):
        end_token = min(start_token + safe_max_tokens, len(tokens))
        start_char = offsets[start_token]
        end_char = len(text) if end_token >= len(tokens) else offsets[end_token]
        chunk_text = text[start_char:end_char]

        chunk = _build_raw_chunk(
            text=text,
            start=start_char,
            end=end_char,
            chunk_index=len(chunks),
            strategy="Hybrid",
            include_metadata=include_metadata,
            heading_index=heading_index,
        )
        if "token_count" in include_metadata:
            chunk.metadata["token_count"] = end_token - start_token
        chunks.append(chunk)

        if end_token >= len(tokens):
            break
        start_token = end_token - safe_overlap if safe_overlap else end_token

    return chunks


def _extract_metadata_from_chunk(
    native_chunk: Any,
    include_metadata: list[str],
    chunk_index: int,
    strategy: str,
    tokenizer_name: str = "cl100k_base",
) -> dict[str, Any]:
    """Extract metadata from a native Docling chunk.
    
    Args:
        native_chunk: Native BaseChunk from Docling
        include_metadata: List of metadata fields to include
        chunk_index: Index of this chunk in the sequence
        strategy: Chunking strategy name ("Hierarchical" or "Hybrid")
        tokenizer_name: Name of tokenizer for token counting
        
    Returns:
        Dictionary of metadata fields
    """
    chunk_text = native_chunk.text
    metadata: dict[str, Any] = {
        "strategy": strategy,
        "provider": "Docling",
        "chunk_index": chunk_index,
        "size": len(chunk_text),
    }
    
    # Extract metadata from native chunk
    if hasattr(native_chunk, "meta") and native_chunk.meta:
        meta = native_chunk.meta
        
    # Section hierarchy from headings (always include for UI)
    if hasattr(meta, "headings"):
        if meta.headings:
            # Headings might be strings or objects with .text attribute
            metadata["section_hierarchy"] = [
                h.text if hasattr(h, "text") else str(h) for h in meta.headings
            ]
            last_heading = meta.headings[-1]
            metadata["heading_text"] = (
                last_heading.text if hasattr(last_heading, "text") else str(last_heading)
            )
        
        # Element type from doc_items
        if "element_type" in include_metadata and hasattr(meta, "doc_items"):
            if meta.doc_items:
                # Get unique labels from doc items
                labels = list(set(str(item.label) for item in meta.doc_items))
                metadata["element_type"] = labels if len(labels) > 1 else labels[0] if labels else "text"

    # Page numbers from provenance (often outside meta in Docling)
    if "page_no" in include_metadata:
        pages = set()
        # Check native_chunk.prov directly (preferred in newer Docling)
        if hasattr(native_chunk, "prov") and native_chunk.prov:
            for p in native_chunk.prov:
                if hasattr(p, "page_no"):
                    pages.add(p.page_no)
        
        # Fallback to doc_items provenance if direct prov is empty/missing
        if not pages and hasattr(native_chunk, "meta") and native_chunk.meta:
            if hasattr(native_chunk.meta, "doc_items") and native_chunk.meta.doc_items:
                for item in native_chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        for p in item.prov:
                            if hasattr(p, "page_no"):
                                pages.add(p.page_no)
        
        if pages:
            metadata["page_no"] = sorted(list(pages))
    
    # Token count
    if "token_count" in include_metadata:
        metadata["token_count"] = _count_tokens(chunk_text, tokenizer_name)
    
    return metadata


def _build_markdown_heading_index(
    text: str,
) -> tuple[list[int], list[list[str]], list[str]]:
    """Build an index of markdown headings for section hierarchy lookup."""
    positions: list[int] = []
    hierarchies: list[list[str]] = []
    headings: list[str] = []
    current: list[str] = []

    for match in re.finditer(r"^(#{1,6})\s+(.*)$", text, flags=re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        if not title:
            continue
        current = current[: level - 1]
        current.append(title)
        positions.append(match.start())
        hierarchies.append(current.copy())
        headings.append(title)

    return positions, hierarchies, headings


def _build_raw_chunk(
    text: str,
    start: int,
    end: int,
    chunk_index: int,
    strategy: str,
    include_metadata: list[str],
    heading_index: tuple[list[int], list[list[str]], list[str]] | None,
) -> Chunk:
    metadata: dict[str, Any] = {
        "strategy": strategy,
        "provider": "Docling",
        "chunk_index": chunk_index,
        "size": max(0, end - start),
    }
    if heading_index:
        positions, hierarchies, headings = heading_index
        idx = bisect_right(positions, start) - 1
        if idx >= 0:
            metadata["section_hierarchy"] = hierarchies[idx]
            metadata["heading_text"] = headings[idx]

    return Chunk(
        text=text[start:end],
        start_index=start,
        end_index=end,
        metadata=metadata,
    )


def _text_to_docling_document(text: str, output_format: str):
    """Convert plain text/markdown to a DoclingDocument for chunking.
    
    Uses Docling's document converter to parse markdown text into a structured document.
    """
    import tempfile
    from pathlib import Path
    
    normalized = (output_format or "markdown").strip().lower()
    if normalized == "html":
        suffix = ".html"
    else:
        suffix = ".md"

    # Create a temporary file matching the parsed output format
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        temp_path = f.name
    
    try:
        converter = DocumentConverter()
        result = converter.convert(Path(temp_path))
        return result.document
    finally:
        # Clean up temp file
        try:
            Path(temp_path).unlink()
        except Exception:
            pass


class DoclingProvider(ChunkingProvider):
    """Docling-style chunking provider with structure and token-aware strategies."""

    @property
    def name(self) -> str:
        return "Docling"

    @property
    def display_name(self) -> str:
        return "Docling"

    @property
    def attribution(self) -> str:
        return "Structure-aware chunking"

    def get_available_splitters(self) -> list[SplitterInfo]:
        """Return available Docling chunking strategies."""
        return [
            SplitterInfo(
                name="HierarchicalChunker",
                display_name="Hierarchical",
                description="Structure-aware chunking that creates one chunk per document element (paragraph, header, list, code block). Best for preserving document structure.",
                category="Structure-Aware",
                parameters=[
                    ParameterInfo(
                        "merge_small_chunks",
                        "bool",
                        True,
                        "Merge very small adjacent chunks",
                    ),
                    ParameterInfo(
                        "min_chunk_size",
                        "int",
                        50,
                        "Minimum chunk size in characters before merging",
                        min_value=10,
                        max_value=500,
                    ),
                    # Note: min_chunk_size kept for backwards compatibility with export templates
                    # Native HierarchicalChunker doesn't use this parameter directly
                    ParameterInfo(
                        "include_metadata",
                        "multiselect",
                        DEFAULT_METADATA,
                        "Metadata fields to include in each chunk",
                        options=METADATA_DISPLAY_OPTIONS,
                    ),
                ],
            ),
            SplitterInfo(
                name="HybridChunker",
                display_name="Hybrid",
                description="Token-aware chunking that respects structure while maintaining token limits. Best for embedding models with fixed context windows.",
                category="Token-Aware",
                parameters=[
                    ParameterInfo(
                        "max_tokens",
                        "int",
                        512,
                        "Maximum tokens per chunk",
                        min_value=64,
                        max_value=8192,
                    ),
                    ParameterInfo(
                        "chunk_overlap",
                        "int",
                        50,
                        "Token overlap between chunks",
                        min_value=0,
                        max_value=512,
                    ),
                    ParameterInfo(
                        "tokenizer",
                        "str",
                        "cl100k_base",
                        "Tokenizer to use for counting",
                        options=["cl100k_base", "p50k_base", "o200k_base"],
                    ),
                    ParameterInfo(
                        "include_metadata",
                        "multiselect",
                        DEFAULT_METADATA,
                        "Metadata fields to include in each chunk",
                        options=METADATA_DISPLAY_OPTIONS,
                    ),
                ],
            ),
        ]

    def chunk(
        self, splitter_name: str, text: str, **params: Any  # noqa: ANN401
    ) -> list[Chunk]:
        """Split text using specified Docling chunking strategy."""
        if not text:
            return []

        output_format = params.pop("output_format", "markdown")
        normalized_format = (output_format or "markdown").strip().lower()
        if normalized_format in ("markdown", "html"):
            include_metadata = params.get("include_metadata", DEFAULT_METADATA)
            if include_metadata is None:
                include_metadata = DEFAULT_METADATA
            heading_index = (
                _build_markdown_heading_index(text)
                if normalized_format == "markdown"
                else None
            )
            if splitter_name == "HierarchicalChunker":
                merge_small_chunks = params.get("merge_small_chunks", True)
                min_chunk_size = int(params.get("min_chunk_size", 50) or 50)
                return _chunk_raw_hierarchical(
                    text=text,
                    merge_small_chunks=merge_small_chunks,
                    min_chunk_size=min_chunk_size,
                    include_metadata=include_metadata,
                    heading_index=heading_index,
                )
            if splitter_name == "HybridChunker":
                tokenizer_name = params.get("tokenizer", "cl100k_base")
                max_tokens = params.get("max_tokens", 512)
                chunk_overlap = params.get("chunk_overlap", 50)
                return _chunk_raw_hybrid(
                    text=text,
                    tokenizer_name=tokenizer_name,
                    max_tokens=max_tokens,
                    chunk_overlap=chunk_overlap,
                    include_metadata=include_metadata,
                    heading_index=heading_index,
                )

        # Convert text to DoclingDocument
        try:
            doc = _text_to_docling_document(text, output_format)
        except Exception as e:
            # Fallback: if conversion fails, return single chunk
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    metadata={
                        "strategy": splitter_name,
                        "provider": "Docling",
                        "chunk_index": 0,
                        "size": len(text),
                        "error": f"Document conversion failed: {str(e)}",
                    },
                )
            ]

        if splitter_name == "HierarchicalChunker":
            return self._chunk_hierarchical(doc, text, **params)
        elif splitter_name == "HybridChunker":
            return self._chunk_hybrid(doc, text, **params)
        else:
            raise ValueError(f"Unknown splitter: {splitter_name}")

    def _chunk_hierarchical(
        self,
        doc: Any,
        original_text: str,
        **params: Any,  # noqa: ANN401
    ) -> list[Chunk]:
        """Hierarchical chunking using native HierarchicalChunker."""
        merge_list_items = params.get("merge_small_chunks", True)
        include_metadata = params.get("include_metadata", DEFAULT_METADATA)
        if include_metadata is None:
            include_metadata = DEFAULT_METADATA

        # Create native HierarchicalChunker
        chunker = HierarchicalChunker(merge_list_items=merge_list_items)

        # Generate chunks using native chunker
        try:
            native_chunks = list(chunker.chunk(dl_doc=doc))
        except Exception as e:
            # Fallback: return single chunk on error
            return [
                Chunk(
                    text=original_text,
                    start_index=0,
                    end_index=len(original_text),
                    metadata={
                        "strategy": "Hierarchical",
                        "provider": "Docling",
                        "chunk_index": 0,
                        "size": len(original_text),
                        "error": f"Chunking failed: {str(e)}",
                    },
                )
            ]

        # Convert native chunks to our Chunk dataclass
        chunks: list[Chunk] = []
        for i, native_chunk in enumerate(native_chunks):
            chunk_text = native_chunk.text
            
            # Extract metadata using helper
            metadata = _extract_metadata_from_chunk(
                native_chunk, include_metadata, i, "Hierarchical"
            )

            # Find chunk position in original text (approximate)
            start_index = original_text.find(chunk_text[:50]) if len(chunk_text) >= 50 else original_text.find(chunk_text)
            if start_index == -1:
                start_index = 0
            end_index = start_index + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    metadata=metadata,
                )
            )

        return chunks

    def _chunk_hybrid(
        self,
        doc: Any,
        original_text: str,
        **params: Any,  # noqa: ANN401
    ) -> list[Chunk]:
        """Hybrid chunking using native HybridChunker."""
        max_tokens = params.get("max_tokens", 512)
        merge_peers = params.get("chunk_overlap", 50) > 0  # Convert overlap to merge_peers
        tokenizer_name = params.get("tokenizer", "cl100k_base")
        include_metadata = params.get("include_metadata", DEFAULT_METADATA)
        if include_metadata is None:
            include_metadata = DEFAULT_METADATA

        # Create tokenizer for HybridChunker
        try:
            enc = tiktoken.get_encoding(tokenizer_name)
            tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=max_tokens)
        except Exception:
            # Fallback to default
            enc = tiktoken.get_encoding("cl100k_base")
            tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=max_tokens)

        # Create native HybridChunker
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=merge_peers)

        # Generate chunks using native chunker
        try:
            native_chunks = list(chunker.chunk(dl_doc=doc))
        except Exception as e:
            # Fallback: return single chunk on error
            return [
                Chunk(
                    text=original_text,
                    start_index=0,
                    end_index=len(original_text),
                    metadata={
                        "strategy": "Hybrid",
                        "provider": "Docling",
                        "chunk_index": 0,
                        "size": len(original_text),
                        "error": f"Chunking failed: {str(e)}",
                    },
                )
            ]

        # Convert native chunks to our Chunk dataclass
        chunks: list[Chunk] = []
        for i, native_chunk in enumerate(native_chunks):
            chunk_text = native_chunk.text
            
            # Extract metadata using helper
            metadata = _extract_metadata_from_chunk(
                native_chunk, include_metadata, i, "Hybrid", tokenizer_name
            )

            # Find chunk position in original text (approximate)
            start_index = original_text.find(chunk_text[:50]) if len(chunk_text) >= 50 else original_text.find(chunk_text)
            if start_index == -1:
                start_index = 0
            end_index = start_index + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    metadata=metadata,
                )
            )

        return chunks

