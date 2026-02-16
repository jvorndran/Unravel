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


def _iter_lines_with_offsets(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    offsets: list[tuple[int, int, str]] = []
    cursor = 0
    for line in lines:
        start = cursor
        end = cursor + len(line)
        offsets.append((start, end, line))
        cursor = end
    return offsets


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges_sorted = sorted(ranges, key=lambda r: r[0])
    merged = [list(ranges_sorted[0])]
    for start, end in ranges_sorted[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _find_fenced_code_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    fence = None
    fence_start = None
    for start, end, line in _iter_lines_with_offsets(text):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = "```" if stripped.startswith("```") else "~~~"
            if fence is None:
                fence = marker
                fence_start = start
            elif marker == fence and fence_start is not None:
                ranges.append((fence_start, end))
                fence = None
                fence_start = None
    if fence is not None and fence_start is not None:
        ranges.append((fence_start, len(text)))
    return ranges


def _is_markdown_table_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$", line))


def _find_markdown_table_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    lines = _iter_lines_with_offsets(text)
    for i, (start, end, line) in enumerate(lines):
        if not _is_markdown_table_separator(line):
            continue
        if i == 0:
            continue
        prev_line = lines[i - 1][2]
        if "|" not in prev_line:
            continue
        table_start = lines[i - 1][0]
        table_end = end
        j = i + 1
        while j < len(lines):
            next_line = lines[j][2]
            if not next_line.strip() or "|" not in next_line:
                break
            table_end = lines[j][1]
            j += 1
        ranges.append((table_start, table_end))
    return ranges


def _is_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-+*]|\d+[.)])\s+", line))


def _is_indented_continuation(line: str) -> bool:
    return bool(re.match(r"^\s+", line)) and bool(line.strip())


def _find_list_block_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    lines = _iter_lines_with_offsets(text)
    i = 0
    while i < len(lines):
        start, _, line = lines[i]
        if not _is_list_item(line):
            i += 1
            continue
        block_start = start
        block_end = lines[i][1]
        i += 1
        while i < len(lines):
            _, end, next_line = lines[i]
            if not next_line.strip():
                break
            if _is_list_item(next_line) or _is_indented_continuation(next_line):
                block_end = end
                i += 1
                continue
            break
        ranges.append((block_start, block_end))
    return ranges


def _find_blank_line_boundaries(text: str) -> list[int]:
    boundaries: list[int] = []
    for match in re.finditer(r"\n{2,}", text):
        boundaries.append(match.end())
    return boundaries


def _find_last_boundary(boundaries: list[int], start_char: int, end_char: int) -> int | None:
    if not boundaries:
        return None
    idx = bisect_right(boundaries, end_char) - 1
    while idx >= 0 and boundaries[idx] <= start_char:
        idx -= 1
    return boundaries[idx] if idx >= 0 else None


def _adjust_for_protected_ranges(
    start_char: int,
    end_char: int,
    protected_ranges: list[tuple[int, int]],
) -> tuple[int, int]:
    if not protected_ranges:
        return start_char, end_char
    for range_start, range_end in protected_ranges:
        if range_start <= start_char < range_end:
            start_char = range_start
            if end_char < range_end:
                end_char = range_end
        if range_start < end_char <= range_end:
            end_char = range_end
    return start_char, end_char


def _chunk_raw_hybrid(
    text: str,
    tokenizer_name: str,
    max_tokens: int,
    chunk_overlap: int,
    include_metadata: list[str],
    heading_index: tuple[list[int], list[list[str]], list[str]] | None,
    paragraph_aligned: bool = False,
    merge_list_items: bool = False,
    keep_code_blocks: bool = False,
    keep_tables: bool = False,
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

    protected_ranges: list[tuple[int, int]] = []
    if keep_code_blocks:
        protected_ranges.extend(_find_fenced_code_ranges(text))
    if keep_tables:
        protected_ranges.extend(_find_markdown_table_ranges(text))
    if merge_list_items:
        protected_ranges.extend(_find_list_block_ranges(text))
    protected_ranges = _merge_ranges(protected_ranges)

    blank_line_boundaries = _find_blank_line_boundaries(text) if paragraph_aligned else []

    chunks: list[Chunk] = []
    start_token = 0
    while start_token < len(tokens):
        base_end_token = min(start_token + safe_max_tokens, len(tokens))
        start_char = offsets[start_token]
        base_end_char = len(text) if base_end_token >= len(tokens) else offsets[base_end_token]
        end_char = base_end_char

        if paragraph_aligned:
            boundary = _find_last_boundary(blank_line_boundaries, start_char, end_char)
            if boundary:
                end_char = boundary

        start_char, end_char = _adjust_for_protected_ranges(start_char, end_char, protected_ranges)

        if end_char <= start_char:
            end_char = base_end_char
            start_char, end_char = _adjust_for_protected_ranges(
                start_char, end_char, protected_ranges
            )

        if end_char <= start_char:
            end_char = min(len(text), start_char + 1)

        end_token = bisect_right(offsets, max(0, end_char - 1))
        if end_token <= start_token:
            end_token = min(start_token + 1, len(tokens))

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
            chunk.metadata["token_count"] = _count_tokens(chunk_text, tokenizer_name)
        chunks.append(chunk)

        if end_token >= len(tokens) or end_char >= len(text):
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
                metadata["element_type"] = (
                    labels if len(labels) > 1 else labels[0] if labels else "text"
                )

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


def _build_html_heading_index(
    text: str,
) -> tuple[list[int], list[list[str]], list[str]]:
    """Build an index of HTML headings for section hierarchy lookup.

    NOTE: Currently unused. HTML format uses native Docling chunking instead of raw text chunking.
    This function is kept for potential future use if raw HTML text chunking is needed.

    Parses HTML heading tags (h1-h6) and builds a hierarchy similar to markdown headings.
    """
    positions: list[int] = []
    hierarchies: list[list[str]] = []
    headings: list[str] = []
    current: list[str] = []

    # Match HTML heading tags (h1-h6) and extract text content
    for match in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", text, flags=re.IGNORECASE | re.DOTALL):
        level = int(match.group(1))
        # Extract text content and strip HTML tags from it
        raw_title = match.group(2)
        # Remove any nested HTML tags and decode entities
        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        # Decode common HTML entities
        title = (
            title.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
        )
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
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
                        "paragraph_aligned",
                        "bool",
                        False,
                        "Prefer chunk boundaries at blank lines when possible",
                    ),
                    ParameterInfo(
                        "merge_list_items",
                        "bool",
                        False,
                        "Keep contiguous list items together",
                    ),
                    ParameterInfo(
                        "keep_code_blocks",
                        "bool",
                        False,
                        "Avoid splitting fenced code blocks",
                    ),
                    ParameterInfo(
                        "keep_tables",
                        "bool",
                        False,
                        "Avoid splitting markdown tables",
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

    def chunk(self, splitter_name: str, text: str, **params: Any) -> list[Chunk]:  # noqa: ANN401
        """Split text using specified Docling chunking strategy."""
        if not text:
            return []

        output_format = params.pop("output_format", "markdown")
        normalized_format = (output_format or "markdown").strip().lower()
        # Only use raw text chunking for markdown format
        # HTML format uses native Docling chunking to preserve document structure
        if normalized_format == "markdown":
            include_metadata = params.get("include_metadata", DEFAULT_METADATA)
            if include_metadata is None:
                include_metadata = DEFAULT_METADATA
            # Build heading index for markdown
            heading_index = _build_markdown_heading_index(text)
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
                paragraph_aligned = params.get("paragraph_aligned", False)
                merge_list_items = params.get("merge_list_items", False)
                keep_code_blocks = params.get("keep_code_blocks", False)
                keep_tables = params.get("keep_tables", False)
                return _chunk_raw_hybrid(
                    text=text,
                    tokenizer_name=tokenizer_name,
                    max_tokens=max_tokens,
                    chunk_overlap=chunk_overlap,
                    include_metadata=include_metadata,
                    heading_index=heading_index,
                    paragraph_aligned=paragraph_aligned,
                    merge_list_items=merge_list_items,
                    keep_code_blocks=keep_code_blocks,
                    keep_tables=keep_tables,
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
            start_index = (
                original_text.find(chunk_text[:50])
                if len(chunk_text) >= 50
                else original_text.find(chunk_text)
            )
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
            start_index = (
                original_text.find(chunk_text[:50])
                if len(chunk_text) >= 50
                else original_text.find(chunk_text)
            )
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
