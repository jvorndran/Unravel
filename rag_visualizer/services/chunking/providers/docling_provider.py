"""Docling-style chunking provider.

Provides structure-aware and token-aware chunking strategies inspired by Docling's
HierarchicalChunker and HybridChunker concepts.
"""

import re
from typing import Any

from ..core import Chunk
from .base import ChunkingProvider, ParameterInfo, SplitterInfo

# Try to import tiktoken for token counting, fall back to character-based
try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


def _count_tokens(text: str, tokenizer: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken or fallback to character estimate."""
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding(tokenizer)
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: estimate ~4 chars per token
    return len(text) // 4


def _split_into_elements(text: str) -> list[dict[str, Any]]:
    """Split text into structural elements (paragraphs, headers, lists, etc.).

    Returns list of dicts with 'type', 'text', 'start', 'end' keys.
    """
    elements = []
    current_pos = 0

    # Pattern to match structural elements
    # Headers (markdown-style)
    header_pattern = r"^(#{1,6})\s+(.+)$"
    # List items
    list_pattern = r"^(\s*[-*+]|\s*\d+\.)\s+(.+)$"
    # Code blocks
    code_block_pattern = r"^```[\s\S]*?```$"

    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        line_start = text.find(line, current_pos)
        if line_start == -1:
            line_start = current_pos

        # Check for code block start
        if line.strip().startswith("```"):
            # Find the end of the code block
            code_content = [line]
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code_content.append(lines[j])
                j += 1
            if j < len(lines):
                code_content.append(lines[j])
            code_text = "\n".join(code_content)
            code_start = line_start
            code_end = code_start + len(code_text)
            elements.append({
                "type": "code",
                "text": code_text,
                "start": code_start,
                "end": code_end,
            })
            current_pos = code_end + 1
            i = j + 1
            continue

        # Check for header
        header_match = re.match(header_pattern, line)
        if header_match:
            level = len(header_match.group(1))
            elements.append({
                "type": f"header_{level}",
                "text": line,
                "start": line_start,
                "end": line_start + len(line),
            })
            current_pos = line_start + len(line) + 1
            i += 1
            continue

        # Check for list item
        list_match = re.match(list_pattern, line)
        if list_match:
            elements.append({
                "type": "list_item",
                "text": line,
                "start": line_start,
                "end": line_start + len(line),
            })
            current_pos = line_start + len(line) + 1
            i += 1
            continue

        # Check for empty line (paragraph separator)
        if not line.strip():
            current_pos = line_start + len(line) + 1
            i += 1
            continue

        # Regular paragraph - collect consecutive non-empty lines
        para_lines = [line]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not re.match(header_pattern, lines[j]) and not lines[j].strip().startswith("```"):
            # Check if next line is a list item
            if re.match(list_pattern, lines[j]):
                break
            para_lines.append(lines[j])
            j += 1

        para_text = "\n".join(para_lines)
        para_end = line_start + len(para_text)
        elements.append({
            "type": "paragraph",
            "text": para_text,
            "start": line_start,
            "end": para_end,
        })
        current_pos = para_end + 1
        i = j

    return elements


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
                        "include_headers",
                        "bool",
                        True,
                        "Include section headers in chunk metadata for context",
                    ),
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
                ],
            ),
        ]

    def chunk(
        self, splitter_name: str, text: str, **params: Any  # noqa: ANN401
    ) -> list[Chunk]:
        """Split text using specified Docling chunking strategy."""
        if not text:
            return []

        if splitter_name == "HierarchicalChunker":
            return self._chunk_hierarchical(text, **params)
        elif splitter_name == "HybridChunker":
            return self._chunk_hybrid(text, **params)
        else:
            raise ValueError(f"Unknown splitter: {splitter_name}")

    def _chunk_hierarchical(
        self, text: str, **params: Any  # noqa: ANN401
    ) -> list[Chunk]:
        """Hierarchical chunking: one chunk per structural element."""
        include_headers = params.get("include_headers", True)
        merge_small = params.get("merge_small_chunks", True)
        min_size = params.get("min_chunk_size", 50)

        elements = _split_into_elements(text)
        if not elements:
            # Fallback: treat entire text as single chunk
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    metadata={
                        "strategy": "Hierarchical",
                        "provider": "Docling",
                        "element_type": "text",
                        "chunk_index": 0,
                        "size": len(text),
                    },
                )
            ]

        chunks = []
        current_headers: list[str] = []
        pending_chunk: dict[str, Any] | None = None

        for elem in elements:
            # Track headers for context
            if elem["type"].startswith("header_"):
                level = int(elem["type"].split("_")[1])
                # Truncate headers list to current level
                current_headers = current_headers[: level - 1]
                current_headers.append(elem["text"].lstrip("#").strip())

            # Build metadata
            metadata: dict[str, Any] = {
                "strategy": "Hierarchical",
                "provider": "Docling",
                "element_type": elem["type"],
                "size": len(elem["text"]),
            }

            if include_headers and current_headers:
                metadata["section_hierarchy"] = current_headers.copy()

            chunk_data = {
                "text": elem["text"],
                "start_index": elem["start"],
                "end_index": elem["end"],
                "metadata": metadata,
            }

            # Handle merging small chunks
            if merge_small and len(elem["text"]) < min_size:
                if pending_chunk is None:
                    pending_chunk = chunk_data
                else:
                    # Merge with pending
                    pending_chunk["text"] += "\n\n" + chunk_data["text"]
                    pending_chunk["end_index"] = chunk_data["end_index"]
                    pending_chunk["metadata"]["size"] = len(pending_chunk["text"])
                    pending_chunk["metadata"]["element_type"] = "merged"
            else:
                # Flush pending chunk if exists
                if pending_chunk is not None:
                    if len(pending_chunk["text"]) >= min_size:
                        pending_chunk["metadata"]["chunk_index"] = len(chunks)
                        chunks.append(Chunk(**pending_chunk))
                    else:
                        # Merge pending into current
                        chunk_data["text"] = pending_chunk["text"] + "\n\n" + chunk_data["text"]
                        chunk_data["start_index"] = pending_chunk["start_index"]
                        chunk_data["metadata"]["size"] = len(chunk_data["text"])
                        chunk_data["metadata"]["element_type"] = "merged"
                    pending_chunk = None

                chunk_data["metadata"]["chunk_index"] = len(chunks)
                chunks.append(Chunk(**chunk_data))

        # Flush any remaining pending chunk
        if pending_chunk is not None:
            pending_chunk["metadata"]["chunk_index"] = len(chunks)
            chunks.append(Chunk(**pending_chunk))

        return chunks

    def _chunk_hybrid(
        self, text: str, **params: Any  # noqa: ANN401
    ) -> list[Chunk]:
        """Hybrid chunking: token-aware with structure preservation."""
        max_tokens = params.get("max_tokens", 512)
        overlap = params.get("chunk_overlap", 50)
        tokenizer = params.get("tokenizer", "cl100k_base")

        # First, get structural elements
        elements = _split_into_elements(text)

        if not elements:
            # Fallback to simple token-based splitting
            return self._split_by_tokens(text, max_tokens, overlap, tokenizer)

        chunks = []
        current_text = ""
        current_start = 0
        current_headers: list[str] = []

        for elem in elements:
            # Track headers
            if elem["type"].startswith("header_"):
                level = int(elem["type"].split("_")[1])
                current_headers = current_headers[: level - 1]
                current_headers.append(elem["text"].lstrip("#").strip())

            elem_text = elem["text"]
            elem_tokens = _count_tokens(elem_text, tokenizer)

            # If element alone exceeds max_tokens, split it
            if elem_tokens > max_tokens:
                # First, flush current accumulator
                if current_text.strip():
                    chunks.append(self._make_chunk(
                        current_text,
                        current_start,
                        len(chunks),
                        current_headers.copy(),
                    ))

                # Split the large element by tokens
                sub_chunks = self._split_by_tokens(
                    elem_text, max_tokens, overlap, tokenizer, elem["start"]
                )
                for sub in sub_chunks:
                    sub.metadata["chunk_index"] = len(chunks)
                    sub.metadata["section_hierarchy"] = current_headers.copy()
                    chunks.append(sub)

                current_text = ""
                current_start = elem["end"] + 1
                continue

            # Check if adding this element would exceed max_tokens
            combined = current_text + ("\n\n" if current_text else "") + elem_text
            combined_tokens = _count_tokens(combined, tokenizer)

            if combined_tokens > max_tokens and current_text.strip():
                # Flush current accumulator
                chunks.append(self._make_chunk(
                    current_text,
                    current_start,
                    len(chunks),
                    current_headers.copy(),
                ))

                # Handle overlap by keeping some context
                if overlap > 0:
                    # Keep last part of previous chunk for overlap
                    overlap_text = self._get_overlap_text(current_text, overlap, tokenizer)
                    current_text = overlap_text + "\n\n" + elem_text if overlap_text else elem_text
                    current_start = elem["start"]
                else:
                    current_text = elem_text
                    current_start = elem["start"]
            else:
                if not current_text:
                    current_start = elem["start"]
                current_text = combined

        # Flush remaining text
        if current_text.strip():
            chunks.append(self._make_chunk(
                current_text,
                current_start,
                len(chunks),
                current_headers.copy(),
            ))

        return chunks

    def _make_chunk(
        self,
        text: str,
        start_index: int,
        chunk_index: int,
        headers: list[str],
    ) -> Chunk:
        """Create a Chunk object with standard metadata."""
        metadata: dict[str, Any] = {
            "strategy": "Hybrid",
            "provider": "Docling",
            "chunk_index": chunk_index,
            "size": len(text),
        }
        if headers:
            metadata["section_hierarchy"] = headers
        return Chunk(
            text=text,
            start_index=start_index,
            end_index=start_index + len(text),
            metadata=metadata,
        )

    def _split_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap: int,
        tokenizer: str,
        base_offset: int = 0,
    ) -> list[Chunk]:
        """Split text into chunks based on token count."""
        chunks = []
        words = text.split()
        current_words: list[str] = []
        current_start = 0

        for i, word in enumerate(words):
            current_words.append(word)
            current_text = " ".join(current_words)

            if _count_tokens(current_text, tokenizer) >= max_tokens:
                # Create chunk (excluding last word that pushed us over)
                if len(current_words) > 1:
                    chunk_text = " ".join(current_words[:-1])
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_index=base_offset + current_start,
                        end_index=base_offset + current_start + len(chunk_text),
                        metadata={
                            "strategy": "Hybrid",
                            "provider": "Docling",
                            "chunk_index": len(chunks),
                            "size": len(chunk_text),
                        },
                    ))

                    # Handle overlap
                    if overlap > 0:
                        overlap_words = self._get_overlap_words(current_words[:-1], overlap, tokenizer)
                        current_words = overlap_words + [word]
                    else:
                        current_words = [word]

                    current_start = text.find(current_words[0], current_start + len(chunk_text))
                    if current_start == -1:
                        current_start = base_offset + len(chunk_text)

        # Add remaining text
        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append(Chunk(
                text=chunk_text,
                start_index=base_offset + current_start,
                end_index=base_offset + current_start + len(chunk_text),
                metadata={
                    "strategy": "Hybrid",
                    "provider": "Docling",
                    "chunk_index": len(chunks),
                    "size": len(chunk_text),
                },
            ))

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int, tokenizer: str) -> str:
        """Get the last N tokens worth of text for overlap."""
        words = text.split()
        overlap_words: list[str] = []
        for word in reversed(words):
            test = " ".join([word] + overlap_words)
            if _count_tokens(test, tokenizer) > overlap_tokens:
                break
            overlap_words.insert(0, word)
        return " ".join(overlap_words)

    def _get_overlap_words(self, words: list[str], overlap_tokens: int, tokenizer: str) -> list[str]:
        """Get the last N tokens worth of words for overlap."""
        overlap_words: list[str] = []
        for word in reversed(words):
            test = " ".join([word] + overlap_words)
            if _count_tokens(test, tokenizer) > overlap_tokens:
                break
            overlap_words.insert(0, word)
        return overlap_words
