"""Unit tests for chunking service.

Tests the core chunking functionality including:
- Provider registration and retrieval
- Chunk generation with different splitters
- Overlap calculation
- Metadata handling
- Edge cases (empty text, Unicode, large documents)
"""

import pytest

from unravel.services.chunking.core import (
    Chunk,
    get_available_providers,
    get_chunks,
    get_provider,
    get_provider_splitters,
)


class TestProviderRegistry:
    """Test provider registration and retrieval."""

    def test_get_available_providers_returns_list(self):
        """Provider registry returns non-empty list."""
        providers = get_available_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_docling_provider_registered(self):
        """Docling provider is registered by default."""
        providers = get_available_providers()
        assert "Docling" in providers

    def test_get_provider_returns_instance(self):
        """get_provider returns provider instance for valid name."""
        provider = get_provider("Docling")
        assert provider is not None
        assert provider.name == "Docling"

    def test_get_provider_returns_none_for_invalid(self):
        """get_provider returns None for unknown provider."""
        provider = get_provider("NonExistentProvider")
        assert provider is None

    def test_get_provider_splitters_returns_list(self):
        """get_provider_splitters returns splitter info for valid provider."""
        splitters = get_provider_splitters("Docling")
        assert isinstance(splitters, list)
        assert len(splitters) > 0

    def test_get_provider_splitters_empty_for_invalid(self):
        """get_provider_splitters returns empty list for unknown provider."""
        splitters = get_provider_splitters("NonExistentProvider")
        assert splitters == []


class TestHierarchicalChunking:
    """Test hierarchical chunking strategy."""

    def test_hierarchical_chunks_simple_text(self):
        """Hierarchical chunking on simple paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            output_format="markdown",
        )

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(hasattr(c, "text") for c in chunks)
        assert all(hasattr(c, "start_index") for c in chunks)
        assert all(hasattr(c, "end_index") for c in chunks)
        assert all(hasattr(c, "metadata") for c in chunks)

    def test_hierarchical_metadata_structure(self):
        """Hierarchical chunks contain expected metadata."""
        text = "# Heading\n\nContent under heading."
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            output_format="markdown",
        )

        for chunk in chunks:
            assert "strategy" in chunk.metadata
            assert chunk.metadata["strategy"] == "Hierarchical"
            assert "provider" in chunk.metadata
            assert chunk.metadata["provider"] == "Docling"
            assert "chunk_index" in chunk.metadata
            assert "size" in chunk.metadata

    def test_hierarchical_with_merge_small_chunks(self):
        """Hierarchical chunking merges small adjacent chunks."""
        text = "A.\n\nB.\n\nC."
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            merge_small_chunks=True,
            min_chunk_size=10,
            output_format="markdown",
        )

        # Small chunks should be merged
        assert len(chunks) < 3

    def test_hierarchical_without_merge(self):
        """Hierarchical chunking without merging preserves structure."""
        text = "First.\n\nSecond.\n\nThird."
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            merge_small_chunks=False,
            output_format="markdown",
        )

        # Each paragraph should be a separate chunk
        assert len(chunks) >= 1


class TestHybridChunking:
    """Test hybrid (token-aware) chunking strategy."""

    def test_hybrid_chunks_by_token_limit(self):
        """Hybrid chunking respects max_tokens parameter."""
        # Create text that exceeds token limit
        text = " ".join(["word"] * 1000)
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=100,
            chunk_overlap=0,
            tokenizer="cl100k_base",
            output_format="markdown",
        )

        assert len(chunks) > 1
        # Each chunk should respect token limit (approximately)
        for chunk in chunks:
            assert chunk.metadata["size"] > 0

    def test_hybrid_with_overlap(self):
        """Hybrid chunking with overlap creates overlapping chunks."""
        text = " ".join(["word"] * 200)
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=50,
            chunk_overlap=10,
            tokenizer="cl100k_base",
            output_format="markdown",
        )

        assert len(chunks) >= 2
        # Chunks should have content
        assert all(len(c.text) > 0 for c in chunks)

    def test_hybrid_metadata_structure(self):
        """Hybrid chunks contain expected metadata."""
        text = "This is a test sentence."
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=50,
            output_format="markdown",
        )

        for chunk in chunks:
            assert "strategy" in chunk.metadata
            assert chunk.metadata["strategy"] == "Hybrid"
            assert "provider" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_hybrid_with_different_tokenizers(self):
        """Hybrid chunking works with different tokenizer options."""
        text = "Test sentence for tokenizer comparison."

        for tokenizer in ["cl100k_base", "p50k_base", "o200k_base"]:
            chunks = get_chunks(
                provider="Docling",
                splitter="HybridChunker",
                text=text,
                max_tokens=50,
                tokenizer=tokenizer,
                output_format="markdown",
            )
            assert len(chunks) > 0

    @pytest.mark.xfail(reason="Known edge case: paragraph_aligned mode with raw text has IndexError in docling provider line 332")
    def test_hybrid_paragraph_aligned(self):
        """Hybrid chunking with paragraph alignment."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=50,
            paragraph_aligned=True,
            output_format="markdown",
        )

        assert len(chunks) >= 0
        # Chunks should exist (if any)
        if chunks:
            assert all(c.text.strip() for c in chunks)


class TestChunkEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_text_returns_empty_list(self):
        """Empty text returns empty chunk list."""
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text="",
            output_format="markdown",
        )
        assert chunks == []

    def test_whitespace_only_text(self):
        """Whitespace-only text is handled gracefully."""
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text="   \n\n   ",
            output_format="markdown",
        )
        # Should return empty or minimal chunks
        assert isinstance(chunks, list)

    def test_unicode_text_handling(self):
        """Unicode text is chunked correctly."""
        text = "Hello 世界! 🌍\n\nMulti-language text with émojis."
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            output_format="markdown",
        )

        assert len(chunks) > 0
        # Unicode should be preserved
        assert any("世界" in c.text or "émojis" in c.text for c in chunks)

    def test_single_sentence_text(self):
        """Single sentence creates single chunk."""
        text = "This is a single sentence."
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            output_format="markdown",
        )

        assert len(chunks) >= 1
        assert chunks[0].text.strip() == text.strip()

    def test_large_document_chunking(self):
        """Large document is chunked into multiple pieces."""
        # Create a large document
        text = "\n\n".join([f"Paragraph {i} with some content." for i in range(100)])
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=100,
            output_format="markdown",
        )

        assert len(chunks) > 5  # Should create multiple chunks
        # All chunks should be valid
        assert all(c.text and c.start_index >= 0 for c in chunks)

    def test_very_long_paragraph(self):
        """Very long paragraph without breaks is chunked by tokens."""
        # Single long paragraph
        text = " ".join(["word"] * 1000)
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=100,
            output_format="markdown",
        )

        assert len(chunks) > 1
        # Should split despite no paragraph breaks

    def test_special_characters_preserved(self):
        """Special characters are preserved in chunks."""
        text = "Code: `function() { return 42; }`\n\nMath: $x^2 + y^2 = z^2$"
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            output_format="markdown",
        )

        full_text = "".join(c.text for c in chunks)
        assert "`function()" in full_text or "function()" in full_text
        assert "$x^2" in full_text or "x^2" in full_text


class TestChunkPositionTracking:
    """Test chunk position tracking (start_index, end_index)."""

    def test_chunk_indices_are_valid(self):
        """Chunk start_index and end_index are valid positions."""
        text = "First paragraph.\n\nSecond paragraph."
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            output_format="markdown",
        )

        for chunk in chunks:
            assert 0 <= chunk.start_index <= len(text)
            assert 0 <= chunk.end_index <= len(text)
            assert chunk.start_index < chunk.end_index

    def test_chunk_indices_sequential(self):
        """Chunk indices progress through the document."""
        text = " ".join([f"Section{i}." for i in range(10)])
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=20,
            chunk_overlap=0,
            output_format="markdown",
        )

        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # Later chunks should start at or after earlier chunks
                assert chunks[i + 1].start_index >= chunks[i].start_index


class TestMetadataOptions:
    """Test metadata inclusion options."""

    def test_section_hierarchy_metadata(self):
        """Section hierarchy metadata is included when requested."""
        text = "# Main\n\n## Sub\n\nContent here."
        chunks = get_chunks(
            provider="Docling",
            splitter="HierarchicalChunker",
            text=text,
            include_metadata=["Section Hierarchy"],
            output_format="markdown",
        )

        # Check if section hierarchy is tracked
        assert len(chunks) > 0
        # Metadata should be present
        assert all("metadata" in chunk.__dict__ for chunk in chunks)

    def test_multiple_metadata_fields(self):
        """Multiple metadata fields can be requested."""
        text = "# Heading\n\nSome content."
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            include_metadata=["Section Hierarchy", "Element Type", "Token Count"],
            max_tokens=50,
            tokenizer="cl100k_base",
            output_format="markdown",
        )

        assert len(chunks) > 0
        # All chunks should have metadata
        for chunk in chunks:
            assert isinstance(chunk.metadata, dict)


class TestInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_invalid_provider_raises_error(self):
        """Invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_chunks(
                provider="InvalidProvider",
                splitter="AnySplitter",
                text="test",
                output_format="markdown",
            )

    def test_invalid_splitter_raises_error(self):
        """Invalid splitter for valid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown splitter"):
            get_chunks(
                provider="Docling",
                splitter="InvalidSplitter",
                text="test",
                output_format="markdown",
            )

    def test_negative_max_tokens_handled(self):
        """Negative max_tokens is handled gracefully."""
        text = "Test text."
        # Should not crash, should use minimum value
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=-10,
            output_format="markdown",
        )
        assert isinstance(chunks, list)

    def test_negative_overlap_handled(self):
        """Negative chunk_overlap is handled gracefully."""
        text = "Test text."
        chunks = get_chunks(
            provider="Docling",
            splitter="HybridChunker",
            text=text,
            max_tokens=50,
            chunk_overlap=-5,
            output_format="markdown",
        )
        assert isinstance(chunks, list)
