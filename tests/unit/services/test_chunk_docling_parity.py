"""Test that chunks shown in UI match DoclingDocument chunks.

This test verifies that:
- HTML format: chunks match native DoclingDocument chunks
- Markdown format: chunks are from raw text (expected to differ from DoclingDocument chunks)

Uses real example documents from tests/fixtures/documents/ to test realistic scenarios.
"""

import tempfile
from pathlib import Path

import pytest

from unravel.services.chunking import get_chunks
from unravel.utils.parsers import parse_document


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the fixtures directory path."""
    return Path(__file__).parent.parent.parent / "fixtures" / "documents"


@pytest.fixture
def sample_html_document(fixtures_dir) -> bytes:
    """Load real HTML document from fixtures."""
    html_file = fixtures_dir / "sample_document.html"
    assert html_file.exists(), f"HTML fixture not found: {html_file}"
    return html_file.read_bytes()


@pytest.fixture
def sample_markdown_document(fixtures_dir) -> bytes:
    """Load real markdown document from fixtures."""
    md_file = fixtures_dir / "sample_document.md"
    assert md_file.exists(), f"Markdown fixture not found: {md_file}"
    return md_file.read_bytes()


def test_html_chunks_match_docling_native(sample_html_document):
    """Test that HTML format chunks match native DoclingDocument chunks."""
    # Parse document with HTML output format
    parsed_text, output_format, _ = parse_document(
        filename="test.html",
        content=sample_html_document,
        params={
            "output_format": "html",
            "docling_enable_ocr": False,
        },
    )

    assert output_format.lower() == "html"
    assert parsed_text is not None

    # Get chunks using our chunking service
    ui_chunks = get_chunks(
        provider="Docling",
        splitter="HierarchicalChunker",
        text=parsed_text,
        output_format="html",
        merge_small_chunks=True,
    )

    # Verify we got multiple chunks
    assert len(ui_chunks) > 1, "Should have multiple chunks for structured HTML"

    # Get native DoclingDocument chunks for comparison
    from docling.document_converter import DocumentConverter
    from docling.chunking import HierarchicalChunker

    # Create temp file for Docling to parse
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".html", delete=False) as f:
        f.write(sample_html_document)
        temp_path = f.name

    try:
        # Parse with Docling
        converter = DocumentConverter()
        result = converter.convert(Path(temp_path))
        doc = result.document

        # Chunk with native Docling chunker
        chunker = HierarchicalChunker(merge_list_items=True)
        docling_chunks = list(chunker.chunk(doc))

        # Verify chunk counts match
        assert len(ui_chunks) == len(
            docling_chunks
        ), f"Chunk count mismatch: UI has {len(ui_chunks)}, Docling has {len(docling_chunks)}"

        # Verify chunk text matches (allowing for minor whitespace differences)
        for i, (ui_chunk, docling_chunk) in enumerate(zip(ui_chunks, docling_chunks)):
            ui_text = ui_chunk.text.strip()
            docling_text = docling_chunk.text.strip()

            # Compare chunk text directly
            # The text should match since HTML format uses native Docling chunking
            assert (
                ui_text == docling_text
            ), f"Chunk {i} text mismatch:\nUI: {ui_text[:200]}...\nDocling: {docling_text[:200]}..."

    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_markdown_chunks_use_raw_text_chunking(sample_markdown_document):
    """Test that markdown format uses raw text chunking (not native DoclingDocument chunks).

    This verifies markdown chunks are created from the exported text, which is expected
    and allows for markdown-specific features like heading detection.
    """
    # Parse document with markdown output format
    parsed_text, output_format, _ = parse_document(
        filename="sample_document.md",
        content=sample_markdown_document,
        params={
            "output_format": "markdown",
        },
    )

    assert output_format.lower() == "markdown"
    assert parsed_text is not None

    # Get chunks using HybridChunker with token limit to ensure multiple chunks
    # (HierarchicalChunker might merge everything into one chunk)
    ui_chunks = get_chunks(
        provider="Docling",
        splitter="HybridChunker",
        text=parsed_text,
        output_format="markdown",
        max_tokens=512,
        chunk_overlap=50,
    )

    # Verify we got multiple chunks due to token limit
    assert len(ui_chunks) > 1, f"Should have multiple chunks for structured markdown, got {len(ui_chunks)}"

    # Verify chunks contain markdown text (not HTML)
    for chunk in ui_chunks:
        chunk_text = chunk.text
        # Should have markdown headings or content
        # Not all chunks will have headings, but they should have markdown content
        assert (
            "#" in chunk_text
            or "RAG" in chunk_text
            or "retrieval" in chunk_text.lower()
            or "chunk" in chunk_text.lower()
        ), f"Chunk doesn't contain expected markdown content: {chunk_text[:100]}"

    # Verify heading metadata was extracted from raw text parsing
    has_heading_metadata = any(
        "section_hierarchy" in chunk.metadata or "heading_text" in chunk.metadata
        for chunk in ui_chunks
    )
    assert has_heading_metadata, "Markdown chunks should have heading metadata from raw text parsing"


def test_html_chunks_preserve_docling_metadata(sample_html_document):
    """Test that HTML chunks preserve DoclingDocument metadata."""
    # Parse document with HTML output format
    parsed_text, output_format, _ = parse_document(
        filename="test.html",
        content=sample_html_document,
        params={
            "output_format": "html",
            "docling_enable_ocr": False,
        },
    )

    # Get chunks
    ui_chunks = get_chunks(
        provider="Docling",
        splitter="HierarchicalChunker",
        text=parsed_text,
        output_format="html",
        merge_small_chunks=True,
    )

    # Verify metadata is present
    for chunk in ui_chunks:
        assert "strategy" in chunk.metadata
        # Strategy is stored as display name "Hierarchical", not "HierarchicalChunker"
        assert "Hierarchical" in chunk.metadata["strategy"]

        # Check for Docling-specific metadata
        # (element_type, section_hierarchy, etc. come from native Docling chunker)
        assert "chunk_index" in chunk.metadata
        assert "size" in chunk.metadata


def test_hybrid_chunker_html_vs_markdown():
    """Test that HybridChunker works correctly for both HTML and markdown."""
    # Create a longer document to test token-based chunking
    long_content = """# Introduction

""" + (
        "This is a sentence. " * 100
    ) + """

## Section 1

""" + (
        "Another sentence here. " * 100
    )

    long_html = f"""<!DOCTYPE html>
<html><body>
<h1>Introduction</h1>
<p>{'This is a sentence. ' * 100}</p>
<h2>Section 1</h2>
<p>{'Another sentence here. ' * 100}</p>
</body></html>"""

    # Test markdown
    md_chunks = get_chunks(
        provider="Docling",
        splitter="HybridChunker",
        text=long_content,
        output_format="markdown",
        max_tokens=256,
        chunk_overlap=50,
    )

    # Test HTML
    html_chunks = get_chunks(
        provider="Docling",
        splitter="HybridChunker",
        text=long_html,
        output_format="html",
        max_tokens=256,
        chunk_overlap=50,
    )

    # Both should create multiple chunks due to token limit
    assert len(md_chunks) > 1, "Markdown should be split into multiple chunks"
    assert len(html_chunks) > 1, "HTML should be split into multiple chunks"

    # Verify metadata (strategy is stored as display name "Hybrid")
    for chunk in md_chunks:
        assert "Hybrid" in chunk.metadata["strategy"]

    for chunk in html_chunks:
        assert "Hybrid" in chunk.metadata["strategy"]
