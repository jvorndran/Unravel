"""Code generation logic for pipeline export."""

from dataclasses import dataclass
from typing import Any

from .templates import (
    CHUNKING_GENERIC,
    CHUNKING_TEMPLATES,
    EMBEDDING_TEMPLATE,
    PARSING_DOCX,
    PARSING_TEMPLATES,
    PARSING_TEXT,
)

# Embedding model info (mirror of embedders.py)
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "Fast, lightweight model (22M params). Good for experimentation.",
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "description": "Higher quality embeddings (110M params). Best overall quality.",
    },
    "paraphrase-MiniLM-L3-v2": {
        "dimension": 384,
        "description": "Very fast, smallest model (17M params). Quick prototyping.",
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "dimension": 384,
        "description": "Optimized for semantic search and QA retrieval.",
    },
}

# Dependency mappings
PARSER_DEPENDENCIES = {
    "docling": ["docling"],
}

SPLITTER_DEPENDENCIES = {
    "RecursiveCharacterTextSplitter": ["langchain-text-splitters"],
    "CharacterTextSplitter": ["langchain-text-splitters"],
    "TokenTextSplitter": ["langchain-text-splitters", "tiktoken"],
    "MarkdownTextSplitter": ["langchain-text-splitters"],
    "LatexTextSplitter": ["langchain-text-splitters"],
    "PythonCodeTextSplitter": ["langchain-text-splitters"],
    "HTMLHeaderTextSplitter": ["langchain-text-splitters"],
    "RecursiveJsonSplitter": ["langchain-text-splitters"],
    "SentenceTransformersTokenTextSplitter": ["langchain-text-splitters", "sentence-transformers"],
    "NLTKTextSplitter": ["langchain-text-splitters", "nltk"],
    "SpacyTextSplitter": ["langchain-text-splitters", "spacy"],
}

EMBEDDING_DEPENDENCIES = ["sentence-transformers", "numpy"]


@dataclass
class ExportConfig:
    """Configuration for code export."""
    parsing_params: dict[str, Any]
    chunking_params: dict[str, Any]
    embedding_model: str
    file_format: str | None = None  # PDF, DOCX, Markdown, Text


def _build_post_processing(params: dict[str, Any]) -> str:
    """Build post-processing code block."""
    lines = []

    if params.get("normalize_whitespace", False):
        lines.append("    # Normalize whitespace")
        lines.append("    text = re.sub(r' +', ' ', text)")
        lines.append("    text = re.sub(r'\\\\n\\\\n+', '\\\\n\\\\n', text)")
        lines.append("    lines = [line.strip() for line in text.split('\\\\n')]")
        lines.append("    text = '\\\\n'.join(lines)")

    if params.get("remove_special_chars", False):
        lines.append("    # Remove special characters")
        lines.append("    text = re.sub(r'[^\\\\w\\\\s.,-!?:;\\\\n]', '', text)")

    if lines:
        return "\n" + "\n".join(lines)
    return ""


def _get_separator(params: dict[str, Any]) -> str:
    """Get the appropriate page separator."""
    output_format = params.get("output_format", "original")
    if output_format == "markdown":
        return "\\n\\n---\\n\\n"
    return "\\n\\n"


def generate_parsing_code(config: ExportConfig) -> str:
    """Generate document parsing code based on configuration."""
    params = config.parsing_params
    file_format = config.file_format

    # Handle non-PDF formats
    if file_format == "DOCX":
        return PARSING_DOCX.format(
            output_format=params.get("output_format", "markdown"),
            normalize_whitespace=params.get("normalize_whitespace", False),
            remove_special_chars=params.get("remove_special_chars", False),
            post_processing=_build_post_processing(params),
        )
    elif file_format in ["Text", "Markdown"]:
        return PARSING_TEXT.format(
            normalize_whitespace=params.get("normalize_whitespace", False),
            remove_special_chars=params.get("remove_special_chars", False),
            post_processing=_build_post_processing(params),
        )

    # PDF parsing with Docling
    template = PARSING_TEMPLATES["docling"]
    return template.format(
        enable_ocr=params.get("docling_enable_ocr", False),
        enable_tables=params.get("docling_table_structure", True),
        num_threads=params.get("docling_threads", 4),
        post_processing=_build_post_processing(params),
    )


def generate_chunking_code(config: ExportConfig) -> str:
    """Generate text chunking code based on configuration."""
    params = config.chunking_params
    splitter = params.get("splitter", "RecursiveCharacterTextSplitter")

    template = CHUNKING_TEMPLATES.get(splitter)

    if template is None:
        # Use generic template for unknown splitters
        return _generate_generic_chunking(splitter, params)

    # Build format kwargs based on splitter type
    format_kwargs = {}

    if splitter == "RecursiveCharacterTextSplitter":
        format_kwargs = {
            "chunk_size": params.get("chunk_size", 500),
            "chunk_overlap": params.get("chunk_overlap", 50),
        }
    elif splitter == "CharacterTextSplitter":
        separator = params.get("separator", "\\n\\n")
        format_kwargs = {
            "chunk_size": params.get("chunk_size", 500),
            "chunk_overlap": params.get("chunk_overlap", 50),
            "separator": repr(separator),
            "separator_display": repr(separator),
        }
    elif splitter == "TokenTextSplitter":
        format_kwargs = {
            "chunk_size": params.get("chunk_size", 500),
            "chunk_overlap": params.get("chunk_overlap", 50),
            "encoding_name": params.get("encoding_name", "cl100k_base"),
        }
    elif splitter in ["MarkdownTextSplitter", "LatexTextSplitter", "PythonCodeTextSplitter"]:
        format_kwargs = {
            "chunk_size": params.get("chunk_size", 500),
            "chunk_overlap": params.get("chunk_overlap", 50),
        }
    elif splitter == "RecursiveJsonSplitter":
        format_kwargs = {
            "max_chunk_size": params.get("max_chunk_size", 500),
            "min_chunk_size": params.get("min_chunk_size", 100),
        }
    elif splitter == "SentenceTransformersTokenTextSplitter":
        format_kwargs = {
            "tokens_per_chunk": params.get("tokens_per_chunk", 256),
            "chunk_overlap": params.get("chunk_overlap", 50),
            "model_name": params.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        }
    elif splitter == "NLTKTextSplitter":
        format_kwargs = {
            "chunk_size": params.get("chunk_size", 500),
            "chunk_overlap": params.get("chunk_overlap", 50),
        }
    elif splitter == "SpacyTextSplitter":
        format_kwargs = {
            "chunk_size": params.get("chunk_size", 500),
            "chunk_overlap": params.get("chunk_overlap", 50),
            "pipeline": params.get("pipeline", "en_core_web_sm"),
        }
    elif splitter == "HTMLHeaderTextSplitter":
        # No parameters for HTML header splitter
        format_kwargs = {}

    return template.format(**format_kwargs)


def _generate_generic_chunking(splitter_name: str, params: dict[str, Any]) -> str:
    """Generate generic chunking code for unknown splitters."""
    # Build config comment
    config_lines = []
    params_lines = []

    for key, value in params.items():
        if key not in ["provider", "splitter"]:
            config_lines.append(f"- {key}: {value}")
            if isinstance(value, str):
                params_lines.append(f"        {key}={repr(value)},")
            else:
                params_lines.append(f"        {key}={value},")

    return CHUNKING_GENERIC.format(
        splitter_name=splitter_name,
        config_comment="\n".join(config_lines) if config_lines else "Default settings",
        params_code="\n".join(params_lines) if params_lines else "        # Default parameters",
    )


def generate_embedding_code(config: ExportConfig) -> str:
    """Generate embedding generation code."""
    model_name = config.embedding_model
    model_info = EMBEDDING_MODELS.get(model_name, {
        "dimension": 384,
        "description": "Custom embedding model",
    })

    return EMBEDDING_TEMPLATE.format(
        model_name=model_name,
        dimension=model_info["dimension"],
        description=model_info["description"],
    )


def generate_installation_command(config: ExportConfig) -> str:
    """Generate pip install command with required dependencies."""
    deps = set()

    # Parser dependencies
    file_format = config.file_format

    if file_format == "DOCX":
        deps.add("python-docx")
    elif file_format in ["PDF", None]:
        deps.update(PARSER_DEPENDENCIES["docling"])

    # Splitter dependencies
    splitter = config.chunking_params.get("splitter", "RecursiveCharacterTextSplitter")
    splitter_deps = SPLITTER_DEPENDENCIES.get(splitter, ["langchain-text-splitters"])
    deps.update(splitter_deps)

    # Embedding dependencies
    deps.update(EMBEDDING_DEPENDENCIES)

    # Sort for consistent output
    sorted_deps = sorted(deps)

    return f"pip install {' '.join(sorted_deps)}"


def get_config_summary(config: ExportConfig) -> dict[str, str]:
    """Get a summary of the configuration for display."""
    splitter = config.chunking_params.get("splitter", "RecursiveCharacterTextSplitter")
    chunk_size = config.chunking_params.get("chunk_size", 500)
    chunk_overlap = config.chunking_params.get("chunk_overlap", 50)

    return {
        "parser": "docling",
        "splitter": splitter,
        "chunk_size": str(chunk_size),
        "chunk_overlap": str(chunk_overlap),
        "embedding_model": config.embedding_model,
        "file_format": config.file_format or "PDF",
    }
