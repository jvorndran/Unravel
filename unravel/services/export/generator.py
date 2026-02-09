"""Code generation logic for pipeline export."""

from dataclasses import dataclass
from typing import Any

from unravel.services.embedders import EMBEDDING_MODELS
from .templates import (
    BACKEND_DISPLAY_NAMES,
    CHUNKING_GENERIC,
    CHUNKING_TEMPLATES,
    EMBEDDING_ENCODE_QUERY,
    EMBEDDING_ENCODE_TEXTS,
    EMBEDDING_IMPORTS,
    EMBEDDING_MODEL_INIT,
    EMBEDDING_TEMPLATE,
    LLM_TEMPLATE,
    PARSING_DOCX,
    PARSING_TEMPLATES,
    PARSING_TEXT,
    RERANKING_TEMPLATE,
    RETRIEVAL_TEMPLATES,
)

# Dependency mappings
PARSER_DEPENDENCIES = {
    "docling": ["docling"],
}

SPLITTER_DEPENDENCIES = {
    # Docling chunkers (primary)
    "HierarchicalChunker": ["docling"],
    "HybridChunker": ["docling", "tiktoken"],
    # Legacy LangChain splitters (for backwards compatibility)
    "RecursiveCharacterTextSplitter": ["langchain-text-splitters"],
    "CharacterTextSplitter": ["langchain-text-splitters"],
    "TokenTextSplitter": ["langchain-text-splitters", "tiktoken"],
    "MarkdownTextSplitter": ["langchain-text-splitters"],
    "LatexTextSplitter": ["langchain-text-splitters"],
    "PythonCodeTextSplitter": ["langchain-text-splitters"],
    "HTMLHeaderTextSplitter": ["langchain-text-splitters"],
    "RecursiveJsonSplitter": ["langchain-text-splitters"],
    "SentenceTransformersTokenTextSplitter": [
        "langchain-text-splitters",
        "sentence-transformers",
    ],
    "NLTKTextSplitter": ["langchain-text-splitters", "nltk"],
    "SpacyTextSplitter": ["langchain-text-splitters", "spacy"],
}

EMBEDDING_DEPENDENCIES = {
    "sentence-transformers": ["sentence-transformers", "numpy"],
    "flagembedding": ["FlagEmbedding", "numpy"],
}


@dataclass
class ExportConfig:
    """Configuration for code export."""
    parsing_params: dict[str, Any]
    chunking_params: dict[str, Any]
    embedding_model: str
    file_format: str | None = None  # PDF, DOCX, Markdown, Text
    retrieval_strategy: str | None = None
    retrieval_params: dict[str, Any] | None = None
    reranking_config: dict[str, Any] | None = None
    llm_config: dict[str, Any] | None = None




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
        )
    elif file_format in ["Text", "Markdown"]:
        return PARSING_TEXT.format()

    # PDF parsing with Docling
    template = PARSING_TEMPLATES["docling"]
    return template.format(
        enable_ocr=params.get("docling_enable_ocr", False),
        enable_tables=params.get("docling_table_structure", True),
        num_threads=params.get("docling_threads", 4),
    )


def generate_chunking_code(config: ExportConfig) -> str:
    """Generate text chunking code based on configuration."""
    params = config.chunking_params
    splitter = params.get("splitter", "HybridChunker")

    template = CHUNKING_TEMPLATES.get(splitter)

    if template is None:
        # Use generic template for unknown splitters
        return _generate_generic_chunking(splitter, params)

    # Build format kwargs based on splitter type
    format_kwargs = {}

    # Docling chunkers
    if splitter == "HierarchicalChunker":
        format_kwargs = {
            "include_headers": params.get("include_headers", True),
            "merge_small_chunks": params.get("merge_small_chunks", True),
            "min_chunk_size": params.get("min_chunk_size", 50),
        }
    elif splitter == "HybridChunker":
        format_kwargs = {
            "max_tokens": params.get("max_tokens", 512),
            "chunk_overlap": params.get("chunk_overlap", 50),
            "tokenizer": params.get("tokenizer", "cl100k_base"),
        }
    # Legacy LangChain splitters
    elif splitter == "RecursiveCharacterTextSplitter":
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
    """Generate embedding code based on model backend.

    Args:
        config: Export configuration with embedding_model

    Returns:
        Python code string for embedding generation
    """
    model_name = config.embedding_model

    # Get model info from registry
    if model_name in EMBEDDING_MODELS:
        model_info = EMBEDDING_MODELS[model_name]
        backend = model_info["backend"]
        dimension = model_info["dimension"]
        description = model_info["description"]
    else:
        # Fallback for unknown models
        backend = "sentence-transformers"
        dimension = 768
        description = "Custom embedding model"

    # Get backend-specific code snippets
    import_statement = EMBEDDING_IMPORTS[backend]
    model_init = EMBEDDING_MODEL_INIT[backend].format(model_name=model_name)
    encode_texts_code = EMBEDDING_ENCODE_TEXTS[backend]
    encode_query_code = EMBEDDING_ENCODE_QUERY[backend]
    backend_display = BACKEND_DISPLAY_NAMES[backend]

    # Format template
    code = EMBEDDING_TEMPLATE.format(
        backend=backend,
        backend_display=backend_display,
        model_name=model_name,
        dimension=dimension,
        description=description,
        import_statement=import_statement,
        model_init=model_init,
        encode_texts_code=encode_texts_code,
        encode_query_code=encode_query_code,
    )

    return code


def generate_retrieval_code(config: ExportConfig) -> str | None:
    """Generate retrieval code based on configuration."""
    if not config.retrieval_strategy:
        return None

    template = RETRIEVAL_TEMPLATES.get(config.retrieval_strategy)
    if not template:
        return None

    params = config.retrieval_params or {}
    top_k = params.get("top_k", 5)

    if config.retrieval_strategy == "HybridRetriever":
        return template.format(
            top_k=top_k,
            dense_weight=params.get("dense_weight", 0.5),
            sparse_weight=params.get("sparse_weight", 0.5),
            fusion_method=params.get("fusion_method", "weighted_sum"),
        )
    else:
        return template.format(top_k=top_k)


def generate_reranking_code(config: ExportConfig) -> str | None:
    """Generate reranking code based on configuration."""
    if not config.reranking_config:
        return None

    rerank_config = config.reranking_config
    if not rerank_config.get("enabled"):
        return None

    model_name = rerank_config.get("model")
    top_n = rerank_config.get("top_n", 5)

    if not model_name:
        return None

    # Look up backend from model registry
    from unravel.services.retrieval.reranking import RERANKER_MODELS

    if model_name not in RERANKER_MODELS:
        return None

    backend = RERANKER_MODELS[model_name]["backend"]
    library_display = RERANKER_MODELS[model_name]["library"]

    # Backend-specific configurations
    if backend == "flashrank":
        import_statement = "from flashrank import Ranker, RerankRequest"
        init_code = f'# Initialize FlashRank model\nranker = Ranker(model_name="{model_name}", cache_dir="./models")'
        rerank_code = '''    # Prepare passages for reranking
    passages = [{"text": chunk} for chunk in chunks]

    # Create rerank request
    rerank_request = RerankRequest(query=query, passages=passages)

    # Rerank
    results = ranker.rerank(rerank_request)

    # Sort by score and return top_n
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
    return [(r["text"], r["score"]) for r in ranked]'''
        description = "Fast CPU-based reranking using FlashRank"

    elif backend == "sentence-transformers":
        import_statement = "from sentence_transformers import CrossEncoder"
        init_code = f'# Initialize cross-encoder model\ncross_encoder = CrossEncoder("{model_name}")'
        rerank_code = '''    # Create query-chunk pairs
    pairs = [[query, chunk] for chunk in chunks]

    # Get relevance scores
    scores = cross_encoder.predict(pairs)

    # Combine chunks with scores and sort
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top_n
    return [(chunk, float(score)) for chunk, score in scored_chunks[:top_n]]'''
        description = "High-quality reranking using Sentence-Transformers cross-encoder"

    elif backend == "flagembedding":
        import_statement = "from FlagEmbedding import FlagReranker"
        init_code = f'# Initialize FlagEmbedding reranker\nreranker = FlagReranker("{model_name}", use_fp16=True)'
        rerank_code = '''    # Create query-chunk pairs
    pairs = [[query, chunk] for chunk in chunks]

    # Get relevance scores
    scores = reranker.compute_score(pairs)

    # Handle single score vs batch scores
    if isinstance(scores, (int, float)):
        scores = [scores]

    # Combine chunks with scores and sort
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top_n
    return [(chunk, float(score)) for chunk, score in scored_chunks[:top_n]]'''
        description = "State-of-the-art reranking using FlagEmbedding"

    else:
        return None

    return RERANKING_TEMPLATE.format(
        model_name=model_name,
        library=library_display,
        description=description,
        top_n=top_n,
        import_statement=import_statement,
        init_code=init_code,
        rerank_code=rerank_code,
    )


def generate_llm_code(config: ExportConfig) -> str | None:
    """Generate LLM integration code based on configuration."""
    if not config.llm_config:
        return None

    llm_config = config.llm_config
    provider = llm_config.get("provider")
    model = llm_config.get("model")
    temperature = llm_config.get("temperature", 0.7)
    max_tokens = llm_config.get("max_tokens", 1024)
    base_url = llm_config.get("base_url")
    system_prompt = llm_config.get("system_prompt", "You are a helpful assistant. Answer questions based on the provided context.")

    if not provider or not model:
        return None

    # Provider-specific configurations
    if provider == "OpenAI":
        import_statement = "from openai import OpenAI"
        client_init = "# Initialize OpenAI client\nclient = OpenAI()"
        generation_code = '''    # Generate response
    response = client.chat.completions.create(
        model="{model}",
        messages=[
            {{"role": "system", "content": system_prompt}},
            {{"role": "user", "content": user_message}},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content'''.format(model=model)
        base_url_display = ""

    elif provider == "Anthropic":
        import_statement = "from anthropic import Anthropic"
        client_init = "# Initialize Anthropic client\nclient = Anthropic()"
        generation_code = '''    # Generate response
    response = client.messages.create(
        model="{model}",
        system=system_prompt,
        messages=[
            {{"role": "user", "content": user_message}},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.content[0].text'''.format(model=model)
        base_url_display = ""

    elif provider == "OpenAI-Compatible":
        import_statement = "from openai import OpenAI"
        base_url_str = f'base_url="{base_url}"' if base_url else ""
        client_init = f"# Initialize OpenAI-compatible client\nclient = OpenAI({base_url_str})"
        generation_code = '''    # Generate response
    response = client.chat.completions.create(
        model="{model}",
        messages=[
            {{"role": "system", "content": system_prompt}},
            {{"role": "user", "content": user_message}},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content'''.format(model=model)
        base_url_display = f"- Base URL: {base_url}" if base_url else ""

    else:
        return None

    return LLM_TEMPLATE.format(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url_display=base_url_display,
        system_prompt=system_prompt,
        import_statement=import_statement,
        client_init=client_init,
        generation_code=generation_code,
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
    splitter = config.chunking_params.get("splitter", "HybridChunker")
    splitter_deps = SPLITTER_DEPENDENCIES.get(splitter, [])
    deps.update(splitter_deps)

    # Embedding dependencies (backend-specific)
    if config.embedding_model in EMBEDDING_MODELS:
        backend = EMBEDDING_MODELS[config.embedding_model]["backend"]
        embedding_deps = EMBEDDING_DEPENDENCIES.get(
            backend, ["sentence-transformers", "numpy"]
        )
        deps.update(embedding_deps)
    else:
        deps.update(["sentence-transformers", "numpy"])

    # Retrieval dependencies
    if config.retrieval_strategy in ["SparseRetriever", "HybridRetriever"]:
        deps.add("rank-bm25")

    # Reranking dependencies
    if config.reranking_config and config.reranking_config.get("enabled"):
        model_name = config.reranking_config.get("model")
        if model_name:
            from unravel.services.retrieval.reranking import RERANKER_MODELS

            if model_name in RERANKER_MODELS:
                backend = RERANKER_MODELS[model_name]["backend"]
                if backend == "flashrank":
                    deps.add("flashrank")
                elif backend == "sentence-transformers":
                    deps.add("sentence-transformers")
                elif backend == "flagembedding":
                    deps.add("FlagEmbedding")

    # LLM dependencies
    if config.llm_config:
        provider = config.llm_config.get("provider")
        if provider == "OpenAI" or provider == "OpenAI-Compatible":
            deps.add("openai")
        elif provider == "Anthropic":
            deps.add("anthropic")

    # Sort for consistent output
    sorted_deps = sorted(deps)

    return f"pip install {' '.join(sorted_deps)}"


def get_config_summary(config: ExportConfig) -> dict[str, str]:
    """Get a summary of the configuration for display."""
    splitter = config.chunking_params.get("splitter", "HybridChunker")
    # Use max_tokens for Docling chunkers, chunk_size for legacy
    if splitter in ["HierarchicalChunker", "HybridChunker"]:
        chunk_size = config.chunking_params.get("max_tokens", 512)
        chunk_size_label = "max_tokens"
    else:
        chunk_size = config.chunking_params.get("chunk_size", 500)
        chunk_size_label = "chunk_size"
    chunk_overlap = config.chunking_params.get("chunk_overlap", 50)

    return {
        "parser": "docling",
        "splitter": splitter,
        "chunk_size": str(chunk_size),
        "chunk_size_key": chunk_size_label,
        "chunk_overlap": str(chunk_overlap),
        "embedding_model": config.embedding_model,
        "file_format": config.file_format or "PDF",
    }
