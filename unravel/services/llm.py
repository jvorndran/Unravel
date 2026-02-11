"""
LLM service for RAG response generation.

Provides a unified interface for generating responses using various LLM providers.
"""

import base64
import os
from collections.abc import Generator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from PIL import Image as PILImage

# Lazy imports for LLM clients
_openai_client: Any | None = None
_anthropic_client: Any | None = None


# Available LLM providers and their models
LLM_PROVIDERS: dict[str, dict[str, Any]] = {
    "OpenAI": {
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "default": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI's GPT models",
    },
    "Anthropic": {
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
        ],
        "default": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic's Claude models",
    },
    "OpenAI-Compatible": {
        "models": [],  # User specifies model name
        "default": "",
        "env_key": "",
        "description": "Any OpenAI-compatible API (Ollama, LM Studio, etc.)",
        "requires_base_url": True,
    },
}

# Models that support vision/image input
VISION_CAPABLE_MODELS: dict[str, list[str]] = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "Anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "OpenAI-Compatible": [],  # Assume user knows if their model supports vision
}

DEFAULT_IMAGE_CAPTION_PROMPT = "Describe this image concisely in 1-2 sentences for document search indexing. Focus on the key visual content and any text visible in the image."

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on the provided context. "
    "Use the information from the context to answer the user's question accurately. "
    "If the context doesn't contain enough information to answer the question, "
    "say so clearly. Be concise but thorough in your response."
)

DEFAULT_QUERY_REWRITE_PROMPT = (
    "Generate {count} alternate phrasings of the user's question for search retrieval. "
    "Keep the meaning the same. Use varied wording and terminology. "
    "Return only the rewrites, one per line, with no numbering or bullets.\n\n"
    "Question: {query}"
)

DEFAULT_QUERY_REWRITE_SYSTEM_PROMPT = "You rewrite user questions into effective search queries."


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class RAGContext:
    """Context for RAG generation."""

    query: str
    chunks: list[str]
    scores: list[float] | None = None


def _get_openai_client(api_key: str, base_url: str | None = None) -> Any:
    """Get or create OpenAI client."""
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key, base_url=base_url)
    except ImportError as err:
        raise ImportError("OpenAI library not installed. Install with: pip install openai") from err


def _get_anthropic_client(api_key: str) -> Any:
    """Get or create Anthropic client."""
    try:
        from anthropic import Anthropic

        return Anthropic(api_key=api_key)
    except ImportError as err:
        raise ImportError(
            "Anthropic library not installed. Install with: pip install anthropic"
        ) from err


def _build_context_prompt(context: RAGContext) -> str:
    """Build the context section of the prompt from retrieved chunks."""
    if not context.chunks:
        return ""

    parts = []
    for i, chunk in enumerate(context.chunks):
        if context.scores:
            parts.append(f"[Chunk {i+1} (relevance: {context.scores[i]:.2f})]\n{chunk}")
        else:
            parts.append(f"[Chunk {i+1}]\n{chunk}")

    return "\n\n".join(parts)


def _build_user_prompt(context: RAGContext) -> str:
    """Build the complete user prompt with context and question."""
    context_text = _build_context_prompt(context)

    if context_text:
        return f"{context_text}\n\nQuestion: {context.query}"
    return f"Question: {context.query}"


def _parse_rewrite_variations(text: str, max_count: int) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        if line and line[0].isdigit():
            idx = 1
            while idx < len(line) and line[idx].isdigit():
                idx += 1
            if idx < len(line) and line[idx] in (".", ")"):
                line = line[idx + 1 :].strip()
        if line:
            lines.append(line)
        if len(lines) >= max_count:
            break
    return lines


def rewrite_query_variations(
    config: LLMConfig,
    query: str,
    count: int = 4,
    prompt: str = DEFAULT_QUERY_REWRITE_PROMPT,
    system_prompt: str = DEFAULT_QUERY_REWRITE_SYSTEM_PROMPT,
) -> list[str]:
    """Generate alternate phrasings of a query for multi-query retrieval."""
    rewrite_config = LLMConfig(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0.2,
        max_tokens=min(256, config.max_tokens),
    )
    user_prompt = prompt.format(query=query, count=count)

    if rewrite_config.provider in ("OpenAI", "OpenAI-Compatible"):
        response = _generate_openai(rewrite_config, system_prompt, user_prompt)
    elif rewrite_config.provider == "Anthropic":
        response = _generate_anthropic(rewrite_config, system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown provider: {rewrite_config.provider}")

    return _parse_rewrite_variations(response, count)


def generate_response(
    config: LLMConfig,
    context: RAGContext,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Generate a response using the configured LLM.

    Args:
        config: LLM configuration
        context: RAG context with query and chunks
        system_prompt: System prompt for the LLM

    Returns:
        Generated response text
    """
    user_prompt = _build_user_prompt(context)

    if config.provider == "OpenAI" or config.provider == "OpenAI-Compatible":
        return _generate_openai(config, system_prompt, user_prompt)
    elif config.provider == "Anthropic":
        return _generate_anthropic(config, system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def generate_response_stream(
    config: LLMConfig,
    context: RAGContext,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Generator[str, None, None]:
    """Generate a streaming response using the configured LLM.

    Args:
        config: LLM configuration
        context: RAG context with query and chunks
        system_prompt: System prompt for the LLM

    Yields:
        Response text chunks
    """
    user_prompt = _build_user_prompt(context)

    if config.provider == "OpenAI" or config.provider == "OpenAI-Compatible":
        yield from _generate_openai_stream(config, system_prompt, user_prompt)
    elif config.provider == "Anthropic":
        yield from _generate_anthropic_stream(config, system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _generate_openai(config: LLMConfig, system_prompt: str, user_prompt: str) -> str:
    """Generate response using OpenAI API."""
    client = _get_openai_client(config.api_key, config.base_url)

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    return cast(str, response.choices[0].message.content)


def _generate_openai_stream(
    config: LLMConfig, system_prompt: str, user_prompt: str
) -> Generator[str, None, None]:
    """Generate streaming response using OpenAI API."""
    client = _get_openai_client(config.api_key, config.base_url)

    stream = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield cast(str, content)


def _generate_anthropic(config: LLMConfig, system_prompt: str, user_prompt: str) -> str:
    """Generate response using Anthropic API."""
    client = _get_anthropic_client(config.api_key)

    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    return cast(str, response.content[0].text)


def _generate_anthropic_stream(
    config: LLMConfig, system_prompt: str, user_prompt: str
) -> Generator[str, None, None]:
    """Generate streaming response using Anthropic API."""
    client = _get_anthropic_client(config.api_key)

    with client.messages.stream(
        model=config.model,
        max_tokens=config.max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    ) as stream:
        yield from stream.text_stream


def get_api_key_from_env(provider: str) -> str | None:
    """Get API key from environment variable for a provider.

    Loads from ~/.unravel/.env file first, then falls back to system environment.

    Args:
        provider: LLM provider name

    Returns:
        API key if found, None otherwise
    """
    if provider not in LLM_PROVIDERS:
        return None

    env_key = LLM_PROVIDERS[provider].get("env_key", "")
    if not env_key:
        return None

    # Load from ~/.unravel/.env
    dotenv_path = Path.home() / ".unravel" / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)

    # Check environment (includes .env variables now)
    return os.environ.get(env_key)


def list_providers() -> list[dict[str, Any]]:
    """List available LLM providers with their details."""
    return [{"name": name, **info} for name, info in LLM_PROVIDERS.items()]


def get_provider_models(provider: str) -> list[str]:
    """Get available models for a provider."""
    if provider not in LLM_PROVIDERS:
        return []
    return cast(list[str], LLM_PROVIDERS[provider].get("models", []))


def validate_config(config: LLMConfig) -> tuple[bool, str]:
    """Validate LLM configuration.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not config.api_key:
        return False, "API key is required"

    if not config.model:
        return False, "Model name is required"

    if config.provider == "OpenAI-Compatible" and not config.base_url:
        return False, "Base URL is required for OpenAI-Compatible provider"

    return True, ""


class ModelWrapper:
    """Unified model interface wrapper (Vercel AI SDK style)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def stream(
        self, context: RAGContext, system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> Generator[str, None, None]:
        """Stream a response using the configured model.

        Args:
            context: RAG context with query and chunks
            system_prompt: System prompt for the LLM

        Yields:
            Response text chunks
        """
        yield from generate_response_stream(self.config, context, system_prompt)

    def generate(self, context: RAGContext, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Generate a complete response using the configured model.

        Args:
            context: RAG context with query and chunks
            system_prompt: System prompt for the LLM

        Returns:
            Generated response text
        """
        return generate_response(self.config, context, system_prompt)


def get_model(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> ModelWrapper:
    """Get a model instance with unified interface (Vercel AI SDK style).

    Args:
        provider: LLM provider name (OpenAI, Anthropic, OpenAI-Compatible)
        model: Model name/identifier
        api_key: API key for the provider
        base_url: Optional base URL (required for OpenAI-Compatible)
        temperature: Temperature setting (default: 0.7)
        max_tokens: Maximum tokens (default: 1024)

    Returns:
        ModelWrapper instance with stream() and generate() methods

    Example:
        >>> model = get_model("OpenAI", "gpt-4o-mini", api_key="sk-...")
        >>> for chunk in model.stream(context):
        ...     print(chunk, end="")
    """
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return ModelWrapper(config)


def is_vision_capable(provider: str, model: str) -> bool:
    """Check if a model supports vision/image input.

    Args:
        provider: LLM provider name
        model: Model name

    Returns:
        True if the model supports vision input
    """
    if provider == "OpenAI-Compatible":
        # For OpenAI-compatible, assume user knows what they're doing
        return True
    return model in VISION_CAPABLE_MODELS.get(provider, [])


def _resize_image_for_captioning(pil_image: PILImage.Image, max_size: int = 1024) -> PILImage.Image:
    """Resize large images to optimize API costs and latency.

    This is an optimization, not required by API limits. Both OpenAI and Anthropic
    can handle larger images, but smaller images reduce costs and improve latency.

    Args:
        pil_image: PIL image to resize
        max_size: Maximum dimension (width or height) in pixels

    Returns:
        Resized image (or original if already small enough)
    """
    if max(pil_image.size) > max_size:
        pil_image = pil_image.copy()
        pil_image.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
    return pil_image


def _pil_to_base64(pil_image: PILImage.Image) -> str:
    """Convert PIL image to base64 string.

    Args:
        pil_image: PIL image

    Returns:
        Base64 encoded PNG string
    """
    # Convert to RGB if necessary (handles RGBA, palette modes, etc.)
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")

    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _generate_image_caption_openai(img_base64: str, prompt: str, config: LLMConfig) -> str:
    """Generate image caption using OpenAI vision API.

    Uses standard OpenAI vision message format with text prompt and image.

    Args:
        img_base64: Base64 encoded PNG image
        prompt: Text prompt describing the task
        config: LLM configuration

    Returns:
        Generated caption text
    """
    client = _get_openai_client(config.api_key, config.base_url)

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ],
        max_tokens=150,
        temperature=0.3,
    )

    return cast(str, response.choices[0].message.content)


def _generate_image_caption_anthropic(img_base64: str, prompt: str, config: LLMConfig) -> str:
    """Generate image caption using Anthropic vision API.

    Uses standard Anthropic vision message format with image and text prompt.
    Note: Anthropic requires image content before text in the content array.

    Args:
        img_base64: Base64 encoded PNG image
        prompt: Text prompt describing the task
        config: LLM configuration

    Returns:
        Generated caption text
    """
    client = _get_anthropic_client(config.api_key)

    response = client.messages.create(
        model=config.model,
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return cast(str, response.content[0].text)


def generate_image_caption(
    pil_image: PILImage.Image,
    config: LLMConfig,
    prompt: str = DEFAULT_IMAGE_CAPTION_PROMPT,
) -> str:
    """Generate a caption for an image using a vision-capable LLM.

    Follows standard vision prompting patterns for OpenAI and Anthropic APIs.
    Images are optionally resized for cost/latency optimization.

    Args:
        pil_image: PIL image to caption
        config: LLM configuration (must be vision-capable model)
        prompt: Text prompt for caption generation

    Returns:
        Generated caption text

    Raises:
        ValueError: If provider doesn't support vision
    """
    # Optional resize for cost/latency optimization (not required by API limits)
    resized = _resize_image_for_captioning(pil_image)
    img_base64 = _pil_to_base64(resized)

    if config.provider in ("OpenAI", "OpenAI-Compatible"):
        return _generate_image_caption_openai(img_base64, prompt, config)
    elif config.provider == "Anthropic":
        return _generate_image_caption_anthropic(img_base64, prompt, config)
    else:
        raise ValueError(f"Provider {config.provider} does not support vision")
