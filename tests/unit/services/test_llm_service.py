"""Unit tests for the LiteLLM-based LLM service integration.

These tests ensure provider/model mapping, env key loading rules, config
validation, and request construction all behave deterministically without
calling any real provider APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pytest
from PIL import Image as PILImage

from unravel.services.llm import (
    LLMConfig,
    RAGContext,
    _litellm_kwargs,
    _litellm_model,
    generate_image_caption,
    generate_response,
    generate_response_stream,
    get_api_key_from_env,
    rewrite_query_variations,
    validate_config,
)


@dataclass
class _DummyMessage:
    content: str


@dataclass
class _DummyChoice:
    message: _DummyMessage


@dataclass
class _DummyDelta:
    content: str | None = None


@dataclass
class _DummyStreamChoice:
    delta: _DummyDelta


@dataclass
class _DummyResponse:
    choices: list[Any]


@pytest.mark.unit
def test_get_api_key_from_env_reads_only_unravel_dotenv(tmp_path, monkeypatch):
    """System environment should not affect API key status (dotenv-only)."""
    # Patch Path.home() as used in unravel.services.llm
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Even if set in system environment, key should be ignored when ~/.unravel/.env is missing
    monkeypatch.setenv("GEMINI_API_KEY", "system-key")
    assert get_api_key_from_env("Gemini") is None

    # Create ~/.unravel/.env and ensure value is read from there
    unravel_dir = tmp_path / ".unravel"
    unravel_dir.mkdir(parents=True, exist_ok=True)
    (unravel_dir / ".env").write_text("GEMINI_API_KEY=dotenv-key\n")

    assert get_api_key_from_env("Gemini") == "dotenv-key"


@pytest.mark.unit
def test_validate_config_requires_api_key_for_keyed_providers():
    ok, msg = validate_config(LLMConfig(provider="OpenAI", model="gpt-5-mini", api_key=""))
    assert ok is False
    assert "OPENAI_API_KEY" in msg

    ok, msg = validate_config(LLMConfig(provider="Anthropic", model="claude-opus-4-6", api_key=""))
    assert ok is False
    assert "ANTHROPIC_API_KEY" in msg

    ok, msg = validate_config(LLMConfig(provider="Gemini", model="gemini-2.5-flash", api_key=""))
    assert ok is False
    assert "GEMINI_API_KEY" in msg

    ok, msg = validate_config(LLMConfig(provider="OpenRouter", model="anthropic/claude-opus-4-6", api_key=""))
    assert ok is False
    assert "OPENROUTER_API_KEY" in msg


@pytest.mark.unit
def test_validate_config_requires_base_url_for_openai_compatible():
    ok, msg = validate_config(
        LLMConfig(provider="OpenAI-Compatible", model="llama2", api_key="not-needed", base_url=None)
    )
    assert ok is False
    assert "Base URL" in msg


@pytest.mark.unit
def test_litellm_model_prefixing():
    assert _litellm_model(LLMConfig(provider="OpenAI", model="gpt-5-mini", api_key="k")) == "gpt-5-mini"
    assert (
        _litellm_model(LLMConfig(provider="Anthropic", model="claude-opus-4-6", api_key="k"))
        == "anthropic/claude-opus-4-6"
    )
    assert (
        _litellm_model(LLMConfig(provider="Gemini", model="gemini-2.5-flash", api_key="k"))
        == "gemini/gemini-2.5-flash"
    )
    assert (
        _litellm_model(LLMConfig(provider="OpenRouter", model="anthropic/claude-opus-4-6", api_key="k"))
        == "openrouter/anthropic/claude-opus-4-6"
    )


@pytest.mark.unit
def test_litellm_kwargs_openai_compatible_sets_api_base_and_not_needed_key():
    config = LLMConfig(
        provider="OpenAI-Compatible",
        model="llama2",
        api_key="",
        base_url="http://localhost:11434/v1",
    )
    kwargs = _litellm_kwargs(config)
    assert kwargs["api_base"] == "http://localhost:11434/v1"
    assert kwargs["api_key"] == "not-needed"


@pytest.mark.unit
def test_generate_response_constructs_litellm_request(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> _DummyResponse:
        calls.append(kwargs)
        return _DummyResponse(choices=[_DummyChoice(message=_DummyMessage(content="ok"))])

    monkeypatch.setattr("unravel.services.llm.litellm.completion", fake_completion)

    config = LLMConfig(provider="Gemini", model="gemini-2.5-flash", api_key="gk")
    context = RAGContext(query="q", chunks=["c1"])
    result = generate_response(config, context)

    assert result == "ok"
    assert calls, "Expected litellm.completion to be called"
    assert calls[0]["model"] == "gemini/gemini-2.5-flash"
    assert calls[0]["api_key"] == "gk"
    assert "stream" not in calls[0]  # non-streaming calls omit stream=True


@pytest.mark.unit
def test_generate_response_stream_yields_chunks(monkeypatch):
    def fake_stream() -> Iterable[_DummyResponse]:
        yield _DummyResponse(choices=[_DummyStreamChoice(delta=_DummyDelta(content="a"))])
        yield _DummyResponse(choices=[_DummyStreamChoice(delta=_DummyDelta(content="b"))])
        yield _DummyResponse(choices=[_DummyStreamChoice(delta=_DummyDelta(content=None))])

    def fake_completion(**kwargs: Any) -> Any:
        assert kwargs["stream"] is True
        return fake_stream()

    monkeypatch.setattr("unravel.services.llm.litellm.completion", fake_completion)

    config = LLMConfig(provider="OpenAI", model="gpt-5-mini", api_key="ok")
    context = RAGContext(query="q", chunks=["c1"])

    out = "".join(list(generate_response_stream(config, context)))
    assert out == "ab"


@pytest.mark.unit
def test_rewrite_query_variations_parses_bullets_and_numbers(monkeypatch):
    def fake_completion(**kwargs: Any) -> _DummyResponse:
        text = "1) first rewrite\n- second rewrite\n\n3. third rewrite\n"
        return _DummyResponse(choices=[_DummyChoice(message=_DummyMessage(content=text))])

    monkeypatch.setattr("unravel.services.llm.litellm.completion", fake_completion)

    config = LLMConfig(provider="OpenAI", model="gpt-5-mini", api_key="ok")
    variations = rewrite_query_variations(config, query="x", count=3)
    assert variations == ["first rewrite", "second rewrite", "third rewrite"]


@pytest.mark.unit
def test_generate_image_caption_sends_openai_vision_message(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _DummyResponse:
        nonlocal seen
        seen = kwargs
        return _DummyResponse(choices=[_DummyChoice(message=_DummyMessage(content="caption"))])

    monkeypatch.setattr("unravel.services.llm.litellm.completion", fake_completion)

    img = PILImage.new("RGB", (2, 2), color=(255, 0, 0))
    config = LLMConfig(provider="OpenAI", model="gpt-4o-mini", api_key="ok")
    caption = generate_image_caption(img, config)

    assert caption == "caption"
    assert seen["model"] == "gpt-4o-mini"
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)
    image_part = next(part for part in content if part.get("type") == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

