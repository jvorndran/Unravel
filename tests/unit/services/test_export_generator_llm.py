"""Unit tests for LiteLLM-based export code generation."""

from __future__ import annotations

import pytest

from unravel.services.export.generator import ExportConfig, generate_installation_command, generate_llm_code


@pytest.mark.unit
def test_generate_llm_code_uses_litellm_and_provider_prefix():
    cfg = ExportConfig(
        parsing_params={},
        chunking_params={},
        embedding_model="all-MiniLM-L6-v2",
        llm_config={
            "provider": "Anthropic",
            "model": "claude-opus-4-6",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    )
    code = generate_llm_code(cfg)
    assert code is not None
    assert "import litellm" in code
    assert 'model="anthropic/claude-opus-4-6"' in code
    assert "litellm.drop_params = True" in code


@pytest.mark.unit
def test_generate_llm_code_openai_compatible_includes_api_base():
    cfg = ExportConfig(
        parsing_params={},
        chunking_params={},
        embedding_model="all-MiniLM-L6-v2",
        llm_config={
            "provider": "OpenAI-Compatible",
            "model": "llama2",
            "base_url": "http://localhost:11434/v1",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    )
    code = generate_llm_code(cfg)
    assert code is not None
    assert 'api_base = "http://localhost:11434/v1"' in code
    assert 'model="openai/llama2"' in code
    assert "api_base=api_base" in code


@pytest.mark.unit
def test_generate_installation_command_includes_litellm_when_llm_config_present():
    cfg = ExportConfig(
        parsing_params={},
        chunking_params={},
        embedding_model="all-MiniLM-L6-v2",
        llm_config={"provider": "OpenAI", "model": "gpt-5-mini"},
    )
    cmd = generate_installation_command(cfg)
    assert "litellm" in cmd.split()

