"""Unit tests for Vertex AI integration.

These tests verify that the Vertex AI provider is properly configured
and integrated with the LLM service, including:
- Configuration validation
- Environment variable handling
- LiteLLM kwargs construction
- Model prefixing
- Vision capability detection
"""

from __future__ import annotations

from typing import Any

import pytest

from unravel.services.llm import (
    LLM_PROVIDERS,
    VISION_CAPABLE_MODELS,
    LLMConfig,
    _get_env_value,
    _litellm_kwargs,
    _litellm_model,
    get_api_key_from_env,
    get_provider_models,
    is_vision_capable,
    validate_config,
)


# ---------------------------------------------------------------------------
# Provider Configuration Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vertex_ai_provider_exists():
    """Vertex AI should be registered as a provider."""
    assert "Vertex AI" in LLM_PROVIDERS


@pytest.mark.unit
def test_vertex_ai_has_required_configuration():
    """Vertex AI provider should have all required configuration fields."""
    provider = LLM_PROVIDERS["Vertex AI"]

    assert "models" in provider
    assert isinstance(provider["models"], list)
    assert len(provider["models"]) > 0

    assert "default" in provider
    assert provider["default"] in provider["models"]

    assert "env_key" in provider
    assert provider["env_key"] == "VERTEXAI_PROJECT"

    assert "description" in provider
    assert "Vertex AI" in provider["description"]

    assert "requires_location" in provider
    assert provider["requires_location"] is True


@pytest.mark.unit
def test_vertex_ai_models_list():
    """Vertex AI should include expected Gemini models."""
    models = LLM_PROVIDERS["Vertex AI"]["models"]

    # Check for key models
    assert "gemini-2.5-pro" in models
    assert "gemini-1.5-pro" in models

    # Verify default model exists
    assert LLM_PROVIDERS["Vertex AI"]["default"] in models


@pytest.mark.unit
def test_vertex_ai_get_provider_models():
    """get_provider_models should return Vertex AI models."""
    models = get_provider_models("Vertex AI")

    assert isinstance(models, list)
    assert len(models) > 0
    assert "gemini-2.5-pro" in models


# ---------------------------------------------------------------------------
# Vision Capability Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vertex_ai_vision_capable_models_registered():
    """Vertex AI models should be registered as vision-capable."""
    assert "Vertex AI" in VISION_CAPABLE_MODELS
    assert isinstance(VISION_CAPABLE_MODELS["Vertex AI"], list)
    assert len(VISION_CAPABLE_MODELS["Vertex AI"]) > 0


@pytest.mark.unit
def test_vertex_ai_gemini_models_support_vision():
    """All Vertex AI Gemini models should support vision."""
    vision_models = VISION_CAPABLE_MODELS["Vertex AI"]

    # Key models should be vision-capable
    assert "gemini-2.5-pro" in vision_models
    assert "gemini-1.5-pro" in vision_models


@pytest.mark.unit
def test_is_vision_capable_vertex_ai():
    """is_vision_capable should correctly identify Vertex AI vision models."""
    assert is_vision_capable("Vertex AI", "gemini-2.5-pro") is True
    assert is_vision_capable("Vertex AI", "gemini-1.5-pro") is True
    assert is_vision_capable("Vertex AI", "gemini-2.5-flash-preview-09-2025") is True


# ---------------------------------------------------------------------------
# Environment Variable Handling Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_env_value_reads_system_environment(monkeypatch):
    """_get_env_value should read from system environment variables."""
    monkeypatch.setenv("VERTEXAI_PROJECT", "test-project-123")

    value = _get_env_value("VERTEXAI_PROJECT")
    assert value == "test-project-123"


@pytest.mark.unit
def test_get_env_value_reads_dotenv_file(tmp_path, monkeypatch):
    """_get_env_value should fall back to ~/.unravel/.env file."""
    # Patch Path.home() to use tmp_path
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Create ~/.unravel/.env with VERTEXAI_PROJECT
    unravel_dir = tmp_path / ".unravel"
    unravel_dir.mkdir(parents=True, exist_ok=True)
    (unravel_dir / ".env").write_text("VERTEXAI_PROJECT=dotenv-project-456\n")

    # Should read from .env file
    value = _get_env_value("VERTEXAI_PROJECT")
    assert value == "dotenv-project-456"


@pytest.mark.unit
def test_get_env_value_system_env_takes_precedence(tmp_path, monkeypatch):
    """System environment should take precedence over .env file."""
    # Patch Path.home()
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Create .env file
    unravel_dir = tmp_path / ".unravel"
    unravel_dir.mkdir(parents=True, exist_ok=True)
    (unravel_dir / ".env").write_text("VERTEXAI_PROJECT=dotenv-project\n")

    # Set system environment (should take precedence)
    monkeypatch.setenv("VERTEXAI_PROJECT", "system-project")

    value = _get_env_value("VERTEXAI_PROJECT")
    assert value == "system-project"


@pytest.mark.unit
def test_get_env_value_returns_default_when_not_found(tmp_path, monkeypatch):
    """_get_env_value should return default when variable not found."""
    # Patch Path.home() to empty temp directory
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path / "empty")

    # Clear any existing VERTEXAI_PROJECT
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)

    value = _get_env_value("VERTEXAI_PROJECT", default="default-project")
    assert value == "default-project"

    value = _get_env_value("VERTEXAI_PROJECT")
    assert value is None


@pytest.mark.unit
def test_get_env_value_handles_location_default(tmp_path, monkeypatch):
    """_get_env_value should handle VERTEXAI_LOCATION with default."""
    # Patch Path.home() to empty temp directory
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path / "empty")

    # Clear any existing VERTEXAI_LOCATION
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)

    value = _get_env_value("VERTEXAI_LOCATION", default="us-central1")
    assert value == "us-central1"


@pytest.mark.unit
def test_get_api_key_from_env_returns_project_for_vertex_ai(tmp_path, monkeypatch):
    """get_api_key_from_env should return project ID for Vertex AI provider."""
    # Patch Path.home()
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Create .env file with VERTEXAI_PROJECT
    unravel_dir = tmp_path / ".unravel"
    unravel_dir.mkdir(parents=True, exist_ok=True)
    (unravel_dir / ".env").write_text("VERTEXAI_PROJECT=my-gcp-project\n")

    # Should return project ID
    project = get_api_key_from_env("Vertex AI")
    assert project == "my-gcp-project"


# ---------------------------------------------------------------------------
# Configuration Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_config_requires_vertex_project(tmp_path, monkeypatch):
    """validate_config should require VERTEXAI_PROJECT for Vertex AI."""
    # Patch Path.home()
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Clear any system environment VERTEXAI_PROJECT
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)

    # No .env file, no project
    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",  # Vertex AI doesn't use api_key field
    )

    is_valid, error_msg = validate_config(config)
    assert is_valid is False
    assert "VERTEXAI_PROJECT" in error_msg
    assert ".env" in error_msg


@pytest.mark.unit
def test_validate_config_passes_with_vertex_project_in_system_env(monkeypatch):
    """validate_config should pass when VERTEXAI_PROJECT is in system env."""
    monkeypatch.setenv("VERTEXAI_PROJECT", "test-project")

    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    is_valid, error_msg = validate_config(config)
    assert is_valid is True
    assert error_msg == ""


@pytest.mark.unit
def test_validate_config_passes_with_vertex_project_in_dotenv(tmp_path, monkeypatch):
    """validate_config should pass when VERTEXAI_PROJECT is in .env file."""
    # Patch Path.home()
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Clear system env
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)

    # Create .env file
    unravel_dir = tmp_path / ".unravel"
    unravel_dir.mkdir(parents=True, exist_ok=True)
    (unravel_dir / ".env").write_text("VERTEXAI_PROJECT=my-project\n")

    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    is_valid, error_msg = validate_config(config)
    assert is_valid is True
    assert error_msg == ""


@pytest.mark.unit
def test_validate_config_does_not_require_api_key_for_vertex_ai(tmp_path, monkeypatch):
    """Vertex AI should not require api_key field (uses ADC instead)."""
    # Patch Path.home()
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Set project
    monkeypatch.setenv("VERTEXAI_PROJECT", "test-project")

    # Empty api_key should be OK for Vertex AI
    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",  # Empty is fine for Vertex AI
    )

    is_valid, error_msg = validate_config(config)
    assert is_valid is True


@pytest.mark.unit
def test_validate_config_requires_model_name():
    """All providers including Vertex AI should require a model name."""
    config = LLMConfig(
        provider="Vertex AI",
        model="",  # Empty model
        api_key="",
    )

    is_valid, error_msg = validate_config(config)
    assert is_valid is False
    assert "Model name is required" in error_msg


# ---------------------------------------------------------------------------
# LiteLLM Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_litellm_model_uses_vertex_ai_prefix():
    """Vertex AI models should be prefixed with vertex_ai/."""
    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    model_name = _litellm_model(config)
    assert model_name == "vertex_ai/gemini-2.5-pro"


@pytest.mark.unit
def test_litellm_kwargs_includes_vertex_project_and_location(monkeypatch):
    """_litellm_kwargs should include vertex_project and vertex_location."""
    monkeypatch.setenv("VERTEXAI_PROJECT", "my-gcp-project")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-west1")

    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    kwargs = _litellm_kwargs(config)

    assert "vertex_project" in kwargs
    assert kwargs["vertex_project"] == "my-gcp-project"

    assert "vertex_location" in kwargs
    assert kwargs["vertex_location"] == "us-west1"

    # Should NOT include api_key
    assert "api_key" not in kwargs


@pytest.mark.unit
def test_litellm_kwargs_reads_vertex_config_from_dotenv(tmp_path, monkeypatch):
    """_litellm_kwargs should read Vertex AI config from .env file."""
    # Patch Path.home()
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Clear system env
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)

    # Create .env file
    unravel_dir = tmp_path / ".unravel"
    unravel_dir.mkdir(parents=True, exist_ok=True)
    (unravel_dir / ".env").write_text(
        "VERTEXAI_PROJECT=dotenv-project\n"
        "VERTEXAI_LOCATION=europe-west1\n"
    )

    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    kwargs = _litellm_kwargs(config)

    assert kwargs["vertex_project"] == "dotenv-project"
    assert kwargs["vertex_location"] == "europe-west1"


@pytest.mark.unit
def test_litellm_kwargs_uses_default_location_when_not_set(monkeypatch):
    """_litellm_kwargs should use us-central1 as default location."""
    monkeypatch.setenv("VERTEXAI_PROJECT", "test-project")
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)

    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    kwargs = _litellm_kwargs(config)

    assert kwargs["vertex_project"] == "test-project"
    assert kwargs["vertex_location"] == "us-central1"  # Default


@pytest.mark.unit
def test_litellm_kwargs_only_adds_vertex_params_for_vertex_ai():
    """_litellm_kwargs should only add Vertex AI params for Vertex AI provider."""
    # Test with OpenAI (should not have vertex params)
    config = LLMConfig(
        provider="OpenAI",
        model="gpt-5-mini",
        api_key="test-key",
    )

    kwargs = _litellm_kwargs(config)

    assert "vertex_project" not in kwargs
    assert "vertex_location" not in kwargs
    assert kwargs["api_key"] == "test-key"


# ---------------------------------------------------------------------------
# Integration Boundary Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vertex_ai_config_distinct_from_gemini_provider():
    """Vertex AI should be a separate provider from Gemini (AI Studio)."""
    assert "Gemini" in LLM_PROVIDERS
    assert "Vertex AI" in LLM_PROVIDERS

    # They should have different env keys
    assert LLM_PROVIDERS["Gemini"]["env_key"] == "GEMINI_API_KEY"
    assert LLM_PROVIDERS["Vertex AI"]["env_key"] == "VERTEXAI_PROJECT"

    # They may have overlapping models, but different purposes
    assert LLM_PROVIDERS["Gemini"]["description"] != LLM_PROVIDERS["Vertex AI"]["description"]


@pytest.mark.unit
def test_vertex_ai_validation_does_not_break_other_providers(tmp_path, monkeypatch):
    """Vertex AI validation should not affect other provider validation."""
    # Patch Path.home()
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Test OpenAI validation still works
    config = LLMConfig(
        provider="OpenAI",
        model="gpt-5-mini",
        api_key="",  # Missing API key
    )

    is_valid, error_msg = validate_config(config)
    assert is_valid is False
    assert "OPENAI_API_KEY" in error_msg

    # Test with valid OpenAI config
    config.api_key = "test-key"
    is_valid, error_msg = validate_config(config)
    assert is_valid is True


@pytest.mark.unit
def test_vertex_ai_model_prefix_distinct_from_gemini():
    """Vertex AI and Gemini should use different model prefixes."""
    from unravel.services.llm import _PROVIDER_PREFIX

    assert _PROVIDER_PREFIX["Gemini"] == "gemini/"
    assert _PROVIDER_PREFIX["Vertex AI"] == "vertex_ai/"

    # Verify in practice
    gemini_config = LLMConfig(provider="Gemini", model="gemini-2.5-pro", api_key="k")
    vertex_config = LLMConfig(provider="Vertex AI", model="gemini-2.5-pro", api_key="")

    assert _litellm_model(gemini_config) == "gemini/gemini-2.5-pro"
    assert _litellm_model(vertex_config) == "vertex_ai/gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vertex_ai_handles_missing_env_file_gracefully(tmp_path, monkeypatch):
    """Should handle missing .env file without crashing."""
    # Patch Path.home() to non-existent directory
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path / "nonexistent")

    # Clear system env
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)

    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    # Should fail validation gracefully
    is_valid, error_msg = validate_config(config)
    assert is_valid is False
    assert "VERTEXAI_PROJECT" in error_msg


@pytest.mark.unit
def test_vertex_ai_handles_empty_project_value(tmp_path, monkeypatch):
    """Should reject empty VERTEXAI_PROJECT value."""
    # Patch Path.home()
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path)

    # Create .env with empty value
    unravel_dir = tmp_path / ".unravel"
    unravel_dir.mkdir(parents=True, exist_ok=True)
    (unravel_dir / ".env").write_text("VERTEXAI_PROJECT=\n")

    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    is_valid, error_msg = validate_config(config)
    assert is_valid is False


@pytest.mark.unit
def test_litellm_kwargs_skips_vertex_params_when_project_missing(tmp_path, monkeypatch):
    """_litellm_kwargs should skip vertex params if project is not set."""
    # Patch Path.home() to empty temp directory
    monkeypatch.setattr("unravel.services.llm.Path.home", lambda: tmp_path / "empty")

    # Clear all vertex env vars
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)

    config = LLMConfig(
        provider="Vertex AI",
        model="gemini-2.5-pro",
        api_key="",
    )

    kwargs = _litellm_kwargs(config)

    # Should not include vertex params if project is missing
    # (This allows validate_config to catch the error)
    assert "vertex_project" not in kwargs or kwargs["vertex_project"] is None
