"""Infrastructure tests to verify test setup is working correctly."""

import pytest


@pytest.mark.unit
def test_pytest_working():
    """Smoke test to verify pytest is configured correctly."""
    assert True


@pytest.mark.unit
def test_mock_storage_dir(mock_storage_dir):
    """Test that mock_storage_dir fixture creates expected structure."""
    assert mock_storage_dir.exists()
    assert (mock_storage_dir / "documents").exists()
    assert (mock_storage_dir / "config").exists()
    assert (mock_storage_dir / "session").exists()
    assert (mock_storage_dir / "documents" / "document_a.pdf").exists()


@pytest.mark.unit
def test_mock_qdrant_client(mock_qdrant_client):
    """Test that mock_qdrant_client fixture provides basic mocking."""
    client = mock_qdrant_client.return_value
    collections = client.get_collections()
    assert collections.collections == []


@pytest.mark.unit
def test_mock_sentence_transformer(mock_sentence_transformer):
    """Test that mock_sentence_transformer fixture provides embedding mocks."""
    model = mock_sentence_transformer.return_value
    embeddings = model.encode(["test sentence"])
    assert embeddings.shape == (10, 384)
    assert model.get_sentence_embedding_dimension() == 384


@pytest.mark.unit
def test_widget_constants_available():
    """Test that widget key constants are accessible."""
    from unravel.ui.constants import WidgetKeys

    # Test a few key constants
    assert hasattr(WidgetKeys, "UPLOAD_DELETE_BTN")
    assert hasattr(WidgetKeys, "CHUNKS_APPLY_BTN")
    assert hasattr(WidgetKeys, "EMBEDDINGS_QUERY_INPUT")
    assert hasattr(WidgetKeys, "QUERY_INPUT")
    assert hasattr(WidgetKeys, "EXPORT_GOTO_CHUNKS")
