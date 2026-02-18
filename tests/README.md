# Unravel Test Suite

Comprehensive test coverage for the RAG Visualizer application.

## Structure

```
tests/
├── conftest.py                        # Shared fixtures and test helpers
├── test_infrastructure.py             # Infrastructure/setup tests
├── API_ENDPOINT_TESTS.md              # API endpoint test documentation
├── unit/                              # Unit tests for service layer
│   ├── services/
│   │   ├── test_api_server.py        # API server unit tests
│   │   ├── test_chunking_service.py
│   │   ├── test_embedders.py
│   │   ├── test_llm_service.py
│   │   └── ...
│   └── utils/
│       └── test_server_manager.py    # Server manager unit tests
├── ui/                                # UI component tests
│   └── steps/
│       ├── test_chunks_step.py
│       ├── test_export_step.py
│       └── test_upload_step.py
└── integration/                       # Cross-layer integration tests
    └── test_api_endpoint.py          # API endpoint integration tests
```

## Running Tests

### All tests
```bash
pytest
```

### By marker
```bash
pytest -m unit              # Fast unit tests only
pytest -m ui                # UI tests only
pytest -m integration       # Integration tests only
pytest -m slow              # Long-running tests
pytest -m requires_docker   # Tests requiring Docker/Qdrant
```

### With coverage
```bash
pytest --cov=unravel --cov-report=html
open htmlcov/index.html
```

### Specific test file
```bash
pytest tests/test_infrastructure.py -v
```


## Test Markers

- `@pytest.mark.unit` - Pure unit tests (fast, no external dependencies)
- `@pytest.mark.ui` - Streamlit UI tests (uses AppTest framework)
- `@pytest.mark.integration` - Cross-layer integration tests
- `@pytest.mark.slow` - Tests that take >1 second
- `@pytest.mark.requires_docker` - Tests requiring Docker/Qdrant

## Fixtures

### Storage Fixtures
- `mock_storage_dir` - Temporary storage directory with test documents
- `patched_storage` - Context manager to patch storage directory

### Mock Fixtures
- `mock_qdrant_client` - Mock Qdrant vector database client
- `mock_sentence_transformer` - Mock SentenceTransformer for embeddings

### UI Fixtures
- `sidebar_app_script` - Streamlit app script for sidebar testing
- `chunking_config_app_script` - App script for chunking config testing

### Utilities
- `element_exists(at, element_type, key)` - Check if widget exists
- `get_form_submit_button(at, label)` - Find form submit button
- `clear_streamlit_cache` - Auto-clears Streamlit caches (autouse)

## Widget Keys

All widget keys are centralized in `unravel/ui/constants.py` to ensure consistency between implementation and tests. Use `WidgetKeys.*` constants in tests rather than hardcoding strings.

Example:
```python
from unravel.ui.constants import WidgetKeys

def test_upload_delete_button(at):
    at.button(key=WidgetKeys.UPLOAD_DELETE_BTN).click()
```

## Writing Tests

### UI Tests (Streamlit AppTest)

```python
import pytest
from streamlit.testing.v1 import AppTest
from unravel.ui.constants import WidgetKeys

@pytest.mark.ui
def test_chunks_apply_button():
    at = AppTest.from_file("unravel/ui/steps/chunks.py")
    at.run()

    # Check button exists
    assert at.button(key=WidgetKeys.CHUNKS_APPLY_BTN).disabled

    # Simulate user interaction
    at.session_state.chunking_params["max_tokens"] = 1024
    at.run()

    # Verify button enabled after changes
    assert not at.button(key=WidgetKeys.CHUNKS_APPLY_BTN).disabled
```

### Unit Tests (Service Layer)

```python
import pytest

@pytest.mark.unit
def test_chunking_service(mock_storage_dir):
    from unravel.services.chunking import get_chunks

    chunks = get_chunks(
        provider="Docling",
        splitter="HybridChunker",
        text="Test document content",
        max_tokens=512,
    )

    assert len(chunks) > 0
    assert all(hasattr(c, "text") for c in chunks)
```

### Integration Tests

```python
import pytest

@pytest.mark.integration
def test_full_rag_pipeline(mock_storage_dir, mock_qdrant_client):
    # Test upload -> parse -> chunk -> embed -> query flow
    pass
```

## Test Suites

### API Endpoint Tests
Comprehensive tests for the API endpoint feature that exposes the RAG pipeline as a REST API.

**Files:**
- `unit/services/test_api_server.py` - FastAPI server unit tests
- `unit/utils/test_server_manager.py` - Server lifecycle management tests
- `integration/test_api_endpoint.py` - End-to-end API tests

**Quick Start:**
```bash
# Run all API endpoint tests
bash scripts/run_api_tests.sh --verbose

# Run with coverage
bash scripts/run_api_tests.sh --coverage
```

**Documentation:** [API_ENDPOINT_TESTS.md](./API_ENDPOINT_TESTS.md)

### Other Test Suites
- UI component tests (`ui/steps/`)
- Service layer tests (`unit/services/`)
- Infrastructure tests (`test_infrastructure.py`)

## Coverage Goals

- Service layer: >= 80%
- UI steps: >= 70%
- Overall: >= 75%
- API endpoint: >= 85% ✅
