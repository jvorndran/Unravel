# Service Layer Unit Tests

Comprehensive unit tests for the Unravel service layer, covering core business logic independent of the UI.

## Test Structure

### Files Created

1. **test_chunking_service.py** (436 lines, 30 tests + 1 xfail)
   - Provider registration and retrieval
   - Hierarchical chunking strategy
   - Hybrid (token-aware) chunking strategy
   - Edge cases (empty text, Unicode, large documents)
   - Metadata handling and options
   - Position tracking (start_index, end_index)
   - Error handling for invalid inputs

2. **test_embedders.py** (416 lines, 31 tests)
   - Embedder initialization and configuration
   - Backend detection (sentence-transformers, FlagEmbedding)
   - Dimension validation and handling
   - Lazy model loading
   - Batch embedding generation
   - Query embedding
   - Normalization verification
   - Model registry validation
   - Edge cases (empty lists, Unicode, very long text)

3. **test_vector_store.py** (564 lines, 27 tests)
   - Vector store initialization with different metrics
   - Adding embeddings (single, multiple, with/without metadata)
   - Vector similarity search
   - Save/load operations
   - Payload caching
   - Size tracking
   - Clear operations
   - Edge cases (1D arrays, k exceeds size, dimension mismatches)

4. **test_storage.py** (584 lines, 43 tests)
   - Directory management and creation
   - Document save/load/delete operations
   - File sanitization (path traversal prevention)
   - JSON save/load operations
   - Session state persistence
   - LLM configuration persistence
   - RAG configuration persistence
   - BM25 index persistence
   - Storage statistics
   - Edge cases (Unicode filenames, empty data, corrupted JSON)

## Test Results

- **Total Tests**: 132
- **Passing**: 131 (99.2%)
- **Expected Failures**: 1 (documented edge case in docling provider)
- **Execution Time**: ~16 seconds
- **Coverage**:
  - `chunking/core.py`: 100%
  - `embedders.py`: 87%
  - `vector_store.py`: 75%
  - `storage.py`: 67%

## Running Tests

```bash
# Run all service layer tests
pytest tests/unit/services/ -v

# Run with coverage
pytest tests/unit/services/ --cov=unravel/services --cov-report=term-missing

# Run specific test file
pytest tests/unit/services/test_chunking_service.py -v

# Run specific test class
pytest tests/unit/services/test_embedders.py::TestEmbedderInitialization -v

# Run specific test
pytest tests/unit/services/test_storage.py::TestDocumentStorage::test_save_document_creates_file -v
```

## Test Patterns and Best Practices

### Organization
- Tests are organized into logical classes (e.g., `TestProviderRegistry`, `TestHierarchicalChunking`)
- Each class focuses on a specific aspect of functionality
- Test names clearly describe what is being tested

### Mocking Strategy
- All external dependencies are properly mocked:
  - Qdrant client (`mock_get_client`)
  - ML models (`mock_load`)
  - File system operations (`tmp_path`, `patch`)
- Mocks are minimal and focused on the interface being tested

### Edge Cases Covered
- Empty inputs (empty strings, empty lists)
- Unicode and special characters
- Very large inputs
- Dimension mismatches
- Invalid parameters
- Missing files/configurations
- Security issues (path traversal)

### Assertions
- Clear, specific assertions
- Test both positive and negative cases
- Verify error messages when testing exceptions

## Known Issues

### Expected Failures (xfail)

1. **test_hybrid_paragraph_aligned**: Known edge case in docling provider line 332
   - Occurs when using paragraph_aligned mode with raw markdown text
   - IndexError when processing certain text structures
   - Issue documented for future fix in production code

## Future Enhancements

Potential areas for additional testing:
- Integration tests between services
- Performance benchmarks for large datasets
- Stress testing for concurrent operations
- More complex chunking scenarios
- Additional edge cases for embeddings (model failures, memory limits)
- Network failure scenarios for vector store

## Dependencies

Tests use the following testing libraries:
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `unittest.mock`: Mocking framework
- `numpy`: Array operations in tests

No additional dependencies beyond the main project requirements.
