# Contributing to Unravel

Thank you for considering contributing to Unravel. This guide will help you get started.

## Getting Started

### Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/unravel.git
   cd unravel
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   uv sync --all-extras
   ```

3. Verify your setup:
   ```bash
   uv run pytest
   unravel  # Should launch the app
   ```

## Development Philosophy

Read [CLAUDE.md](CLAUDE.md) for our coding standards. Key principles:

- **Simplicity over cleverness** - Write straightforward code
- **Minimal changes** - Only modify what's necessary for your task
- **Test your work** - Verify with realistic inputs before submitting
- **Build iteratively** - Start minimal, verify it works, then add complexity

## Types of Contributions

### Bug Fixes

1. Search existing issues to avoid duplicates
2. Open an issue describing the bug with reproduction steps
3. Wait for maintainer feedback before starting work
4. Reference the issue number in your PR

### New Features

Features should align with Unravel's core mission: making RAG experimentation transparent and accessible.

**Good feature examples:**
- New embedding models or chunking strategies
- Enhanced visualizations for understanding retrieval
- Additional document parsing formats
- Performance optimizations for large documents

**Think twice about:**
- Features that increase complexity without clear value
- Production-only features (Unravel is an experimentation tool)
- Heavy dependencies that slow installation

Before building a feature:
1. Open a feature request issue
2. Discuss the approach with maintainers
3. Get approval before investing significant time

### Documentation

Documentation improvements are always welcome:
- Clarifying confusing sections in the README
- Adding examples for common use cases
- Fixing typos or broken links
- Improving code comments in complex areas

## Pull Request Process

### Before Submitting

- [ ] Run tests: `uv run pytest`
- [ ] Format code: `uv run black unravel`
- [ ] Lint code: `uv run ruff check unravel`
- [ ] Test locally with `unravel` command
- [ ] Verify your changes with realistic test data

### PR Guidelines

1. **Clear description**: Explain what changes you made and why
2. **Reference issues**: Link to related issue numbers
3. **Keep it focused**: One logical change per PR
4. **Update tests**: Add or update tests for your changes
5. **Screenshots**: Include for UI changes

### Review Process

- Maintainers typically review PRs within a few days
- Feedback may request changes or clarification
- CI must pass before merging
- Squash and merge is preferred for clean history

## Code Style

This project uses:
- **Black** for formatting (line length: 100)
- **Ruff** for linting
- **Type hints** where they improve clarity
- **Google-style docstrings** for public functions

Run formatters before committing:
```bash
uv run black unravel
uv run ruff check --fix unravel
```

## Testing Guidelines

Write tests for:
- New features and bug fixes
- Edge cases and error conditions
- Different file types (for parsing features)
- Various configurations (for retrieval features)

Run tests:
```bash
uv run pytest                    # All tests
uv run pytest tests/test_foo.py  # Specific test file
uv run pytest -v                 # Verbose output
```

## Project Structure

```
unravel/
├── cli.py              # Entry point and CLI
├── app.py              # Main Streamlit app
├── components/         # UI components for each step
├── core/              # Core RAG functionality
│   ├── parsing/       # Document parsing
│   ├── chunking/      # Text splitting
│   ├── embeddings/    # Embedding models
│   └── retrieval/     # Search and reranking
├── utils/             # Shared utilities
└── storage/           # Local storage management

tests/                 # Test suite mirrors src structure
```

## Areas Looking for Help

Check issues labeled:
- `good first issue` - Good for newcomers
- `help wanted` - Maintainers would appreciate assistance
- `enhancement` - New feature proposals

## Questions?

- Open a discussion for general questions
- Open an issue for bugs or feature requests
- Check existing issues and discussions first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make RAG experimentation more accessible!
