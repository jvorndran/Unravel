# Security Policy

## Supported Versions

Unravel is currently in active development (v0.x). Security updates are provided for the latest release only.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Considerations

### Local-First Architecture

Unravel processes all documents locally on your machine. No data is transmitted to external servers except:

- **LLM API calls** - When using OpenAI, Anthropic, or other cloud LLM providers for query generation
- **Web scraping** - When fetching content from URLs you provide

### API Key Storage

API keys are stored in `~/.unravel/.env` on your local filesystem:

- Keys are never logged or saved to session files
- The `.env` file should have restricted permissions
- Keys are loaded only when needed for LLM requests

**Recommendation**: Use environment-specific API keys with minimal permissions and usage limits.

### Data Storage

All processed data is stored locally in `~/.unravel/`:

```
~/.unravel/
├── documents/          # Your uploaded documents
├── chunks/             # Processed chunks
├── embeddings/         # Vector embeddings
├── indices/            # Search indices
├── .env               # API keys (if configured)
└── *.json             # Configuration files
```

**Security notes:**
- This directory may contain sensitive document content
- Embeddings can potentially leak information about source documents
- Secure this directory according to your data sensitivity requirements

### Document Processing

When processing documents:

- **File uploads**: Unravel parses files using trusted libraries (docling, python-docx, etc.)
- **URL scraping**: Selenium and Trafilatura fetch web content - only scrape trusted URLs
- **OCR processing**: Local processing only, no external services

### Dependencies

Unravel relies on numerous open-source packages. We:

- Pin major versions in `pyproject.toml`
- Use `uv.lock` for reproducible builds
- Monitor security advisories for critical dependencies

## Reporting a Vulnerability

If you discover a security vulnerability, please help us maintain the security of the project and its users.

### How to Report

**Please do NOT open a public issue for security vulnerabilities.**

Instead:

1. **Email maintainers** - Contact the project maintainers privately via GitHub email or open a private security advisory
2. **Provide details** - Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if you have one)
3. **Allow time for response** - We aim to respond within 72 hours

### What to Expect

1. **Acknowledgment** - We'll confirm receipt of your report within 72 hours
2. **Assessment** - We'll investigate and determine severity and impact
3. **Fix development** - We'll develop and test a fix
4. **Disclosure** - We'll coordinate public disclosure with you
5. **Credit** - We'll credit you in release notes (unless you prefer to remain anonymous)

### Scope

**In scope:**
- Code execution vulnerabilities
- Data leakage through embeddings or logs
- API key exposure or theft
- Malicious document processing exploits
- Dependency vulnerabilities in core functionality

**Out of scope:**
- Social engineering attacks
- Physical attacks on user machines
- Vulnerabilities in third-party services (OpenAI, Anthropic, etc.)
- Issues requiring physical access to the user's machine
- Denial of service through resource-intensive documents (expected behavior)

## Security Best Practices for Users

1. **Protect API keys** - Never commit `.env` files or share API keys
2. **Trust your sources** - Only upload documents and scrape URLs you trust
3. **Secure storage** - Protect the `~/.unravel/` directory if processing sensitive documents
4. **Update regularly** - Keep Unravel updated to receive security fixes
5. **Review permissions** - Understand that LLM APIs receive document chunks you send them

## Known Limitations

- **API key security**: Keys stored in plaintext in `.env` (standard practice for local development tools)
- **Embedding privacy**: Vector embeddings may reveal information about source documents
- **LLM data sharing**: Cloud LLM providers receive document chunks and queries per their terms of service
- **Web scraping**: Selenium executes JavaScript from scraped pages (only scrape trusted sites)

---

Thank you for helping keep Unravel and its users secure.
