# Building Unravel: Learning RAG Through Visualization

**GitHub:** [github.com/unravel/unravel](https://github.com/unravel/unravel)

When I started learning about Retrieval-Augmented Generation (RAG), I quickly hit a wall. Not from lack of documentation or tutorials, but from something more fundamental: I couldn't see what was actually happening.

RAG systems are everywhere now. They power chatbots that answer questions about your documentation, search engines that understand context, and assistants that retrieve relevant information before generating responses. But understanding how they work—really understanding the tradeoffs between different approaches—proved surprisingly difficult.

## The Black Box Problem

Most RAG tutorials follow a familiar pattern: install a vector database, chunk your documents, generate embeddings, and query. The code runs, you get results, but crucial questions remain unanswered:

- Why did my chunking strategy return 15 chunks instead of 20?
- How much do my chunks overlap, and is that actually helping?
- What does my embedding space look like? Are similar documents clustering together?
- Why did semantic search return these specific chunks for my query?
- How different are the results between BM25 and dense retrieval?

These aren't academic questions. Each decision—chunk size, overlap percentage, embedding model, retrieval strategy—creates tradeoffs that affect your system's accuracy, speed, and reliability. But most frameworks abstract these details away, leaving you to tune parameters without understanding their effects.

![Unravel Demo](docs/demo.gif)

## Learning By Building

I took a different approach. Instead of just implementing RAG systems, I built Unravel—a tool to visualize every step of the pipeline in real-time. As I learned each concept, I added it to the visualization. When I discovered a new retrieval strategy, I implemented it and compared it visually against the others.

The result is an educational tool that exposes what's typically hidden:

### 1. Document Parsing That Shows Its Work

Upload a PDF and watch it get parsed into structured elements—headers, paragraphs, tables, images. See exactly how the document hierarchy is preserved and which content gets extracted. Toggle OCR and witness scanned text become searchable. The visualization shows you what your RAG system sees, not just what you see when opening the file.

### 2. Chunking Strategies You Can See

Chunking is where most RAG tutorials gloss over critical details. Unravel renders each chunk as a card with full metadata: token count, section breadcrumbs, source page numbers, and overlap with neighboring chunks. Want to understand the difference between hierarchical chunking (preserving document structure) and hybrid chunking (maximizing token consistency)? Compare them side-by-side and see how the same document splits differently.

### 3. Embedding Space Exploration

Embeddings are vectors, but vectors are abstract. Unravel projects your embeddings into 3D space using UMAP, color-codes clusters using k-means, and highlights outliers. Hover over any point to preview its chunk. Run a semantic search and watch your query vector project into the same space, with lines connecting to retrieved results. Suddenly, "cosine similarity" becomes tangible.

### 4. Retrieval Strategy Comparison

Want to understand when dense retrieval beats BM25? Configure both strategies, run the same query, and compare results. Add hybrid fusion (weighted sum or reciprocal rank fusion) to see how combining approaches improves coverage. Enable query expansion and watch multiple query variations merge into better results through reciprocal rank fusion. Enable cross-encoder reranking and observe how results get re-ordered by relevance.

Every retrieval configuration shows its results with similarity scores, source locations, and the exact chunks returned. No guessing, no hidden magic.

### 5. Code Export

Once you've found a configuration that works, export production-ready Python code with all your settings preserved. The generated code includes parsing options, chunking parameters, embedding models, retrieval strategies, and LLM configuration—ready to integrate into your application.

## What I Learned Building This

The process of building Unravel taught me more about RAG than any tutorial could:

**Chunking overlap isn't free.** Overlap helps with context continuity, but increases storage, embedding costs, and retrieval noise. Visualizing chunks with overlap highlighting made this tradeoff concrete.

**Embedding models cluster differently.** Some models create tight, well-separated clusters. Others produce diffuse, overlapping embeddings. The 3D visualization revealed which models worked better for my documents before I ran a single query.

**Hybrid retrieval isn't always better.** Dense and sparse retrieval excel at different things. Sometimes BM25 alone works perfectly for keyword-heavy queries. Sometimes dense retrieval captures semantic nuance that keywords miss. Visualizing both side-by-side helped me understand when to use which.

**Local LLMs are viable.** Running Ollama locally with Unravel proved that you don't need API keys or cloud services to build RAG systems. The performance gap is narrowing.

## The Educational Mission

Unravel exists primarily as an educational tool. It's designed for anyone learning RAG who wants to understand the mechanics, not just copy-paste code. Whether you're experimenting with document parsing, comparing embedding models, or tuning retrieval strategies, Unravel shows you what's happening at each step.

The tool runs entirely locally. Your documents never leave your machine. All vector storage happens in a local Qdrant instance. You control whether to use cloud LLM APIs or local models through Ollama or LM Studio.

## The Vision: Comprehensive RAG Evaluation

While Unravel currently focuses on education and experimentation, I envision it evolving into a comprehensive RAG evaluation tool. The groundwork is there—the visualizations, the multi-strategy support, the export functionality. The next phase would add:

- Systematic evaluation metrics (precision, recall, MRR, NDCG)
- Ground truth datasets and automated benchmarking
- Performance profiling (latency, throughput, cost per query)
- Support for every major RAG strategy (semantic cache, HyDE, reranking variants, multi-vector retrieval)
- Comparative analysis across embedding models, chunking strategies, and retrieval methods

The goal would be a tool that helps you not just understand RAG, but rigorously evaluate and optimize your implementation for production.

## Try It Yourself

Unravel is open source and available now:

```bash
pip install unravel
unravel
```

The app launches in your browser and walks you through five steps: document upload, chunk visualization, embedding exploration, query testing, and code export. Start with the sample documents or upload your own. Experiment with different configurations. Watch how changes propagate through the pipeline.

If you're learning RAG, I hope Unravel makes the concepts clearer and the tradeoffs visible. If you're building RAG systems, I hope it helps you make informed decisions backed by evidence, not guesswork.

The code is on GitHub, contributions are welcome, and I'd love to hear how you use it.

---

**Links:**
- PyPI: [pypi.org/project/unravel](https://pypi.org/project/unravel)

**About the Author:**
A developer who learned RAG by building visualization tools and decided to share the journey.
