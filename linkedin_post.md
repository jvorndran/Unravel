# LinkedIn Post: Unravel

---

When I started learning about Retrieval-Augmented Generation (RAG), I quickly hit a wall.

Not from lack of documentation or tutorials, but from something more fundamental: I couldn't see what was actually happening.

Most RAG tutorials follow the same pattern: install a vector database, chunk your documents, generate embeddings, and query. The code runs, you get results, but crucial questions remain unanswered:

- Why did my chunking strategy return 15 chunks instead of 20?
- How much do my chunks overlap, and is that helping or creating noise?
- What does my embedding space look like?
- Why did semantic search return these specific chunks?
- How different are the results between BM25 and dense retrieval?

These aren't academic questions. Each decision creates tradeoffs that affect accuracy, speed, and reliability. But most frameworks abstract these details away, leaving you to tune parameters without understanding their effects.

**So I took a different approach: I built Unravel to learn by visualizing.**

As I learned each RAG concept, I added it to the tool. When I discovered a new retrieval strategy, I implemented it and compared it visually against the others.

The result is an open-source educational tool that exposes what's typically hidden:

→ Document parsing that shows its work (structure preservation, OCR, metadata extraction)
→ Visual chunk cards with token counts, overlap highlighting, and section breadcrumbs
→ 3D embedding space exploration with cluster analysis and outlier detection
→ Side-by-side retrieval strategy comparison (dense, sparse, hybrid fusion, reranking)
→ Production-ready Python code export with your exact configuration

**What building this taught me:**

Chunking overlap isn't free. It helps with context continuity but increases storage and retrieval noise. Seeing chunks with overlap highlighting made this tradeoff concrete.

Embedding models cluster differently. Some create tight, well-separated clusters. Others produce diffuse embeddings. The 3D visualization revealed which models worked better for my documents before I ran a single query.

Hybrid retrieval isn't always better. Dense and sparse retrieval excel at different things. Visualizing both side-by-side helped me understand when to use which.

**Unravel exists primarily as an educational tool.** It's designed for anyone learning RAG who wants to understand the mechanics, not just copy-paste code.

Everything runs locally. Your documents never leave your machine. Use cloud LLM APIs or local models (Ollama, LM Studio)—your choice.

The vision: evolve this into a comprehensive RAG evaluation tool with systematic metrics, benchmarking, and support for every major RAG strategy.

If you're learning RAG or building RAG systems:

```
pip install unravel
unravel
```

Code is on GitHub. Contributions welcome.

What's your experience learning RAG? Did you face similar "black box" frustrations?

---

**Character count: ~2,450**
