# gh-issue-landscape — Design Spec

## Summary

A reusable CLI tool that creates semantic 2D and 3D visualizations of GitHub issues. Point it at any `owner/repo` and it produces an interactive map where spatial proximity = semantic similarity, revealing thematic clusters across open issues.

## Origin

Inspired by the "data cartography" technique — using text embeddings + dimensionality reduction (UMAP) + clustering (HDBSCAN) to create spatial maps of text documents. The closest existing implementation is Steven Fazzio's "Semantic Map of GitHub" (repos by README similarity using Cohere + UMAP + Toponymy + DataMapPlot). We're applying this to GitHub issues.

## Goals (ordered by priority)

1. **Methodology exploration** — build a reusable tool anyone can point at any GitHub repo
2. **Strategic prioritization** — identify neglected theme clusters, spot emerging patterns
3. **Live exploration** — interactive dashboard with filters (stretch goal)
4. **Cross-project insight** — map multiple repos in one landscape (future)

## Architecture: Hybrid Pipeline

**Chosen approach:** Use BERTopic for the ML pipeline (embeddings, UMAP, HDBSCAN, topic labeling) but render final output with both DataMapPlot (publication-quality 2D) and a custom Three.js viewer (interactive 3D exploration).

### Pipeline Steps

```
GitHub Issues (gh CLI)
    → Text Preparation (title + body, clean markdown/HTML)
    → Embedding (configurable backend)
    → Dimensionality Reduction (UMAP, 2D and 3D)
    → Clustering (HDBSCAN via BERTopic)
    → Topic Labeling (c-TF-IDF + optional LLM refinement)
    → Visualization (multiple outputs)
```

### Embedding Backend (configurable)

The tool must support multiple embedding backends via a `--embedding` flag:

| Backend | Flag value | How it works |
|---------|-----------|--------------|
| Local sentence-transformers | `--embedding local` (default) | Uses `nomic-embed-text-v1.5` or `all-MiniLM-L6-v2` via sentence-transformers |
| OpenAI-compatible API | `--embedding api --api-url <url> --api-key <key>` | Works with OpenAI, z.ai, OMLX, Ollama, vLLM, llama.cpp — anything exposing `/v1/embeddings` |
| OpenAI | `--embedding openai` | Shortcut for `text-embedding-3-small` via OpenAI API |

**Key requirement:** The user has an OMLX (OpenMLX) local deployment. OMLX exposes an OpenAI-compatible API, so the `api` backend covers it. No special OMLX adapter needed — just `--embedding api --api-url http://localhost:PORT`.

### Output Formats

**Phase 1 (v1):**
- **A) 2D Semantic Map** — DataMapPlot interactive HTML. Auto-placed topic labels, density contours, hover for issue details. The "hero" output.
- **D) CLI Summary Report** — Terminal-friendly text showing discovered topics, issue counts per cluster, top keywords, representative issues, and outliers.

**Phase 2 (v2):**
- **B) 3D Interactive Explorer** — Three.js orbit-controlled point cloud. Color by topic/priority/age/label. Filters and search.
- **C) Topic Timeline** — BERTopic's `topics_over_time()` with Plotly rendering. Shows theme evolution.

### Key Libraries

| Component | Library | Purpose |
|-----------|---------|---------|
| Topic modeling | `bertopic` | Orchestrates UMAP + HDBSCAN + c-TF-IDF |
| Dim. reduction | `umap-learn` | Reduce embeddings to 2D/3D coordinates |
| Clustering | `hdbscan` | Discover topic clusters without specifying K |
| Embeddings (local) | `sentence-transformers` | Local embedding models |
| Embeddings (API) | `openai` (Python SDK) | OpenAI-compatible API calls |
| 2D visualization | `datamapplot` | Publication-quality interactive HTML maps |
| 3D visualization | `three.js` (custom HTML) | Interactive 3D point cloud explorer |
| Topic timeline | `plotly` | Temporal topic evolution charts |
| Data collection | `gh` CLI (subprocess) | Fetch issues from any GitHub repo |
| Cluster labeling | c-TF-IDF (BERTopic built-in) | Automatic keyword extraction per cluster |
| LLM labeling | Optional (configurable) | Refine cluster names via LLM API |

### CLI Interface

```bash
# Basic usage — generates 2D map + CLI summary
gh-issue-landscape apicurio/apicurio-registry

# With API embeddings (e.g., OMLX local deployment)
gh-issue-landscape apicurio/apicurio-registry \
  --embedding api \
  --api-url http://localhost:8080 \
  --api-key sk-xxx

# Include closed issues, custom output dir
gh-issue-landscape apicurio/apicurio-registry \
  --state all \
  --output ./output/

# 3D explorer (phase 2)
gh-issue-landscape apicurio/apicurio-registry --3d

# Topic timeline (phase 2)
gh-issue-landscape apicurio/apicurio-registry --timeline
```

### Output Structure

```
output/
├── landscape-2d.html          # Interactive DataMapPlot (self-contained HTML)
├── landscape-3d.html          # Three.js explorer (phase 2)
├── timeline.html              # Topic evolution (phase 2)
├── data/
│   ├── issues.json            # Raw issue data from GitHub
│   ├── embeddings.npy         # Cached embeddings (skip re-computation)
│   ├── umap-2d.npy            # 2D coordinates
│   ├── umap-3d.npy            # 3D coordinates
│   ├── topics.json            # Cluster assignments + labels
│   └── model/                 # Saved BERTopic model (for incremental updates)
```

### Design Decisions

**Why BERTopic over raw UMAP + HDBSCAN?**
BERTopic wraps UMAP + HDBSCAN + c-TF-IDF into a single coherent pipeline with built-in visualization support (including DataMapPlot integration via `visualize_document_datamap()`). It also provides `topics_over_time()` for temporal analysis, `partial_fit()` for incremental updates, and modular representation models for cluster labeling. Reimplementing this would be significant effort with no benefit.

**Why DataMapPlot over Plotly for 2D?**
DataMapPlot produces "data cartography" — beautifully auto-placed labels, density contours, publication-quality aesthetics. Plotly scatter plots are functional but not visually compelling. DataMapPlot was created by Leland McInnes (UMAP/HDBSCAN author) specifically for this use case.

**Why Three.js for 3D over Plotly 3D?**
Plotly 3D is limited in interactivity at scale. Three.js gives full control over orbit controls, particle rendering, color modes, and filtering. For a methodology exploration tool, the 3D view should invite spatial exploration, not just display data.

**Why cache embeddings?**
Embedding 266 issues takes ~10 seconds locally or costs ~$0.01 via API. But for larger repos (5K+ issues), re-embedding on every run is wasteful. Caching in `data/embeddings.npy` with a hash of issue content lets the tool skip unchanged issues.

**Why `gh` CLI over GitHub API directly?**
The `gh` CLI handles authentication, pagination, and rate limiting. Users who can run `gh issue list` can use this tool with zero setup. It also works with GitHub Enterprise.

## Research References

Key tools and papers that informed this design:

- **BERTopic** — github.com/MaartenGr/BERTopic (9K+ stars, most mature topic modeling pipeline)
- **DataMapPlot** — github.com/TutteInstitute/datamapplot (by Leland McInnes)
- **Toponymy** — github.com/TutteInstitute/toponymy (hierarchical LLM-based topic naming)
- **Semantic Map of GitHub** — github.com/stevenfazzio/semantic-github-map (closest prior art)
- **Apple Embedding Atlas** — github.com/apple/embedding-atlas (arXiv:2505.06386, May 2025)
- **UMAP** — github.com/lmcinnes/umap
- **PaCMAP paper** — JMLR 2021, Wang et al. (alternative to UMAP with better global structure)
- **"Stop Misusing t-SNE and UMAP"** — arXiv:2506.08725 (interpretation pitfalls)

## Apicurio Registry as Test Case

The first target repo: `apicurio/apicurio-registry`
- 266 open issues with rich label taxonomy
- Labels include: 42 `area/*` labels, 4 priority levels, 7 issue types
- Good diversity of themes: storage, auth, UI, AI/agents, SDK, rules, serdes
- Manageable size — all tools work without optimization concerns

## Project Structure

```
gh-issue-landscape/
├── pyproject.toml              # Python packaging (pip install -e .)
├── README.md
├── src/
│   └── gh_issue_landscape/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point (argparse or click)
│       ├── collector.py        # GitHub issue fetching via gh CLI
│       ├── embedder.py         # Embedding backend abstraction
│       ├── pipeline.py         # BERTopic orchestration
│       ├── visualizer.py       # Output rendering (2D map, CLI summary)
│       └── viewer_3d.py        # Three.js HTML generator (phase 2)
├── templates/
│   └── explorer-3d.html       # Three.js template (phase 2)
├── tests/
├── output/                     # Default output directory (gitignored)
└── docs/
    └── design-spec.md          # This file
```
