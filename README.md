# WordNLP

A Streamlit app for analyzing Word documents. Comments, redlines, moves, topics, and search, all in one place. No data leaves your machine.

![Demo](assets/wordnlp_demo.gif)

## Features

### Document Statistics
Upload a redlined Word document and get an overview of it:

- **Comments** — count by author, timeline of when comments were made, expanded view with full comment text and highlighted selected text
- **Redlines** — insertions and deletions by author and over time; insertion/deletion pairs by the same author are combined into a single "edit" view with red strikethrough and blue underline
- **Moves** — paragraphs that were relocated, with source/destination indices and distance
- All three views support **Counts** and **Timeline** tabs, date range filtering, and author filtering from the sidebar

### Document Terms
Extracts defined terms from the document — quoted definitions, title-case terms introduced in parentheses, and monetary/date references pulled via NLP (spaCy). Useful for quickly auditing the defined terms section and spotting inconsistencies.

### Topic Explorer
Unsupervised topic modelling across the document's paragraphs or sentences using BERTopic + UMAP:

- Three levels of granularity (high / mid / low) controlled by normalised sliders
- Interactive datamapplot map with hover text, or a static PNG fallback
- Filter and search results by keyword, regex, BM25 relevance, or semantic similarity
- Optional seed words to guide topic discovery
- Configurable embedding model (built-in or any HuggingFace model ID)

### Search
Full-text search across one or more uploaded documents:

- **Keyword** — exact substring match with highlighting
- **Regex** — regular expression match
- **Relevance** — BM25 ranking with stemmed token highlighting
- **Semantic** — cosine similarity via sentence-transformers; supports built-in models or any HuggingFace model ID
- Paginated results; search results are cached so pagination never re-runs the search

---

## Running Locally

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install and run with uv

```bash
git clone https://github.com/your-org/wordnlp.git
cd wordnlp
uv sync
uv run streamlit run app.py
```

### Install and run with pip

```bash
git clone https://github.com/your-org/wordnlp.git
cd wordnlp
pip install -e .
python -m spacy download en_core_web_sm
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
wordnlp/
├── app.py                  # Entry point — registers all pages
├── pages/                  # One file per Streamlit page
│   ├── document_statistics.py
│   ├── document_terms.py
│   ├── topic_explorer.py
│   └── search.py
├── src/                    # Core logic (no Streamlit in hot paths)
│   ├── app_state.py        # Session state keys and typed accessors
│   ├── comments/           # Comment extraction and rendering
│   ├── redlines/           # Redline and move extraction
│   ├── stats/              # Compute, render, and config helpers
│   └── utils/              # Text processing, model loading, page helpers
├── config/
│   └── app.toml            # Tuneable parameters (BM25, UMAP, thresholds, etc.)
├── assets/
│   └── logo.svg
└── tests/
```

## Configuration

`config/app.toml` exposes parameters you might want to tune without touching code:

| Section | Key | Default | Description |
|---|---|---|---|
| `search` | `min_para_chars` | `30` | Minimum paragraph length included in search corpus |
| `search` | `bm25_k1` | `1.5` | BM25 term frequency saturation |
| `search` | `bm25_b` | `0.75` | BM25 length normalisation |
| `topic` | `min_passages` | `12` | Minimum passages required before topic modelling runs |
| `topic` | `static_map_threshold` | `6` | Topic count at or below which the map defaults to static |
| `topic` | `umap_min_dist` | `0.05` | UMAP min_dist — controls cluster tightness |
| `display` | `date_format` | `"%B %-d, %Y"` | Date display format across the app |
| `time_bin` | `day_max_days` | `30` | Spans up to this many days use day-level binning |
| `time_bin` | `week_max_days` | `180` | Spans up to this many days use week-level binning |

---

## Requirements

All dependencies are listed in `pyproject.toml`. Key ones:

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `bertopic` + `umap-learn` | Topic modelling |
| `sentence-transformers` | Embeddings for semantic search and topic modelling |
| `spacy` + `en_core_web_sm` | NER for document terms extraction |
| `datamapplot` | Topic map visualisation |
| `plotly` + `altair` | Charts |
| `pystemmer` | Stemming for BM25 and token highlighting |
