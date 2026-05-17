# Robot Learning Knowledge Base — LLM Instructions

This is a personal research knowledge base about Robot Learning.
The LLM (you) owns and maintains the `wiki/` directory. The human rarely edits it directly.

## Directory Contract

| Directory | Owner | Rule |
|-----------|-------|------|
| `raw/` | Human | Read-only for LLM. Never modify. |
| `wiki/` | LLM | You write, update, and maintain all files here. |
| `outputs/` | LLM | Write reports, slides, and visualizations here. |
| `tools/` | Human | Python CLI tools. Read and use them. |
| `prompts/` | Human | Reusable prompt templates. |

## Wiki Conventions

### File naming
- Concept articles: `wiki/concepts/<kebab-case>.md`
- Paper summaries: `wiki/papers/<author_year_shorttitle>.md`
- Codebase docs: `wiki/codebase/<system>/<component>.md`
- Comparisons: `wiki/comparisons/<topic>.md`

### Every wiki article must have this frontmatter:
```yaml
---
title: <human-readable title>
type: concept | paper | codebase | comparison | open_question
tags: [tag1, tag2]
related: [other-article.md, ...]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [raw/papers/..., ...]
---
```

### The two mandatory index files (keep always up-to-date):
- `wiki/_index.md` — one-line entry per article, grouped by type
- `wiki/_summaries.md` — 3-5 sentence summary per article, for fast LLM context loading

### Backlinks
Every article must have a `## See Also` section at the bottom listing related articles with relative links.

## Core Workflows

### 1. Compile a new paper
When asked to compile `raw/papers/pdf/<file>.pdf` or `raw/papers/clipped/<file>.md`:
1. Read the source thoroughly
2. Create `wiki/papers/<author_year_title>.md` with: abstract, key contributions, methodology, results, limitations.
3. Extract any new concepts not yet in `wiki/concepts/` and create stubs or full articles
4. Update `wiki/_index.md` and `wiki/_summaries.md`
5. Add backlinks in related existing articles

### 2. Compile a codebase
When asked to compile `raw/repos/<name>/`:
1. Identify the main components and their responsibilities
2. Create one article per major component in `wiki/codebase/<name>/`
3. Create `wiki/codebase/<name>/_overview.md` with architecture diagram in Mermaid
4. Link code components to relevant concept articles

### 3. Answer a research question
When given a Q&A query:
1. Load `wiki/_summaries.md` first to orient yourself
2. Read the relevant subset of wiki articles
3. Synthesize the answer
4. Write the output to `outputs/reports/<YYYY-MM-DD>_<short-title>.md`
5. Ask if the output should be filed back into the wiki

### 4. Generate a slide deck
Output file: `outputs/slides/<YYYY-MM-DD>_<title>.md` in Marp format.
Always start the file with:
```
---
marp: true
theme: default
paginate: true
---
```

### 5. Lint the wiki
Check for:
- Articles in `_index.md` that are missing from `_summaries.md` or vice versa
- Broken relative links
- Articles with no `related` frontmatter entries
- Concepts referenced in papers but with no concept article
- Open questions that could now be answered from existing wiki content

## LLM Operating Principles
- Be incremental: always update `_index.md` and `_summaries.md` after any wiki change
- Prefer updating existing articles over creating new ones when the topic overlaps
- Flag conflicts between sources explicitly (e.g., contradictory benchmark results)
- When uncertain about a technical claim, add a `> **Verify:** ...` blockquote
- Never delete a wiki article without asking; instead mark it `status: deprecated`
