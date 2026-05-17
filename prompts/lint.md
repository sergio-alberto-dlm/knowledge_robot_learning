# Prompt: Wiki Health Check

Run a full lint pass over the wiki. Check for and fix or flag:

1. **Index consistency** — every file in `wiki/` must have an entry in `_index.md` and `_summaries.md`
2. **Broken links** — relative links that point to non-existent files
3. **Missing frontmatter fields** — any article missing required frontmatter keys
4. **Orphaned articles** — articles with no backlinks from any other article
5. **Concept stubs** — concept articles with fewer than 3 sentences (mark for expansion)
6. **Contradictions** — papers or articles that make conflicting claims about the same metric or fact
7. **Imputable gaps** — missing data that could be filled by web search (flag, do not invent)
8. **New article candidates** — concepts mentioned in 3+ articles but with no dedicated article

Output a lint report to `outputs/reports/{{YYYY-MM-DD}}_lint.md` with:
- Summary counts per category
- Actionable fix list (sorted by priority)
- Suggested new article titles
