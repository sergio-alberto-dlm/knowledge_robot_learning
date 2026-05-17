# Prompt: Compile Paper

Compile the paper at `{{SOURCE_PATH}}` into the wiki.

Steps:
1. Read the full paper
2. Create `wiki/papers/{{AUTHOR_YEAR_TITLE}}.md` with these sections:
   - **Abstract** (verbatim or paraphrased)
   - **Key Contributions** (bulleted)
   - **Methodology** (with any equations or diagrams worth noting)
   - **Experimental Results** (benchmarks, datasets used, metrics)
   - **Limitations & Open Questions**
   - **See Also** (links to related wiki articles)
3. For each new concept introduced: check `wiki/concepts/` and either update an existing article or create a new stub
4. Update `wiki/_index.md` and `wiki/_summaries.md`
5. Update `.ingest_status.json`
6. Report what was created or updated
