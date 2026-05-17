# Prompt: Research Query

**Question:** {{QUESTION}}

**Output format:** {{report | slides | visualization | wiki_article}}

Steps:
1. Read `wiki/_summaries.md` to identify the most relevant articles
2. Read the full text of relevant articles (start with concepts, then papers, then codebase)
3. If the answer requires information not in the wiki, note the gap explicitly
4. Write the answer to `outputs/{{FORMAT}}/{{YYYY-MM-DD}}_{{short-slug}}.md`
5. At the end of the output, include:
   - **Sources used** (wiki articles consulted)
   - **Gaps identified** (information that would improve this answer)
   - **Follow-up questions** (natural next queries)
6. Ask: "Should I file this output back into the wiki?"
