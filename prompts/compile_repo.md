# Prompt: Compile Repository

Compile the codebase at `{{REPO_PATH}}` into the wiki.

Steps:
1. Read the top-level README and directory structure
2. Identify the major components (e.g., frontend end, backend, modules)
3. For each major component, create `wiki/codebase/{{SYSTEM_NAME}}/{{component}}.md` with:
   - **Purpose** — what problem it solves
   - **Key Files** — list with one-line descriptions
   - **Interfaces** — inputs, outputs, public API or ROS topics/services
   - **Algorithms** — which SLAM algorithms or methods are implemented
   - **Dependencies** — external libraries, other components it calls
   - **Known Issues / TODOs** (from comments or issue tracker if present)
4. Create `wiki/codebase/{{SYSTEM_NAME}}/_overview.md` with a Mermaid component diagram
5. Link each component article to relevant `wiki/concepts/` articles
6. Update `wiki/_index.md` and `wiki/_summaries.md`
