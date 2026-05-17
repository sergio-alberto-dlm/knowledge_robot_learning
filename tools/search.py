#!/usr/bin/env python3
"""
search.py — Full-text search over the wiki. Usable directly or as an LLM tool.

Usage:
    python tools/search.py <query> [--top N] [--type concept|paper|codebase|...]
    python tools/search.py <query> --json     # machine-readable output for LLM tool use

Algorithm: TF-IDF over wiki markdown files (no external dependencies).
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
WIKI = ROOT / "wiki"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_wiki(type_filter: str | None = None) -> dict[str, str]:
    docs = {}
    for f in WIKI.rglob("*.md"):
        if f.name.startswith("_"):
            continue
        if type_filter:
            content = f.read_text(errors="ignore")
            if f"type: {type_filter}" not in content:
                continue
        docs[str(f.relative_to(ROOT))] = f.read_text(errors="ignore")
    return docs


def tfidf_search(query: str, docs: dict[str, str], top_n: int = 5) -> list[dict]:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []

    # Term frequency per document
    tf: dict[str, dict[str, float]] = {}
    df: dict[str, int] = defaultdict(int)
    for path, text in docs.items():
        tokens = tokenize(text)
        if not tokens:
            continue
        counts: dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        tf[path] = {t: c / len(tokens) for t, c in counts.items()}
        for t in counts:
            df[t] += 1

    N = len(docs)
    scores: dict[str, float] = {}
    for path, term_tf in tf.items():
        score = 0.0
        for qt in q_tokens:
            if qt in term_tf:
                idf = math.log((N + 1) / (df[qt] + 1)) + 1
                score += term_tf[qt] * idf
        if score > 0:
            scores[path] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for path, score in ranked:
        text = docs[path]
        # Extract title from frontmatter or first heading
        title_match = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else Path(path).stem
        # Extract a snippet around first query term hit
        snippet = ""
        for qt in q_tokens:
            m = re.search(rf".{{0,100}}{re.escape(qt)}.{{0,100}}", text, re.IGNORECASE)
            if m:
                snippet = "..." + m.group(0).strip() + "..."
                break
        results.append({"path": path, "title": title, "score": round(score, 4), "snippet": snippet})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the SLAM wiki")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top", type=int, default=5, metavar="N", help="Number of results (default 5)")
    parser.add_argument("--type", help="Filter by article type (concept, paper, codebase, ...)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    args = parser.parse_args()

    docs = load_wiki(args.type)
    if not docs:
        print("No wiki articles found.", file=sys.stderr)
        sys.exit(1)

    results = tfidf_search(args.query, docs, args.top)

    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print("No results found.")
        else:
            print(f"Top {len(results)} results for: \"{args.query}\"\n")
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['title']}]({r['path']})  (score: {r['score']})")
                if r["snippet"]:
                    print(f"   {r['snippet']}")
                print()
