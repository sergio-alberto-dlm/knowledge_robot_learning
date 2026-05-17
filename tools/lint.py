#!/usr/bin/env python3
"""
lint.py — Wiki health check. Run before asking the LLM to do a full lint pass.

Usage:
    python tools/lint.py           # print report to stdout
    python tools/lint.py --json    # machine-readable output
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
WIKI = ROOT / "wiki"


def get_articles() -> dict[str, str]:
    return {
        str(f.relative_to(ROOT)): f.read_text(errors="ignore")
        for f in WIKI.rglob("*.md")
        if not f.name.startswith("_")
    }


def check_index_consistency(articles: dict[str, str]) -> list[str]:
    issues = []
    index_text = (WIKI / "_index.md").read_text(errors="ignore") if (WIKI / "_index.md").exists() else ""
    summaries_text = (WIKI / "_summaries.md").read_text(errors="ignore") if (WIKI / "_summaries.md").exists() else ""
    for path in articles:
        fname = Path(path).name
        if fname not in index_text:
            issues.append(f"[index] Missing from _index.md: {path}")
        if fname not in summaries_text:
            issues.append(f"[summaries] Missing from _summaries.md: {path}")
    return issues


def check_broken_links(articles: dict[str, str]) -> list[str]:
    issues = []
    link_re = re.compile(r"\[.*?\]\(([^)]+)\)")
    for path, text in articles.items():
        article_dir = Path(path).parent
        for match in link_re.finditer(text):
            href = match.group(1)
            if href.startswith("http") or href.startswith("#"):
                continue
            target = (ROOT / article_dir / href).resolve()
            if not target.exists():
                issues.append(f"[broken-link] {path} → {href}")
    return issues


def check_frontmatter(articles: dict[str, str]) -> list[str]:
    required = {"title", "type", "tags", "related", "created", "updated"}
    issues = []
    for path, text in articles.items():
        fm_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not fm_match:
            issues.append(f"[frontmatter] Missing frontmatter: {path}")
            continue
        fm = fm_match.group(1)
        for field in required:
            if f"{field}:" not in fm:
                issues.append(f"[frontmatter] Missing field '{field}': {path}")
    return issues


def check_orphans(articles: dict[str, str]) -> list[str]:
    issues = []
    all_text = "\n".join(articles.values())
    for path in articles:
        fname = Path(path).name
        stem = Path(path).stem
        occurrences = all_text.count(fname) + all_text.count(stem)
        # The file references itself in its own text, so threshold is 2
        if occurrences < 2:
            issues.append(f"[orphan] No backlinks found: {path}")
    return issues


def check_stubs(articles: dict[str, str]) -> list[str]:
    issues = []
    for path, text in articles.items():
        # Strip frontmatter
        body = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
        sentences = re.split(r"[.!?]+", body)
        non_empty = [s.strip() for s in sentences if len(s.strip()) > 20]
        if len(non_empty) < 3:
            issues.append(f"[stub] Article too short (< 3 sentences): {path}")
    return issues


def run_lint() -> dict:
    articles = get_articles()
    if not articles:
        return {"total_articles": 0, "issues": [], "summary": "No articles found."}

    issues = []
    issues += check_index_consistency(articles)
    issues += check_broken_links(articles)
    issues += check_frontmatter(articles)
    issues += check_orphans(articles)
    issues += check_stubs(articles)

    by_category: dict[str, list[str]] = {}
    for issue in issues:
        cat = re.match(r"\[([^\]]+)\]", issue).group(1)
        by_category.setdefault(cat, []).append(issue)

    return {
        "total_articles": len(articles),
        "total_issues": len(issues),
        "by_category": {k: len(v) for k, v in by_category.items()},
        "issues": issues,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wiki health check")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = run_lint()

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Wiki Lint Report")
        print(f"{'='*40}")
        print(f"Articles scanned : {report['total_articles']}")
        print(f"Total issues     : {report['total_issues']}")
        if report.get("by_category"):
            print("\nIssues by category:")
            for cat, count in report["by_category"].items():
                print(f"  [{cat}] {count}")
        print()
        if report["issues"]:
            for issue in report["issues"]:
                print(f"  {issue}")
        else:
            print("No issues found. Wiki is healthy.")
