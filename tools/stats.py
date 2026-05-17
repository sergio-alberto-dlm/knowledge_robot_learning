#!/usr/bin/env python3
"""
stats.py — Quick overview of the knowledge base size and growth.

Usage:
    python tools/stats.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
WIKI = ROOT / "wiki"
RAW = ROOT / "raw"


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def main():
    # Raw sources
    raw_files = list(RAW.rglob("*"))
    raw_files = [f for f in raw_files if f.is_file()]

    # Wiki articles
    wiki_files = [f for f in WIKI.rglob("*.md") if not f.name.startswith("_")]
    total_words = sum(count_words(f.read_text(errors="ignore")) for f in wiki_files)

    # Articles by type
    type_counts: dict[str, int] = {}
    for f in wiki_files:
        text = f.read_text(errors="ignore")
        m = re.search(r"^type:\s*(\S+)", text, re.MULTILINE)
        kind = m.group(1) if m else "unknown"
        type_counts[kind] = type_counts.get(kind, 0) + 1

    print(f"SLAM Knowledge Base — Stats")
    print(f"{'='*35}")
    print(f"Raw source files  : {len(raw_files)}")
    print(f"Wiki articles     : {len(wiki_files)}")
    print(f"Total wiki words  : {total_words:,}")
    print()
    print("Articles by type:")
    for kind, count in sorted(type_counts.items()):
        print(f"  {kind:<20} {count}")


if __name__ == "__main__":
    main()
