#!/usr/bin/env python3
"""
ingest.py — Register a new source document into raw/ and queue it for compilation.

Usage:
    python tools/ingest.py <path-to-file-or-url> [--type paper|repo|dataset|image]
    python tools/ingest.py --list          # show all uncompiled sources
    python tools/ingest.py --status        # show compilation status of all sources
"""
import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "raw"
STATUS_FILE = ROOT / ".ingest_status.json"

TYPE_DIR = {
    "paper": RAW / "papers" / "pdf",
    "clipped": RAW / "papers" / "clipped",
    "repo": RAW / "repos",
    "dataset": RAW / "datasets",
    "image": RAW / "images",
}


def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {}


def save_status(status: dict):
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def ingest(src: str, kind: str):
    src_path = Path(src)
    if not src_path.exists():
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)

    dest_dir = TYPE_DIR[kind]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_path.name

    if dest.exists():
        print(f"Already ingested: {dest.relative_to(ROOT)}")
    else:
        shutil.copy2(src_path, dest)
        print(f"Copied to: {dest.relative_to(ROOT)}")

    status = load_status()
    key = str(dest.relative_to(ROOT))
    if key not in status:
        status[key] = {"ingested": str(date.today()), "compiled": False}
        save_status(status)
        print(f"Queued for compilation.")
    else:
        print(f"Status: {'compiled' if status[key]['compiled'] else 'pending compilation'}")


def list_pending():
    status = load_status()
    pending = [k for k, v in status.items() if not v["compiled"]]
    if not pending:
        print("All sources compiled.")
        return
    print("Pending compilation:")
    for p in pending:
        print(f"  {p}")
    print(f"\nTo compile, run in Claude Code:")
    print(f"  Use prompts/compile_paper.md (substitute SOURCE_PATH)")


def show_status():
    status = load_status()
    if not status:
        print("No sources ingested yet.")
        return
    compiled = [k for k, v in status.items() if v["compiled"]]
    pending = [k for k, v in status.items() if not v["compiled"]]
    print(f"Total: {len(status)}  |  Compiled: {len(compiled)}  |  Pending: {len(pending)}")
    print("\nCompiled:")
    for k in compiled:
        print(f"  [x] {k}")
    print("\nPending:")
    for k in pending:
        print(f"  [ ] {k}  (ingested {status[k]['ingested']})")


def mark_compiled(path: str):
    status = load_status()
    if path in status:
        status[path]["compiled"] = True
        save_status(status)
        print(f"Marked as compiled: {path}")
    else:
        print(f"Not found in status: {path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest source documents into raw/")
    parser.add_argument("source", nargs="?", help="File path to ingest")
    parser.add_argument("--type", choices=["paper", "clipped", "repo", "dataset", "image"],
                        default="paper", help="Document type (default: paper)")
    parser.add_argument("--list", action="store_true", help="List pending sources")
    parser.add_argument("--status", action="store_true", help="Show all status")
    parser.add_argument("--mark-compiled", metavar="PATH", help="Mark a path as compiled")
    args = parser.parse_args()

    if args.list:
        list_pending()
    elif args.status:
        show_status()
    elif args.mark_compiled:
        mark_compiled(args.mark_compiled)
    elif args.source:
        ingest(args.source, args.type)
    else:
        parser.print_help()
