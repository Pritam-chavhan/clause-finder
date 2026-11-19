"""Step 1: turn a regulations PDF (or .txt) into a list of clauses.

Usage:
    python ingest.py data/part2.pdf                          # writes clauses.jsonl
    python ingest.py data/part2.pdf --chapters 6-11          # keep only chapters 6 to 11
    python ingest.py data/a.pdf data/b.pdf --out clauses.jsonl

Each record: id, source, clause_no, title, text, page, page_end, part.
  page     = page where THIS piece of text starts
  page_end = page where it ends (differs from page when a long clause runs across pages)

A "clause" starts at a line that begins with a number like 6.2 or 6.2.1 followed by a heading.

Things this script deliberately ignores:
  - contents pages and contents-style lines (they end in a page number)
  - running page headers and footers
  - table rows and stray numbers (a clause heading must start with a word)
  - with --chapters, numbered lines whose first number is outside the chapters you list
Long clauses are cut at sentence or word boundaries, never in the middle of a word.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

CLAUSE_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){1,5})\.?\s+(\S.*)$")
LEADER_RE = re.compile(r"(\.\s?){4,}")
PAGEREF_RE = re.compile(r"\s\d{1,3}\s*$")  # a line that ends in a bare page number
FOOTER_RE = re.compile(
    r"Comprehensive (General )?Development Control Regulations|UD\s*&\s*UHD|^\s*Page\s+\d+",
    re.I,
)
MAX_CHARS = 1200   # target size of a searchable piece
TOC_MAX_PAGE = 10  # contents pages are only looked for in the first pages


def parse_chapters(spec):
    """'6-11' or '6,7,8' or '6-8,10' -> {6, 7, ...}. Empty/None -> None (no filter)."""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out or None


UNIT_START_RE = re.compile(r"^(mtr|mtrs|mts|mt|mtr\.|m|sq|sqm|sq\.mt|metre|metres|meter|meters|ft|feet|nos|no|%)\b", re.I)


def valid_number(num):
    """Real clause numbers never contain a zero component (6.00, 10.00 are table values like '6.00 mtr.')."""
    return all(int(p) > 0 for p in num.split("."))


def valid_heading(rest):
    """A real clause heading starts with a word, not with table numbers, units or symbols."""
    rest = rest.strip()
    if not rest or not (rest[0].isalpha() or rest[0] in "\"'(“"):
        return False
    if UNIT_START_RE.match(rest):
        return False
    letters = sum(c.isalpha() for c in rest)
    non_space = sum(not c.isspace() for c in rest) or 1
    return letters >= 6 and letters / non_space >= 0.6


def is_toc_page(lines):
    """Early page where many lines end in a page number = a table-of-contents page."""
    nonempty = [l for l in lines if l.strip()]
    if len(nonempty) < 8:
        return False
    hits = sum(bool(PAGEREF_RE.search(l)) for l in nonempty)
    return hits >= 8 and hits / len(nonempty) >= 0.4


def read_pages(path: Path):
    """Return a list of (page_number, text)."""
    if path.suffix.lower() == ".txt":
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append((i, page.extract_text() or ""))
    return pages


def split_clauses(pages, source, chapters=None, stats=None):
    stats = stats if stats is not None else {}
    clauses = []
    cur = None
    for page_no, text in pages:
        lines = text.splitlines()
        if page_no <= TOC_MAX_PAGE and is_toc_page(lines):
            stats["toc_pages"] = stats.get("toc_pages", 0) + 1
            continue
        for line in lines:
            line = " ".join(line.split())
            if not line or FOOTER_RE.search(line) or LEADER_RE.search(line):
                continue
            m = CLAUSE_RE.match(line)
            starts_clause = (
                m
                and valid_number(m.group(1))
                and valid_heading(m.group(2))
                and (chapters is None or int(m.group(1).split(".")[0]) in chapters)
                and not (page_no <= TOC_MAX_PAGE and PAGEREF_RE.search(line))  # contents-style line
            )
            if starts_clause:
                if cur:
                    clauses.append(cur)
                cur = {
                    "source": source,
                    "clause_no": m.group(1),
                    "title": m.group(2)[:120],
                    "text": m.group(2),
                    "marks": [(0, page_no)],  # (character offset, page) pairs
                }
            elif cur:
                if cur["marks"][-1][1] != page_no:
                    cur["marks"].append((len(cur["text"]) + 1, page_no))
                cur["text"] += " " + line
    if cur:
        clauses.append(cur)
    return clauses


def split_text(text, limit=MAX_CHARS):
    """Cut text into pieces of about `limit` characters, at sentence or word boundaries.
    Returns a list of (start_offset, piece)."""
    pieces, start, n = [], 0, len(text)
    while start < n:
        end = min(start + limit, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("; "), window.rfind(": "), window.rfind(") "))
            if cut < limit * 0.5:
                cut = window.rfind(" ")
            if cut <= 0:
                cut = len(window) - 1
            end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append((start, piece))
        start = end
        while start < n and text[start] == " ":
            start += 1
    return pieces


def page_at(marks, offset):
    page = marks[0][1]
    for off, pg in marks:
        if off <= offset:
            page = pg
        else:
            break
    return page


def chunk_long(clauses):
    out = []
    for c in clauses:
        text = c["text"].strip()
        if len(text) < 25:  # headings with no body, stray lines
            continue
        pieces = split_text(text)
        for j, (start, piece) in enumerate(pieces):
            out.append({
                "source": c["source"], "clause_no": c["clause_no"], "title": c["title"],
                "text": piece,
                "page": page_at(c["marks"], start),
                "page_end": page_at(c["marks"], start + len(piece) - 1),
                "part": j + 1,
            })
    for i, c in enumerate(out):
        c["id"] = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="clauses.jsonl")
    ap.add_argument("--chapters", default=None, help="keep only these chapters, e.g. 6-11")
    args = ap.parse_args()
    chapters = parse_chapters(args.chapters)

    all_clauses = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            sys.exit(f"File not found: {p}")
        pages = read_pages(p)
        stats = {}
        found = split_clauses(pages, p.name, chapters, stats)
        print(f"{p.name}: {len(pages)} pages, {len(found)} clauses found, "
              f"{stats.get('toc_pages', 0)} contents page(s) skipped")
        all_clauses.extend(found)

    chunks = chunk_long(all_clauses)
    with open(args.out, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} searchable chunks to {args.out}")

    by_chapter = Counter(c["clause_no"].split(".")[0] for c in chunks)
    summary = ", ".join(f"ch {k}: {v}" for k, v in sorted(by_chapter.items(), key=lambda kv: int(kv[0])))
    print("Chunks per chapter ->", summary)

    if len(chunks) < 20:
        print("Warning: very few clauses found. Is the PDF scanned (images)? "
              "Open it and check you can select text.")


if __name__ == "__main__":
    main()
