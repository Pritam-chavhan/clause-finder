"""Tests for result grouping (each clause shown once) and heading-aware ranking, on invented data.
Run from the project folder:   python tests/test_search.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search import Searcher  # noqa: E402


def clause(no, title, text, page, part=1, page_end=None):
    return {"source": "t.pdf", "clause_no": no, "title": title, "text": text, "page": page,
            "page_end": page_end or page, "part": part, "id": 0}


CLAUSES = [
    clause("7.1", "Road side margin", "The road side margin table gives the distance for each road width 3.0 4.5 6.0 9.0", 10),
    clause("7.2", "Parking", "Parking spaces for residential buildings are required for every dwelling unit", 11),
    clause("7.3", "Permissible uses in margin", "Open space in the margin may be used for parking of vehicles and garden", 12, part=1),
    clause("7.3", "Permissible uses in margin", "Margin space shall not be used for storage and the margin shall stay open", 13, part=2),
    clause("7.4", "Height", "Building height depends on road width and on the margin available", 14),
    clause("7.5", "Rainwater harvesting", "Rainwater harvesting is required on large plots to recharge groundwater", 15),
    clause("7.6", "Lift", "A lift is required in tall buildings above the stated height", 16),
    clause("7.7", "Basement", "A basement may be used for services and parking with ventilation", 17),
]
S = Searcher(CLAUSES, embedder="lsa")
results = []


def check(name, cond):
    results.append(cond)
    print(("PASS  " if cond else "FAIL  ") + name)


res = S.search("margin open space", mode="keyword", k=5, expand=False)
nos = [r["clause_no"] for r in res]
check("each clause appears only once in the results", len(nos) == len(set(nos)))

r73 = [r for r in res if r["clause_no"] == "7.3"][0]
check("two passages of one clause are merged into its context", r73["n_passages"] == 2 and " " in r73["context"])
check("merged passage reports the full page range", r73["page"] == 12 and r73["page_end"] == 13)
check("context contains both original passages unaltered",
      CLAUSES[2]["text"] in r73["context"] and CLAUSES[3]["text"] in r73["context"])

raw = S.search("margin open space", mode="keyword", k=8, expand=False, dedupe=False)
check("dedupe=False keeps both passages of clause 7.3 as separate results",
      len([r for r in raw if r["clause_no"] == "7.3"]) == 2)

with_titles = S.search("Can I build up to the edge of my land", mode="hybrid", k=3, expand=True, titles=True)
check("heading match helps: a clause whose TITLE says 'margin' ranks in the top 3",
      any(r["clause_no"] in ("7.1", "7.3") for r in with_titles))

failed = results.count(False)
print(f"\n{len(results) - failed}/{len(results)} passed")
sys.exit(1 if failed else 0)
