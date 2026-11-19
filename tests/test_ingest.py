"""Tests for the ingestion fixes, built from defects seen in the real CGDCR Part II extraction:
  1. a contents-page entry was kept as a clause
  2. long clauses were cut in the middle of words
  3. every piece of a long clause claimed the page where the clause STARTED
The text below is a short excerpt-style imitation, not a full copy of the regulation.

Run from the project folder:   python tests/test_ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import chunk_long, split_clauses  # noqa: E402

SENT = ("Different uses permissible in various Use Zone are specified in the table below and shall be "
        "regulated according to the width of the road on which the building unit abuts; ")

toc = ["CONTENTS"] + [f"6.{i} Section heading number {i} {10 + i}" for i in range(1, 12)] + [
    "11.2 Communication Infrastructure",
    "Telecommunication infrastructure/ microwave communication tower 67",
    "TABLES",
    "TABLE 6.27: REAR AND SIDE MARGINS FOR BUILDING UNIT UP TO 500 SQ.MT",
]
page9 = ["6.1 Use Zone Classification & Permissible Uses"] + [SENT * 2] * 6
page10 = [SENT * 2] * 4 + ["6.2 Floor Space Index", "The floor space index for residential use depends on the zone and the road width."]
pages = [(6, "\n".join(toc)), (9, "\n".join(page9)), (10, "\n".join(page10))]

stats = {}
clauses = split_clauses(pages, "test.pdf", chapters={6, 7, 8, 9, 10, 11}, stats=stats)
chunks = chunk_long(clauses)
results = []


def check(name, cond):
    results.append(cond)
    print(("PASS  " if cond else "FAIL  ") + name)


check("contents page is skipped", stats.get("toc_pages") == 1 and not any(c["clause_no"] == "11.2" for c in chunks))

c61 = [c for c in chunks if c["clause_no"] == "6.1"]
check("long clause is split into several pieces", len(c61) >= 3)

whole = " ".join(c["text"] for c in c61)
original = [c for c in clauses if c["clause_no"] == "6.1"][0]["text"]
check("no text is lost or altered when splitting", whole == original)

mid_word = False
words = set(original.split())
for c in c61:
    if c["text"].split()[0] not in words or c["text"].split()[-1] not in words:
        mid_word = True
check("pieces never start or end in the middle of a word", not mid_word)

check("each piece is within the size limit", all(len(c["text"]) <= 1300 for c in c61))
check("first piece starts on page 9", c61[0]["page"] == 9)
check("a piece that reaches page 10 reports it", any(c["page_end"] == 10 for c in c61) and c61[-1]["page_end"] == 10)
check("pieces after the page break start on page 10", any(c["page"] == 10 for c in c61))
check("next clause (6.2) is still found, on page 10", any(c["clause_no"] == "6.2" and c["page"] == 10 for c in chunks))

# table fragments seen in the real extraction ("6.00 mtr. for D1-RUDA.", "10.00 mts. In case of hollow plinth ...")
frag_pages = [(34, "\n".join([
    "6.7.1 Road side margin",
    "The road side margin shall be kept open to sky as per the table below for each road width.",
    "6.00 mtr. for D1-RUDA.",
    "10.00 mts. In case of hollow plinth provided at ground floor for parking the margin may reduce.",
    "6.7.2 Rear & Other Side Margins",
    "Rear and side margins shall be provided as given in the tables that follow below.",
]))]
frag = chunk_long(split_clauses(frag_pages, "t.pdf", chapters={6}))
check("table fragments are not treated as clauses", sorted({c["clause_no"] for c in frag}) == ["6.7.1", "6.7.2"])
check("the fragment text stays inside the clause it belongs to",
      any("6.00 mtr. for D1-RUDA" in c["text"] for c in frag if c["clause_no"] == "6.7.1"))
check("real clause numbers ending in 0 (6.10, 6.20) are still accepted",
      {c["clause_no"] for c in chunk_long(split_clauses([(40, "6.10 Addition in building units with existing Buildings\nSome body text that is long enough to keep.\n6.20 Discrepancy Removal\nMore body text that is long enough to keep here.")], "t.pdf"))} == {"6.10", "6.20"})

failed = results.count(False)
print(f"\n{len(results) - failed}/{len(results)} passed")
sys.exit(1 if failed else 0)
