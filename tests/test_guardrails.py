"""Tests for the answer guardrails, using FAKE model outputs (no API key needed).

Run from the project folder:   python tests/test_guardrails.py
They use the invented sample regulations in tests/sample_regulations.txt, never real law.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from functools import partial  # noqa: E402

from answer import answer_question as _aq, verify_answer  # noqa: E402
from ingest import chunk_long, read_pages, split_clauses  # noqa: E402
from render import to_markdown  # noqa: E402
from search import Searcher  # noqa: E402

pages = read_pages(ROOT / "tests" / "sample_regulations.txt")
CLAUSES = chunk_long(split_clauses(pages, "sample_regulations.txt"))
S = Searcher(CLAUSES, embedder="lsa")
# These tests check the GUARDRAILS, not ranking quality. The invented sample needs the term map to retrieve
# its clauses, so the term map is switched on here. The shipped default (expand=False) was chosen from evaluation.
answer_question = partial(_aq, expand=True, judge=False)

Q = "Can I build my house right up to the edge of my land"


def fake(payload):
    return lambda prompt: payload if isinstance(payload, str) else json.dumps(payload)


GOOD = {"insufficient": False,
        "direct_answer": "No. A residential plot must leave open space at the front, and the clauses give 3.0 metres or 4.5 metres depending on plot size.",
        "simple_explanation": "Plots up to 100 square metres leave 3.0 metres at the front; larger plots leave 4.5 metres.",
        "evidence": [{"clause": "3.1", "quote": "For plots up to 100 square metres the front margin is 3.0 metres"}],
        "by_clause": [{"clause": "3.1", "says": "Front open space depends on the plot size."},
                      {"clause": "3.3", "says": "Corner plots provide the front margin on both road sides."}],
        "check_next": ["your plot area in sq.mt"]}

results = []


def check(name, cond):
    results.append((name, cond))
    print(("PASS  " if cond else "FAIL  ") + name)


# 1 good answer passes
r = answer_question(S, Q, llm=fake(GOOD))
check("good answer is accepted", r["status"] == "answered")

# 2 invented number is caught
bad = dict(GOOD, direct_answer="No. You must leave 6 metres at the front.")
r = answer_question(S, Q, llm=fake(bad))
check("invented number (6) is withheld", r["status"] == "unverified" and any("6" in p for p in r["problems"]))

# 3 fake clause number is caught
bad = dict(GOOD, by_clause=GOOD["by_clause"] + [{"clause": "12.4", "says": "Invented."}])
r = answer_question(S, Q, llm=fake(bad))
check("citation to a clause that was not retrieved is withheld", r["status"] == "unverified")

# 4 no citation is caught
bad = dict(GOOD, by_clause=[], evidence=[])
r = answer_question(S, Q, llm=fake(bad))
check("answer without any citation is withheld", r["status"] == "unverified")

# 5 spelled-out number not in evidence is caught
bad = dict(GOOD, simple_explanation="Leave about three metres.")
r = answer_question(S, Q, llm=fake(bad))
check("spelled-out number not in clauses is withheld", r["status"] == "unverified")

# 5b a number that is in ANOTHER cited clause but attributed to the wrong clause is caught
bad = dict(GOOD, by_clause=[GOOD["by_clause"][0], {"clause": "3.3", "says": "Corner plots leave 4.5 metres on both road sides."}])
r = answer_question(S, Q, llm=fake(bad))
check("number attributed to the wrong clause is withheld", r["status"] == "unverified")

# 5c a made-up quote is caught
bad = dict(GOOD, evidence=[{"clause": "3.1", "quote": "Every plot may be built up to the boundary without any margin"}])
r = answer_question(S, Q, llm=fake(bad))
check("made-up quote is withheld", r["status"] == "unverified" and any("quote" in p for p in r["problems"]))

# 5d an answer without any quote is caught
r = answer_question(S, Q, llm=fake(dict(GOOD, evidence=[])))
check("answer without a supporting quote is withheld", r["status"] == "unverified")

# 6 model says insufficient
r = answer_question(S, Q, llm=fake({"insufficient": True, "direct_answer": "", "simple_explanation": "",
                                    "by_clause": [], "check_next": []}))
check("model 'insufficient' becomes a refusal", r["status"] == "insufficient")

# 7 broken JSON / API failure do not crash
r = answer_question(S, Q, llm=fake("this is not json"))
check("bad JSON is handled as llm_error", r["status"] == "llm_error")


def boom(prompt):
    raise RuntimeError("quota exceeded")


r = answer_question(S, Q, llm=boom)
check("API failure is handled as llm_error", r["status"] == "llm_error")

# 8 unrelated question is refused WITHOUT calling the model
called = []
r = answer_question(S, "What is the income tax rate for salaried employees", llm=lambda p: called.append(1) or "{}")
check("unrelated question is refused before calling the model", r["status"] == "insufficient" and not called)

# 9 no model connected
r = answer_question(S, Q, llm=None)
check("no model -> evidence only", r["status"] == "evidence_only")

# 10 code-fenced JSON is accepted
r = answer_question(S, Q, llm=fake("```json\n" + json.dumps(GOOD) + "\n```"))
check("JSON inside code fences is parsed", r["status"] == "answered")

# 11 rendering keeps original text and AI text apart, for every status
labels_ok = True
for llm in (fake(GOOD), fake(dict(GOOD, by_clause=[], evidence=[])), None, boom):
    md = to_markdown(answer_question(S, Q, llm=llm), sources_path=str(ROOT / "sources.json"))
    labels_ok &= "Original regulation text" in md and "Research aid only" in md
md = to_markdown(answer_question(S, Q, llm=fake(GOOD)), sources_path=str(ROOT / "sources.json"))
labels_ok &= "AI-generated" in md
check("rendered page labels AI text vs original text", labels_ok)

# 12 table warning appears when a shown clause mentions a table
table_clauses = [{"source": "t.pdf", "clause_no": "6.7.1", "title": "Road side margin", "page": 34, "page_end": 35,
                  "text": "Road side margin shall be as in Table 6.24: Road Width 7.5 9.0 12.0 margins 3.0 4.5 6.0",
                  "context": "Road side margin shall be as in Table 6.24: Road Width 7.5 9.0 12.0 margins 3.0 4.5 6.0",
                  "n_passages": 1, "sem_score": 0.5, "score": 1.0, "bm25_score": 1.0, "part": 1, "id": 0}]
fake_res = {"question": "q", "info": {"concepts": [], "definitions": [], "terms": [], "missing_details": []},
            "results": table_clauses, "status": "evidence_only", "reason": "", "answer": None, "problems": []}
md = to_markdown(fake_res, sources_path=str(ROOT / "sources.json"))
check("table warning is shown when a clause contains a table", "flattens tables" in md)
fake_res["results"] = [dict(table_clauses[0], text="No tables here, just prose.", context="No tables here, just prose.")]
check("no table warning when no clause has a table", "flattens tables" not in to_markdown(fake_res, sources_path=str(ROOT / "sources.json")))

failed = [n for n, ok in results if not ok]

# ---- second-pass relevance judge
from answer import judge_answer  # noqa: E402
_aq_j = partial(_aq, expand=True, judge=True)


def two_step(answer_payload, judge_payload):
    calls = []

    def f(prompt):
        calls.append(prompt)
        return json.dumps(judge_payload if "strict reviewer" in prompt else answer_payload)
    f.calls = calls
    return f


f = two_step(GOOD, {"supported": True, "applies": True, "reason": ""})
r = _aq_j(S, Q, llm=f)
check("judge says yes -> answered", r["status"] == "answered" and len(f.calls) == 2)
f = two_step(GOOD, {"supported": True, "applies": False, "reason": "special case only"})
r = _aq_j(S, Q, llm=f)
check("judge says clause does not apply -> refused, no answer shown", r["status"] == "insufficient" and r["answer"] is None)
f = two_step(GOOD, {"supported": False, "applies": True, "reason": "x"})
check("judge says unsupported -> refused", _aq_j(S, Q, llm=f)["status"] == "insufficient")
check("judge call failing -> refused (never shows unchecked answer)",
      _aq_j(S, Q, llm=lambda p: json.dumps(GOOD) if "strict reviewer" not in p else "not json")["status"] == "insufficient")
f = two_step(dict(GOOD, evidence=[]), {"supported": True, "applies": True})
_aq_j(S, Q, llm=f)
check("judge is not even called when code checks already failed", len(f.calls) == 1)

check("model returns a JSON list with junk -> llm_error, no crash",
      _aq_j(S, Q, llm=lambda p: "[1, 2]")["status"] == "llm_error")
check("model wraps the object in a one-item list -> still handled",
      _aq(S, Q, llm=lambda p: json.dumps([GOOD]), expand=True, judge=False)["status"] == "answered")

SPLIT = [{"clause_no": "9.9", "text": "Ancillary uses shall have maximum permissible he ight of 3.0 mt s in the side margin."}]
ANS = {"insufficient": False, "direct_answer": "Yes.", "simple_explanation": "Up to 3.0 metres high.", "evidence": [{"clause": "9.9", "quote": "maximum permissible height of 3.0 mts in the side margin"}], "by_clause": [{"clause": "9.9", "says": "Limit of 3.0 metres in height."}]}
check("quote still matches when the PDF split a word ('he ight')", verify_answer(ANS, SPLIT) == [])
BAD = dict(ANS, evidence=[{"clause": "9.9", "quote": "maximum permissible height of 4.0 mts in the side margin"}])
check("a quote with a changed number is still rejected", any("quote" in p for p in verify_answer(BAD, SPLIT)))

print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
