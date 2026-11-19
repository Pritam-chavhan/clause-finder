"""Measure the ANSWER step (separate from retrieval).  Needs a Gemini key:  GEMINI_API_KEY=... python eval_answers.py questions.csv

For each question it runs the whole pipeline and counts the outcomes:
  answerable questions   : answered | unverified (AI text withheld by checks) | insufficient (wrongly refused) | llm_error
                           plus: of the answered ones, how often a cited clause matches the clause you marked correct
  unanswerable questions : insufficient (correct refusal) | answered or unverified (the system tried to answer)

What this does NOT measure: whether the plain-language wording is clear or fully faithful. Read 10-15 answers
yourself and score them by hand (see README). Automatic checks catch invented numbers and fake citations only.
"""
import collections
import csv
import os
import sys
import time

from answer import answer_question, make_gemini_llm
from eval import is_hit
from search import Searcher, load_clauses


def run(searcher, rows, llm, delay=0.0):
    stats, details = collections.Counter(), []
    for r in rows:
        unans = r["qtype"] == "unanswerable" or not r["correct_clause"]
        res = answer_question(searcher, r["question"], llm=llm)
        group = "unanswerable" if unans else "answerable"
        stats[(group, res["status"])] += 1
        from answer import cited_clauses
        cited = cited_clauses(res["answer"] or {})
        cite_ok = None
        if not unans and res["status"] == "answered":
            cite_ok = any(is_hit(c, r["correct_clause"]) for c in cited)
            stats["answered_total"] += 1
            stats["cited_expected"] += int(cite_ok)
        details.append((r["qtype"], r["question"], r["correct_clause"], res["status"], ";".join(cited),
                        cite_ok, " | ".join(res["problems"])))
        if delay:
            time.sleep(delay)
    return stats, details


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "questions.csv"
    clauses_path = sys.argv[2] if len(sys.argv) > 2 else "clauses.jsonl"
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first (see README).")
    llm = make_gemini_llm(key)
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("question") or "").strip()]
    for r in rows:
        r["qtype"] = (r.get("qtype") or "unlabeled").strip().lower()
        r["correct_clause"] = (r.get("correct_clause") or "").strip()
    searcher = Searcher(load_clauses(clauses_path))
    stats, details = run(searcher, rows, llm, delay=float(os.environ.get("CALL_DELAY", 4)))

    print(f"Model: {llm.model_name} | questions: {len(rows)}\n")
    for group in ("answerable", "unanswerable"):
        total = sum(v for k, v in stats.items() if isinstance(k, tuple) and k[0] == group)
        print(f"{group} questions ({total}):")
        for status in ("answered", "unverified", "insufficient", "llm_error", "evidence_only"):
            n = stats[(group, status)]
            if n:
                print(f"  {status:<14}{n:>4}  ({n / total:.0%})")
    if stats["answered_total"]:
        print(f"\nOf {stats['answered_total']} answered, a cited clause matched your expected clause in "
              f"{stats['cited_expected']} ({stats['cited_expected'] / stats['answered_total']:.0%}).")
    with open("answer_eval_details.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qtype", "question", "expected", "status", "cited", "cited_expected", "problems"])
        w.writerows(details)
    print("Per-question results saved to answer_eval_details.csv. Now read 10-15 of the answers yourself.")


if __name__ == "__main__":
    main()
