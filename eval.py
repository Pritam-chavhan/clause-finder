"""Measure retrieval quality honestly.   Usage:  python eval.py questions.csv [clauses.jsonl]

questions.csv columns:  question, correct_clause, qtype
  qtype is one of:  exact | everyday | mismatch | ambiguous | unanswerable
  correct_clause: the clause number you verified in the PDF. Leave it EMPTY for unanswerable questions
  (questions the loaded document cannot answer). A result is correct if its clause number equals the
  expected one or sits underneath it (expected 6.2 matches 6.2.1).

What it reports
  1. Retrieval on answerable questions: hit@5, hit@1, MRR for
        keyword | semantic | hybrid (raw question) | hybrid + term map (the app's real setting)
  2. hit@5 per question type, raw hybrid vs hybrid + term map (does understanding everyday wording help?)
  3. Refusal behaviour of the evidence check:
        answerable questions wrongly refused, unanswerable questions correctly refused
  4. Score diagnostics to help you recalibrate the thresholds in search.py

It measures RETRIEVAL only. Answer-writing quality is a separate test (see README).
Your test set is small and written by you, so treat results as indicative, not as a benchmark.
"""
import csv
import statistics
import sys
from collections import defaultdict

from search import MIN_OVERLAP, MIN_SEM, MIN_SEM_WITH_CONCEPT, Searcher, assess_evidence, load_clauses


def is_hit(clause_no, expected):
    return clause_no == expected or clause_no.startswith(expected + ".")


def rank_of(results, expected):
    for i, r in enumerate(results, start=1):
        if is_hit(r["clause_no"], expected):
            return i
    return None


def metrics(ranks):
    n = len(ranks) or 1
    return (sum(r is not None for r in ranks) / n,
            sum(r == 1 for r in ranks) / n,
            sum(1 / r for r in ranks if r) / n)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "questions.csv"
    clauses_path = sys.argv[2] if len(sys.argv) > 2 else "clauses.jsonl"
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("question") or "").strip()]
    for r in rows:
        r["qtype"] = (r.get("qtype") or "unlabeled").strip().lower()
        r["correct_clause"] = (r.get("correct_clause") or "").strip()
    answerable = [r for r in rows if r["correct_clause"] and r["qtype"] != "unanswerable"]
    unanswerable = [r for r in rows if r["qtype"] == "unanswerable" or not r["correct_clause"]]

    s = Searcher(load_clauses(clauses_path))
    print(f"Embedder: {s.embedder_name} | answerable: {len(answerable)} | unanswerable: {len(unanswerable)}\n")

    FINAL = "hybrid + term map + heading match"   # the app's real setting
    setups = [("keyword", "keyword", False, False), ("semantic", "semantic", False, False),
              ("hybrid (raw question)", "hybrid", False, False), ("hybrid + term map", "hybrid", True, False),
              (FINAL, "hybrid", True, True)]
    ranks = {name: [] for name, _, _, _ in setups}
    details = []
    for r in answerable:
        for name, mode, expand, titles in setups:
            rk = rank_of(s.search(r["question"], mode=mode, k=5, expand=expand, titles=titles), r["correct_clause"])
            ranks[name].append(rk)
            details.append((name, r["qtype"], r["question"], r["correct_clause"], rk))

    print("1. Retrieval on answerable questions (each clause counted once)")
    print(f"{'setup':<36}{'hit@5':>8}{'hit@1':>8}{'MRR':>8}")
    for name, _, _, _ in setups:
        h5, h1, mrr = metrics(ranks[name])
        print(f"{name:<36}{h5:>8.0%}{h1:>8.0%}{mrr:>8.2f}")

    print("\n2. hit@5 by question type (raw hybrid -> final setting)")
    by_type = defaultdict(list)
    for i, r in enumerate(answerable):
        by_type[r["qtype"]].append((ranks["hybrid (raw question)"][i], ranks[FINAL][i]))
    print(f"{'type':<14}{'n':>4}{'raw':>8}{'final':>8}")
    for t, pairs in sorted(by_type.items()):
        raw = sum(p[0] is not None for p in pairs) / len(pairs)
        mp = sum(p[1] is not None for p in pairs) / len(pairs)
        print(f"{t:<14}{len(pairs):>4}{raw:>8.0%}{mp:>8.0%}")

    print("\n3. Evidence check (refusal behaviour, before any AI step)")
    wrongly_refused, sem_ok, sem_bad = [], [], []
    for r in answerable:
        res = s.search(r["question"], k=5)
        info = s.analyze(r["question"])
        status, _ = assess_evidence(r["question"], info, res)
        sem_ok.append(max(x["sem_score"] for x in res))
        if status == "none":
            wrongly_refused.append(r["question"])
    correctly_refused, passed_through = [], []
    for r in unanswerable:
        res = s.search(r["question"], k=5)
        info = s.analyze(r["question"])
        status, _ = assess_evidence(r["question"], info, res)
        sem_bad.append(max(x["sem_score"] for x in res))
        (correctly_refused if status == "none" else passed_through).append(r["question"])
    print(f"answerable questions wrongly refused : {len(wrongly_refused)}/{len(answerable)}")
    for q in wrongly_refused:
        print("    -", q)
    if unanswerable:
        print(f"unanswerable questions refused       : {len(correctly_refused)}/{len(unanswerable)}")
        print("  (the ones that passed through must be caught by the AI step saying 'insufficient')")
        for q in passed_through:
            print("    passed through:", q)

    print("\n4. Score diagnostics (best semantic score per question)")
    if sem_ok:
        print(f"answerable   : min {min(sem_ok):.2f}  median {statistics.median(sem_ok):.2f}")
    if sem_bad:
        print(f"unanswerable : max {max(sem_bad):.2f}  median {statistics.median(sem_bad):.2f}")
    print(f"current thresholds: MIN_SEM={MIN_SEM}, MIN_SEM_WITH_CONCEPT={MIN_SEM_WITH_CONCEPT}, MIN_OVERLAP={MIN_OVERLAP}")
    print("If answerable questions are wrongly refused, lower the thresholds in search.py; if unanswerable ones pass, raise them.")

    with open("eval_details.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["setup", "qtype", "question", "expected_clause", "rank_of_correct (blank = not in top 5)"])
        w.writerows(details)
    print("\nPer-question results saved to eval_details.csv")


if __name__ == "__main__":
    main()
