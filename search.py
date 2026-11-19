"""The search engine. Modes:

  keyword  - BM25 word matching (fast, exact, but misses different wording)
  semantic - embeddings (matches meaning, can be loosely related)
  hybrid   - both combined with Reciprocal Rank Fusion (RRF)

On top of that, expand=True adds regulatory terms found by concepts.py to the query
(for example "edge of my land" also searches for "margin" and "setback").

Embedder choice:
  "minilm" - sentence-transformers all-MiniLM-L6-v2 (real semantic model; needs internet once to download)
  "onnx"   - the same MiniLM model through fastembed/ONNX (much smaller install; used on free hosting)
  "lsa"    - TF-IDF + SVD fallback (runs anywhere, weaker; used if minilm is unavailable)

assess_evidence() decides whether the retrieved clauses are relevant enough to answer from.
Its thresholds are PROVISIONAL: recalibrate them with the diagnostics printed by eval.py.
"""
import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

from concepts import analyze_query

STOP = set("the a an of to in and or for is are be on at by with as that this it from any shall".split())
GENERIC = set("what how can does need much many have must get who when where which about there their "
              "plot building house land rule rules regulation regulations required allowed".split())

MIN_SEM = 0.30                # no topic recognised: best semantic score must reach this
MIN_SEM_WITH_CONCEPT = 0.20   # topic recognised: a lower bar is enough
MIN_OVERLAP = 0.25            # no topic recognised: share of question words found in top clauses


def stem(t):
    """Very light plural stemming so 'margins' matches 'margin' and 'buildings' matches 'building'."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("sses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text):
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def load_clauses(path="clauses.jsonl"):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class Searcher:
    def __init__(self, clauses, embedder="auto"):
        self.clauses = clauses
        self.docs = [c["title"] + ". " + c["text"] for c in clauses]
        self.bm25 = BM25Okapi([tokenize(c["title"] + " " + d) for c, d in zip(clauses, self.docs)])
        self.title_bm25 = BM25Okapi([tokenize(c["title"]) or ["_"] for c in clauses])
        self.embedder_name, self._embed, self.doc_vecs = self._build_embedder(embedder)

    # ---------- embeddings ----------
    def _build_embedder(self, choice):
        """Order tried by "auto": MiniLM via sentence-transformers, then the same MiniLM model through ONNX (fastembed;
        small enough for free hosting), then a plain LSA fallback. Set EMBED_BACKEND=minilm|onnx|lsa to force one."""
        if choice == "auto":
            choice = os.environ.get("EMBED_BACKEND", "auto")
        if choice in ("auto", "minilm"):
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                vecs = model.encode(self.docs, normalize_embeddings=True, show_progress_bar=False)
                return "minilm", (lambda q: model.encode([q], normalize_embeddings=True)[0]), np.asarray(vecs)
            except Exception as e:  # no package / no internet
                if choice == "minilm":
                    raise
                print(f"[search] sentence-transformers unavailable ({type(e).__name__}); trying ONNX.")
        if choice in ("auto", "onnx"):
            try:
                from fastembed import TextEmbedding

                model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

                def unit(v):
                    v = np.asarray(v, dtype=float)
                    return v / (np.linalg.norm(v) or 1.0)

                vecs = np.array([unit(v) for v in model.embed(self.docs)])
                return "onnx", (lambda q: unit(next(iter(model.embed([q]))))), vecs
            except Exception as e:
                if choice == "onnx":
                    raise
                print(f"[search] ONNX MiniLM unavailable ({type(e).__name__}); using LSA fallback.")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        X = tfidf.fit_transform(self.docs)
        k = max(2, min(128, X.shape[0] - 1, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=0)
        vecs = normalize(svd.fit_transform(X))
        return "lsa", (lambda q: normalize(svd.transform(tfidf.transform([q])))[0]), vecs

    # ---------- query understanding ----------
    def analyze(self, query):
        return analyze_query(query)

    # ---------- search ----------
    def search(self, query, mode="hybrid", k=5, expand=True, titles=None, dedupe=True):
        """Return the top-k results.
        expand : add regulatory terms from the term map to the query
        titles : also rank by clause HEADING match (hybrid mode only). Defaults to the value of expand.
        dedupe : show each clause once (its best passage); other matching passages of the same
                 clause are merged into result['context'] so nothing is silently dropped."""
        titles = expand if titles is None else titles
        qx = analyze_query(query)["expanded"] if expand else query
        bm = self.bm25.get_scores(tokenize(qx))
        sem = self.doc_vecs @ self._embed(query)  # meaning search uses the ORIGINAL question; the term list would dilute it

        if mode == "keyword":
            order = list(np.argsort(-bm))
            scores = bm
        elif mode == "semantic":
            order = list(np.argsort(-sem))
            scores = sem
        else:  # hybrid: Reciprocal Rank Fusion of the rankings
            rankings = [list(np.argsort(-bm)), list(np.argsort(-sem))]
            if titles:
                tq = [t for t in tokenize(qx) if t not in GENERIC and len(t) > 2]
                tit = self.title_bm25.get_scores(tq) if tq else np.zeros(len(self.clauses))
                if tit.max() > 0:
                    rankings.append([i for i in np.argsort(-tit)[:10] if tit[i] > 0])
            fused = {}
            for n, ranking in enumerate(rankings):
                weight = 0.5 if n == 2 else 1.0  # heading match is a weaker vote than keyword/meaning
                for rank, idx in enumerate(ranking[:50]):
                    fused[idx] = fused.get(idx, 0.0) + weight / (60 + rank)
            order = sorted(fused, key=fused.get, reverse=True)
            scores = np.zeros(len(self.clauses))
            for i, s in fused.items():
                scores[i] = s

        if not dedupe:
            return [
                {**self.clauses[i], "context": self.clauses[i]["text"], "n_passages": 1, "score": float(scores[i]),
                 "sem_score": float(sem[i]), "bm25_score": float(bm[i])}
                for i in order[:k]
            ]
        return self._group_by_clause(order, scores, sem, bm, k)

    def _group_by_clause(self, order, scores, sem, bm, k):
        groups, ranked = {}, []
        for i in order[: max(80, k * 3)]:
            c = self.clauses[i]
            key = (c["source"], c["clause_no"])
            if key in groups:
                if len(groups[key]["sib"]) < 2:
                    groups[key]["sib"].append(i)
                continue
            if len(ranked) >= k:
                continue  # only collecting extra passages for clauses already chosen
            groups[key] = {"i": i, "sib": []}
            ranked.append(groups[key])
        out = []
        for g in ranked:
            i = g["i"]
            parts = sorted([self.clauses[i]] + [self.clauses[j] for j in g["sib"]], key=lambda c: c.get("part", 1))
            context, prev = "", None
            for c in parts:
                if prev is not None:
                    context += " " if c.get("part", 1) == prev + 1 else " … "
                context += c["text"]
                prev = c.get("part", 1)
            out.append({
                **self.clauses[i],
                "context": context,
                "n_passages": len(parts),
                "page": min(c["page"] for c in parts),
                "page_end": max(c.get("page_end", c["page"]) for c in parts),
                "score": float(scores[i]), "sem_score": float(sem[i]), "bm25_score": float(bm[i]),
            })
        return out


def assess_evidence(query, info, results, min_sem=MIN_SEM, min_sem_concept=MIN_SEM_WITH_CONCEPT,
                    min_overlap=MIN_OVERLAP):
    """Return (status, reason). status is 'ok' or 'none'. Never proves a clause APPLIES; it only
    says whether the retrieved text is related enough to the question to be worth answering from."""
    if not results:
        return "none", "No clauses were retrieved."
    top_sem = max(r["sem_score"] for r in results)
    if info["concepts"]:
        if top_sem >= min_sem_concept:
            return "ok", "Topic recognised and related clauses found."
        return "none", f"Topic recognised but the closest clauses are only weakly related (score {top_sem:.2f})."
    words = [t for t in tokenize(query) if len(t) > 2 and t not in GENERIC]
    blob = " ".join(r["title"] + " " + r["text"] for r in results[:3]).lower()
    overlap = sum(w in blob for w in words) / max(len(words), 1)
    if top_sem >= min_sem and overlap >= min_overlap:
        return "ok", "Related clauses found."
    return "none", f"No recognised topic, and the closest clauses are weakly related (score {top_sem:.2f}, word overlap {overlap:.0%})."


if __name__ == "__main__":
    import sys

    s = Searcher(load_clauses())
    q = " ".join(sys.argv[1:]) or "front margin for a residential plot"
    print(f"Embedder: {s.embedder_name}")
    info = s.analyze(q)
    print("Topics:", info["concepts"], "| added terms:", info["terms"])
    res = s.search(q)
    for r in res:
        print(f"{r['clause_no']:<10} p.{r['page']:<4} sem={r['sem_score']:.2f}  {r['title'][:70]}")
    print("Evidence check:", assess_evidence(q, info, res))
