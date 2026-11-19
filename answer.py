"""Plain-language answers that are grounded in the retrieved clauses and CHECKED afterwards.

Pipeline:
  1. understand the question (concepts.py) and retrieve clauses (search.py)
  2. if the evidence looks too weak -> refuse (status "insufficient")
  3. if no AI model is available    -> show clauses only (status "evidence_only")
  4. otherwise ask the model for a JSON answer using ONLY the retrieved clauses
  5. VERIFY the answer in code. If any check fails, the AI text is withheld (status "unverified")

The checks (verify_answer) are what stop a confident wrong number from reaching the user:
  - every cited clause number must be one of the retrieved clauses
  - every number in the answer must appear in the cited clauses' text
  - every spelled-out number (two, three...) must appear in the cited clauses' text
  - an answer with no citations is rejected unless the model said "insufficient"

These checks reduce risk. They do NOT prove the answer is correct or that a clause applies to
a particular plot. The interface says so.
"""
import json
import os
import re
import time

import requests

from search import assess_evidence

NUM_RE = re.compile(r"\d+(?:\.\d+)*")
NUMBER_WORDS = ("one two three four five six seven eight nine ten eleven twelve fifteen twenty thirty "
                "fifty hundred thousand half double triple").split()

PROMPT = """You explain building regulations to a person with no technical background.
Use ONLY the numbered clauses below. Do not use outside knowledge. Do not guess.

QUESTION: {question}

CLAUSES:
{clauses}

First decide: does any clause DIRECTLY answer the subject of the question? A clause that is only about a related topic
(for example a clause about heritage areas, internal roads or chawls when the question is about an ordinary house) does NOT
count. If no clause directly answers it, set "insufficient" to true, leave the other fields empty, and stop.

Ignore clauses that apply only to a special case (basement, heritage core area, fuelling or electric vehicle station, land-locked
plot, industrial or commercial use, redevelopment, core walled city) unless the question itself mentions that case. If only
special-case clauses are available, set "insufficient" to true.

If a clause does answer it:
- "evidence": one to three entries. Each has the clause number and a "quote" copied WORD FOR WORD from that clause (at most
  30 words) that directly answers the question. Do not change, shorten inside, or paraphrase the quote.
- "direct_answer": ONE or TWO plain sentences. State the actual rule first (what is required, allowed or not allowed), using
  what the clause says. Only after that, say what changes the exact figure. Do NOT start with "It depends" unless the clauses
  really give no rule that applies in every case; even then say first what the clauses require in all cases.
- "simple_explanation": three to five short sentences in everyday words saying what this means for the person.
- "by_clause": one entry for EACH clause you used. "says" explains that clause in simple words, one to three sentences. Give
  a number only if it is written in that same clause's text.
- If a clause gives its values in a table, say what the table is about and which factors choose the row (plot area, road
  width, building height, use) and tell the user to read that table for the exact figure. Tables are flattened, so you cannot
  tell which number belongs to which row: do NOT pick a value from a table. Numbers in ordinary sentences can be stated.
- "check_next": short list of facts the person should find out to get an exact answer (for example "your plot area in sq.mt").
- Never state a number, measurement or percentage unless it appears in the clause you attribute it to.

Return ONLY JSON with these keys:
{{"insufficient": false,
  "evidence": [{{"clause": "6.7.2", "quote": "exact words from the clause"}}],
  "direct_answer": "...",
  "simple_explanation": "...",
  "by_clause": [{{"clause": "6.7.2", "says": "..."}}],
  "check_next": ["..."]}}"""


JUDGE_PROMPT = """You are a strict reviewer of an answer about building regulations.

QUESTION: {question}

TEXT OF THE CLAUSES THE ANSWER RELIES ON (from the regulation):
{quotes}

DRAFT ANSWER: {answer}

Check two things, using ONLY the clause text above:
1. "supported": the statements in the draft answer follow from the clause text above (it may explain the clause in simpler
   words; it must not add rules or figures that the text does not contain).
2. "applies": the clause text is about the SAME situation the question asks about. Answer false if the quotes are about a
   different subject (for example a different structure than the one asked about, such as a compound wall versus a partition
   wall), a special case (heritage area, fuelling station, land-locked plot, redevelopment) when the question is about an
   ordinary house or plot, or a different quantity or unit than the question mentions.
Return ONLY JSON: {{"supported": true, "applies": true, "reason": "one short sentence"}}"""


# ---------------------------------------------------------------- model access
def list_gemini_models(api_key):
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     params={"key": api_key, "pageSize": 200}, timeout=30)
    r.raise_for_status()
    return [m["name"].split("/")[-1] for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]


def pick_gemini_model(api_key):
    """Choose a current general-purpose Flash model from what this key can actually use."""
    if os.environ.get("GEMINI_MODEL"):
        return os.environ["GEMINI_MODEL"]
    names = list_gemini_models(api_key)
    bad = ("lite", "preview", "exp", "image", "tts", "live", "thinking", "audio", "embedding", "vision", "latest")
    flash = sorted([n for n in names if "flash" in n and not any(b in n for b in bad)], reverse=True)
    if flash:
        return flash[0]
    raise RuntimeError("No Flash model found for this key. Models available: " + ", ".join(names[:15]))


def make_gemini_llm(api_key):
    """Return a function prompt -> text, or raise a clear error."""
    model = pick_gemini_model(api_key)

    def llm(prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
        last = None
        for attempt in range(int(os.environ.get("GEMINI_RETRIES", "5"))):  # retry timeouts and temporary overload (429/500/503)
            try:
                r = requests.post(url, params={"key": api_key}, json=body, timeout=int(os.environ.get("GEMINI_TIMEOUT", "150")))
            except requests.exceptions.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
                last = f"Gemini API error {r.status_code}: {r.text[:300]}"
                if r.status_code not in (429, 500, 503):
                    break
            time.sleep(min(8 * (2 ** attempt), 60))
        raise RuntimeError(last)

    llm.model_name = model
    return llm


# ---------------------------------------------------------------- verification
def _norm_num(n):
    n = n.replace(",", "")
    try:
        return "%g" % float(n)
    except ValueError:
        return n


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def cited_clauses(answer):
    """Clause numbers the answer relies on: by_clause entries plus any legacy 'cited' list."""
    out = [str(e.get("clause", "")).strip() for e in answer.get("by_clause", []) if isinstance(e, dict)]
    out += [str(e.get("clause", "")).strip() for e in answer.get("evidence", []) if isinstance(e, dict)]
    out += [str(x).strip() for x in answer.get("cited", [])]
    return [c for i, c in enumerate(out) if c and c not in out[:i]]


def _number_problems(text, allowed, evidence_low, label):
    problems = []
    for n in NUM_RE.findall(str(text).replace(",", "")):
        if _norm_num(n) not in allowed:
            problems.append(f"{label} mentions the number {n}, which is not in the cited clause text")
    for w in NUMBER_WORDS:
        if re.search(rf"\b{w}\b", str(text).lower()) and not re.search(rf"\b{w}\b", evidence_low):
            problems.append(f"{label} mentions '{w}', which is not in the cited clause text")
    return problems


def verify_answer(answer, clauses):
    """Return a list of problems. Empty list = all checks passed."""
    problems = []
    by_no = {}
    for c in clauses:
        by_no.setdefault(c["clause_no"], []).append(c.get("context", c["text"])[:1800])
    cited = cited_clauses(answer)

    unknown = [c for c in cited if c not in by_no]
    if unknown:
        problems.append("cites clauses that were not retrieved: " + ", ".join(unknown))
    if not cited and not answer.get("insufficient"):
        problems.append("gives an answer without citing any clause")

    def allowed_for(nos):
        ev = " ".join(" ".join(by_no[c]) for c in nos if c in by_no)
        nums = {_norm_num(n) for n in NUM_RE.findall(ev.replace(",", ""))}
        nums |= {_norm_num(x) for x in nos} | {_norm_num(x) for x in by_no}
        return nums, ev.lower()

    # every answer must carry at least one quote that really appears, word for word, in the clause it names
    def _norm(t):
        return re.sub(r"[^a-z0-9.%]+", " ", str(t).lower()).strip()

    good_quotes = 0
    for e in answer.get("evidence", []):
        if not isinstance(e, dict):
            continue
        no, quote = str(e.get("clause", "")).strip(), _norm(e.get("quote", ""))
        text = _norm(" ".join(c.get("context", c["text"]) for c in clauses if c["clause_no"] == no))
        if len(quote.split()) >= 4 and quote.replace(" ", "") in text.replace(" ", ""):
            good_quotes += 1
        else:
            problems.append(f"quote from clause {no} does not appear in that clause")
    if not answer.get("insufficient") and good_quotes == 0:
        problems.append("no word-for-word quote from a retrieved clause supports the answer")

    # overall fields may use numbers from any cited clause
    allowed_all, ev_all = allowed_for(cited)
    for label, f in [("the direct answer", answer.get("direct_answer", "")),
                     ("the explanation", answer.get("simple_explanation", ""))] + \
                    [("a 'check next' item", x) for x in answer.get("check_next", [])] + \
                    [("a condition", x) for x in answer.get("conditions", [])]:
        problems += _number_problems(f, allowed_all, ev_all, label)

    # each per-clause explanation may only use numbers from ITS OWN clause
    for e in answer.get("by_clause", []):
        if not isinstance(e, dict):
            continue
        no = str(e.get("clause", "")).strip()
        allowed_one, ev_one = allowed_for([no])
        problems += _number_problems(e.get("says", ""), allowed_one, ev_one, f"the explanation of clause {no}")
    return sorted(set(problems))


# ---------------------------------------------------------------- main entry
def judge_answer(llm, question, answer, results=()):
    """Second, independent model call that sees only the question, the verified quotes and the draft answer.
    Returns (ok, reason). A failed call counts as NOT ok: we prefer withholding over showing an unchecked answer."""
    nos = set(cited_clauses(answer))
    quotes = "\n".join(f"[{r['clause_no']}] {r['title']}: {r.get('context', r['text'])[:1800]}"
                       for r in results if r["clause_no"] in nos)
    try:
        j = parse_json(llm(JUDGE_PROMPT.format(question=question, quotes=quotes,
                                               answer=answer.get("direct_answer", ""))))
    except Exception as e:
        return False, f"relevance check could not run ({type(e).__name__})"
    if j.get("supported") is True and j.get("applies") is True:
        return True, ""
    return False, str(j.get("reason", "the quotes do not answer this question"))[:200]


def answer_question(searcher, question, llm=None, k=5, expand=True, judge=True):  # term map + down-weighted heading match; chosen using 5 independent real questions (see README)
    info = searcher.analyze(question)
    results = searcher.search(question, mode="hybrid", k=k, expand=expand)
    status, reason = assess_evidence(question, info, results)
    out = {"question": question, "info": info, "results": results, "reason": reason,
           "answer": None, "problems": [], "status": None, "model": None}

    if status == "none":
        out["status"] = "insufficient"
        return out
    if llm is None:
        out["status"] = "evidence_only"
        return out

    clause_text = "\n".join(
        f"[{r['clause_no']}] (page {r['page']}) {r['title']}: {r.get('context', r['text'])[:1800]}" for r in results)
    try:
        raw = llm(PROMPT.format(question=question, clauses=clause_text))
        answer = parse_json(raw)
        if isinstance(answer, list) and len(answer) == 1:  # model sometimes wraps the object in a list
            answer = answer[0]
        if not isinstance(answer, dict):
            raise ValueError("model did not return a JSON object")
    except Exception as e:  # network, quota, bad JSON
        out["status"] = "llm_error"
        out["problems"] = [f"{type(e).__name__}: {e}"]
        return out

    out["model"] = getattr(llm, "model_name", None)
    if answer.get("insufficient"):
        out["status"] = "insufficient"
        out["reason"] = "The retrieved clauses do not clearly answer this question."
        return out

    problems = verify_answer(answer, results)
    if not problems and judge:
        ok, why = judge_answer(llm, question, answer, results)
        if not ok:
            out["status"] = "insufficient"
            out["reason"] = "The clauses found do not clearly answer this exact question (" + why + ")"
            return out
    out["answer"] = answer
    out["problems"] = problems
    out["status"] = "unverified" if problems else "answered"
    return out
