"""Turn an answer_question() result into Markdown that keeps original text and AI text clearly apart."""
import json
import os
import re

DISCLAIMER = ("Research aid only. Not legal, engineering or official advice. Check the official, "
              "current notification and your local authority before relying on any clause.")


def _load_sources(path="sources.json"):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return []
    return []


def source_info(filename, sources):
    low = filename.lower()
    for s in sources:
        if any(m in low for m in s.get("match", [])):
            return s
    return None


def to_markdown(res, sources_path="sources.json"):
    sources = _load_sources(sources_path)
    info, status = res["info"], res["status"]
    md = []

    if info["concepts"]:
        md.append("### What the system understood")
        for name, definition in info["definitions"]:
            md.append(f"- **Topic:** {name}\n  - *General explanation (not from the regulation):* {definition}")
        if info["terms"]:
            md.append(f"- **Regulatory terms searched as well:** {', '.join(info['terms'])}")
        if info["missing_details"]:
            md.append("- **Details that often change the answer, not given in your question:** "
                      + ", ".join(info["missing_details"]) + ". Check the clauses below for how these matter.")
        md.append("")

    if status == "answered":
        a = res["answer"]
        md.append("### Plain-language answer  \n*AI-generated from the clauses below. It is not the regulation. Check it against the original text.*")
        md.append(f"**{a.get('direct_answer', '')}**")
        md.append(a.get("simple_explanation", ""))
        if a.get("evidence"):
            md.append("**The regulation says (word for word, checked against the clause text):**")
            md += [f"> “{e.get('quote', '')}”  (clause {e.get('clause')})" for e in a["evidence"] if isinstance(e, dict)]
        if a.get("by_clause"):
            md.append("**What each clause says, in simple words:**")
            md += [f"- **Clause {e.get('clause')}:** {e.get('says', '')}" for e in a["by_clause"] if isinstance(e, dict)]
        if a.get("check_next"):
            md.append("**To get an exact answer, find out:**")
            md += [f"- {c}" for c in a["check_next"]]
        if a.get("conditions"):
            md.append("**Conditions and exceptions stated in the clauses:**")
            md += [f"- {c}" for c in a["conditions"]]
        from answer import cited_clauses
        md.append(f"*Clauses used: {', '.join(cited_clauses(a))}. Automatic checks passed: cited clauses exist and every number appears in the clause it is attributed to. This does not prove the clause applies to your situation.*")
    elif status == "unverified":
        md.append("### Plain-language answer withheld")
        md.append("The AI draft failed automatic checks, so it is not shown: " + "; ".join(res["problems"]) + ".")
        md.append("Read the original clauses below.")
    elif status == "insufficient":
        md.append("### Not enough verified information")
        md.append("I could not find sufficient verified information in the currently loaded documents to establish this requirement. "
                  + res.get("reason", ""))
        md.append("It may be covered by a document that has not been loaded. The closest clauses found are shown below only as a starting point, not as an answer.")
    elif status == "evidence_only":
        md.append("### Matching clauses")
        md.append("No AI model is connected, so there is no plain-language summary. The original clauses are shown below.")
    elif status == "llm_error":
        md.append("### AI explanation unavailable")
        md.append("The AI step failed (" + "; ".join(res["problems"]) + "). The original clauses are shown below.")

    md.append("")
    md.append("### Original regulation text (unaltered extraction)")
    if any(re.search(r"\bTable\s*(No\.?\s*)?\d", r.get("context", r["text"])) for r in res["results"]):
        md.append("*Some clauses below contain tables. PDF extraction flattens tables into plain text, so rows and "
                  "columns can be mixed up. Check the original table in the PDF before using any number.*")
    for i, r in enumerate(res["results"], start=1):
        src = source_info(r["source"], sources)
        end = r.get("page_end", r["page"])
        pages = f"page {r['page']}" if end == r["page"] else f"pages {r['page']}-{end}"
        where = f"`{r['source']}`, {pages}"
        if src:
            where = f"{src.get('title', r['source'])} · {src.get('authority', '')} · {pages}"
        md.append(f"**{i}. Clause {r['clause_no']}**  ·  {where}")
        full = r.get("context", r["text"])
        text = full[:1500] + ("…" if len(full) > 1500 else "")
        if r.get("n_passages", 1) > 1:
            text += f"  *(matching passages from this clause, joined; … marks a gap)*"
        md.append("> " + text.replace("\n", " "))
        if src:
            md.append(f"*Version/date: {src.get('version_note', 'not verified')}. Source: {src.get('url', 'not recorded')}*")
        md.append("")

    md.append(f"---\n*{DISCLAIMER}*")
    return "\n\n".join(m for m in md if m is not None)
