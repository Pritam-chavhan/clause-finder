# Clause finder: plain-language search over Gujarat building regulations

Ask a question the way you would ask a person ("Can I build right up to the edge of my land?"). The tool finds the relevant clauses, explains them in simple language, and shows the **original text, page and source** so you can check it.

**Status: working prototype, not validated with real users. Research aid only, not legal or engineering advice.**

## What it does

1. **Understands everyday wording.** A hand-written map (`concepts.py`) links everyday phrases to regulatory terms (for example "edge of my land" -> margin, setback), fixes common spelling mistakes, and lists details that usually change the answer (plot size, road width and so on).
2. **Finds clauses** with hybrid retrieval: BM25 keyword search plus MiniLM embeddings, combined with Reciprocal Rank Fusion.
3. **Refuses when evidence is weak.** If nothing relevant is found, it says so instead of guessing.
4. **Writes a plain-language answer using only the retrieved clauses** (Gemini), then **checks it in code**: every quote must appear word for word in the clause it names, every number must appear in the clause it is attributed to, and every cited clause must have been retrieved. If a check fails, the AI text is withheld.
5. **Gets a second opinion.** A separate model call reads the question, the cited clause text and the draft, and says whether the clauses really answer *this* question (not a special case such as a basement, heritage area or fuelling station). If not, the tool says it cannot answer.
6. **Keeps original and AI text visibly separate** in the interface.

## Files

| Path | Purpose |
|---|---|
| `notebooks/clause_finder_colab.ipynb` | the whole project as a Colab notebook: run top to bottom |
| `ingest.py` | PDF -> clauses (skips contents pages, headers, table rows; `--chapters` filter) |
| `concepts.py` | everyday-to-regulatory term map, spelling fixes, missing-detail list |
| `search.py` | keyword, semantic and hybrid retrieval; evidence check |
| `answer.py` | model call, answer verification, second-opinion check, refusal logic |
| `render.py` | builds the page; separates original text from AI text |
| `eval.py` | retrieval evaluation by question type, with and without the term map |
| `eval_answers.py` | answer-step evaluation (needs a Gemini key) |
| `app.py` | Streamlit web demo (rate-limited, key from secrets) |
| `clauses.jsonl` | the extracted clauses the demo searches |
| `questions_part2.csv`, `questions_user20.csv` | the 45 + 20 evaluation questions with expected clauses |
| `sources.json` | document title, authority, URL for each source |
| `tests/` | 24 guardrail tests with fake model outputs, plus invented sample data |
| `docs/` | how it works, evaluation protocol, one real example answer, case note |

## Live demo

A hosted demo runs on Streamlit Community Cloud (link in the repository description). It limits the AI step per visitor and per hour; past the limit it shows the original clauses without the AI summary. The free host sleeps when unused, so the first visit can take about a minute. The app uses the same MiniLM model through ONNX (`fastembed`) so it fits the free tier; on the 38 answerable questions in the evaluation set it returned the identical top-5 clauses as the sentence-transformers version.

## Run it (Colab is easiest: use the notebook)

```
pip install -r requirements.txt
GEMINI_API_KEY=your_key streamlit run app.py   # uses the included clauses.jsonl
python tests/test_guardrails.py        # 24 guardrail checks, no key needed
python eval.py questions.csv           # retrieval results
GEMINI_API_KEY=your_key python eval_answers.py questions.csv
```

## How to evaluate honestly

- Write 30-40 questions **yourself, from the PDF, before you look at `concepts.py`**. Mix: exact wording, everyday wording, mismatched wording, ambiguous, and questions the document cannot answer.
- Mark the correct clause by reading the PDF. Leave it empty for unanswerable questions.
- Tune `concepts.py` using only half the questions. Report numbers from the other half.
- Report the dataset size next to every number. A small self-written set is **indicative, not a benchmark**.
- Read 10-15 generated answers yourself and score clarity and faithfulness by hand. Automatic checks only catch invented numbers and fake citations.

## Results

All numbers come from real runs on Part II of CGDCR 2017 (76 pages, 97 clauses, 190 text chunks) with the MiniLM embedding model. The 45 test questions were **written by the author** (`questions_part2.csv`), after the term map existed, so they are **indicative, not a benchmark**. The set was split in two: even rows for tuning, odd rows kept unseen until the final run. hit@5 means the correct clause is among the top 5 results. The question list and every setting are in this repository.

**Held-out half (19 answerable questions)**

| Setup | hit@5 | hit@1 | MRR |
|---|---|---|---|
| keyword (BM25) | 63% | 37% | 0.45 |
| semantic (MiniLM) | 63% | 42% | 0.51 |
| hybrid, raw question | 68% | 53% | 0.60 |
| hybrid + term map | 84% | 47% | 0.61 |
| hybrid + term map + heading match | 79% | 53% | 0.63 |

**Tuning half (19 answerable questions):** keyword 63%, semantic 89%, hybrid 95%, hybrid + term map 84%, hybrid + term map + heading match 95% (hit@5). The term-map setups were tuned on this half, so these figures are optimistic.

**Pooled over both halves (38 questions) for the three setups that were never tuned:** keyword 63%, semantic 76%, hybrid 82%.

**Five questions written in their own words by another person** (no help from the author, kept separate): the term-map search placed the expected clause in the top 5 for 4 of 5; plain hybrid for 2 of 5. This 5-question check was used to choose the default setting, so it is a tie-breaker and not an unbiased score. The one miss in both setups was "How much of my total plot area am I allowed to cover with construction?", because ground coverage is spread across many tables.

**Choosing the default.** Plain hybrid looked best on the tuning half, the term map looked best on the held-out half, and the differences are within the noise of 19 questions. I used the five independent questions as the tie-breaker and switched to the term-map setting. A different reasonable person could have chosen otherwise.

**Refusal behaviour (retrieval stage).** The score-based evidence check does not separate answerable from unanswerable questions in this data (answerable questions scored as low as 0.24, unanswerable ones as high as 0.49), so it is not relied on. Protection against answering unanswerable questions rests on the AI step and the second-opinion check described above.

### Answer step: results on 20 everyday-wording questions

The 20 questions (`questions_user20.csv`) were written by the project owner in their own words. 14 have a correct clause marked by reading the PDF; 6 are not answered by this document. Model: `gemini-flash-lite-latest` (free tier). **This is a development set, not a clean test: the prompt and the checks were changed three times after seeing results on these same questions.** One run of the final version:

| Outcome | Answerable (14) | Unanswerable (6) |
|---|---|---|
| Answered | 8 | 1 |
| Refused ("not enough verified information") | 6 | 5 |
| AI text withheld by checks | 0 | 0 |

Of the 8 answers given to answerable questions, **6 cited the expected clause** and 2 cited a different clause (for example the basement margin 6.8.1 instead of the general margin 6.7.2 for "space next to my neighbour"). So about 6 of 14 answerable questions (43%) got a correct, cited answer. The tool is deliberately cautious: it refuses 6 of 14 answerable questions rather than risk a wrong answer, including "how much of my plot can I cover", where retrieval does not surface the right tables.

**How the same set behaved as the safeguards were added** (so the effect of each step is visible): with the original prompt, ~17 of 20 answers began "It depends…" and all 6 unanswerable questions were answered; with quote checks and the second-opinion step, answers state the rule first and 5 of 6 unanswerable questions are refused. Results change between runs even at temperature 0, and Gemini free-tier models sometimes return temporary overload errors.

**What this does not show:** it does not show the answers are correct for a particular plot, it has not been tested by people other than the author, and it has not been reviewed by a planning professional.

## Known limitations

- Tables are the hard part: margins, heights and FSI values often sit in tables that plain-text extraction handles poorly. The tool may point to the right section while the table values look jumbled.
- Scanned (image-only) PDFs are not supported; there is no OCR.
- Only the documents you load are searched. One document (Part II of CGDCR 2017) is loaded by default; amendments and later versions are not tracked.
- The term map is hand-written and covers about a dozen topics. Anything outside it relies on semantic search alone.
- The score-based evidence check is unreliable on this data (see Results); do not treat its refusals, or its silence, as a safety guarantee.
- Everyday-wording questions are the weak spot, and the test questions were written by the author, so real-user performance is likely lower than the numbers above.
- Answer-step results come from a single run per setting on a small development set; Gemini 'latest' model names change over time.
- The tool refuses a large share of answerable questions (6 of 14 in the last run). That is a deliberate trade-off, not a claim of coverage.
- Automatic checks reduce the risk of invented numbers but do not prove an answer is correct or that a clause applies to a specific plot.

## Future work

Support more documents and states, handle tables properly, track document versions, and compare provisions across jurisdictions. The structure (one clause list plus `sources.json`) is meant to allow this, but only one document has been tested.

## Source document

`clauses.jsonl` is the text of Part II of CGDCR 2017 split into clauses, extracted from the public notification (Urban Development and Urban Housing Department, Government of Gujarat; see `sources.json` for the link and a version note that is marked as not verified). It is included so the demo can run; the original PDF is not. To rebuild it: `python ingest.py <file.pdf> --out clauses.jsonl --chapters 6-11`.

## API keys

Never commit a key. In Colab, store it under Secrets. Locally, set `GEMINI_API_KEY` as an environment variable.
