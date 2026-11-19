# Case note: Clause Finder (half page)

**Problem.** Building rules in India sit in long PDFs (here: Gujarat CGDCR 2017, Part II, 76 pages). A homeowner asks "how much space must I leave next to my neighbour?"; the answer is in a clause, often in a table, under a heading nobody would guess. Search by keyword fails because people and regulations use different words.

**What I built.** A tool that takes an everyday question, finds the right clauses, answers in plain words using only those clauses, and prints the original text and page beside it. The main design decision was trust: an AI answer is only shown if code can verify that its quotes appear word for word in the clause, that every number comes from the clause it is attributed to, and that a second model call agrees the clause answers *this* question. Otherwise the tool says it cannot answer.

**What I measured (small, self-written sets, so indicative only).**
- Retrieval, 38 held-out/tuning questions: correct clause in top 5 for 82% with hybrid search; the hand-written term map lifted the held-out half from 68% to 79-84%.
- Answers, 20 everyday questions: of 14 answerable, 8 were answered and 6 of those cited the right clause; 6 were refused rather than risk a wrong answer; of 6 unanswerable, 5 were refused. Before the safeguards, almost all answers opened with "It depends" and all 6 unanswerable questions were answered.

**What I learned.** (1) A tool that refuses is more useful than one that guesses, but refusing 6 of 14 is a real cost. (2) Checks written in code beat instructions to the model. (3) Tables are the hard part: ground coverage and margins live in tables that PDF extraction flattens. (4) My own test questions were too easy; the questions I wrote in everyday words were where it failed.

**Next.** Ask 5-10 people who work with these rules for real questions; handle tables properly; track amendments.
