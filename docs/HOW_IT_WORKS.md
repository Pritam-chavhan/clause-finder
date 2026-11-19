# How the Clause Finder works (plain language)

1. **Understand the question.** `concepts.py` is a hand-written map from everyday words ("edge of my land") to the terms the regulation uses ("margin", "setback").
2. **Find the clauses.** `search.py` combines keyword search (BM25) and meaning-based search (MiniLM embeddings), merged with Reciprocal Rank Fusion. Clause headings get a small extra vote.
3. **Write a short answer using only those clauses.** `answer.py` asks Gemini for JSON: a direct answer, a simple explanation, and word-for-word quotes.
4. **Check the answer in code.**
   - every quote must appear word for word in the clause it names;
   - every number must appear in the clause it is attributed to;
   - every cited clause must have been retrieved;
   - no supporting quote means the AI text is withheld.
5. **Second opinion.** A separate model call sees only the question, the verified quotes and the draft, and says whether the quotes really answer *this* question (not a special case, not a different structure, not a different unit). If not, the tool says it cannot answer.
6. **Show original text.** `render.py` always prints the original clauses, page numbers and source next to the AI text.

What it does not do: it does not know your plot, it cannot read table values reliably (tables are flattened by PDF extraction), and it does not replace the authority or a professional.
