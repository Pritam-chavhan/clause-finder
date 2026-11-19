# Evaluation protocol

- Questions were written by the author from the PDF. Correct clauses were marked by reading the PDF.
- 45 questions split in two: even rows for tuning, odd rows held out.
- Answer keys are used only by `eval.py` / `eval_answers.py` for scoring. They are never given to search or the model.
- A separate set of 20 everyday-wording questions (`questions_user20.csv`) was run through the live pipeline. The prompt was changed twice after seeing its results, so those 20 are a **development set, not a clean test**.
- Numbers are small-sample and indicative. See README for the tables.
