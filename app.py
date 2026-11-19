"""Public web demo (Streamlit).   Run locally:  streamlit run app.py

The API key is read only from the environment / Streamlit secret GEMINI_API_KEY and is never shown.
Because a public link can be opened by anyone, the AI step is limited (per visitor session and per hour overall).
Past the limit the app still shows the matching ORIGINAL clauses, without the AI summary. Questions are not stored.
"""
import os
import time

import streamlit as st

from answer import answer_question, make_gemini_llm
from render import to_markdown
from search import Searcher, load_clauses

REPO_URL = os.environ.get("REPO_URL", "")
PER_SESSION = int(os.environ.get("PER_SESSION_LIMIT", "6"))
GLOBAL_PER_HOUR = int(os.environ.get("GLOBAL_PER_HOUR", "40"))
MAX_CHARS = 400
EXAMPLES = [
    "Can I extend my kitchen or bedroom into the open space beside my house?",
    "Can I add another floor to my existing two-storey house?",
    "Can I build a balcony that extends over the open space around my house?",
    "How many parking spaces do I need to provide when building a house?",
    "Are there different construction rules for corner plots?",
]

st.set_page_config(page_title="Clause Finder (prototype)", layout="centered")
st.title("Clause Finder (prototype)")
st.markdown(
    "Ask about Gujarat's building regulations (CGDCR 2017, Part II) in everyday words. You get a short plain-language answer "
    "**only if it passes automatic checks** (quotes copied word for word from the clause, numbers taken from that clause, "
    "and a second check that the clause answers your exact question). Otherwise it says it cannot answer. The original "
    "clause text, page and source are always shown.")
st.warning("Prototype tested on small self-written question sets. It refuses often and can pick the wrong clause. "
           "Not legal or engineering advice. Answers can take 20-30 seconds. Questions are sent to Google's Gemini API "
           "to write the answer, so please do not type personal details.")
if REPO_URL:
    st.markdown(f"Code, tests and evaluation: [{REPO_URL}]({REPO_URL})")


@st.cache_resource(show_spinner="Loading the regulation index (first visit takes about a minute)...")
def get_searcher():
    return Searcher(load_clauses("clauses.jsonl"))


@st.cache_resource
def get_llm():
    key = os.environ.get("GEMINI_API_KEY")
    try:
        return make_gemini_llm(key) if key else None
    except Exception as e:
        print("AI step disabled:", e)
        return None


@st.cache_resource
def global_uses():
    return []


def allow_ai():
    now = time.time()
    uses = global_uses()
    uses[:] = [t for t in uses if now - t < 3600]
    if len(uses) >= GLOBAL_PER_HOUR or st.session_state.get("used", 0) >= PER_SESSION:
        return False
    uses.append(now)
    st.session_state["used"] = st.session_state.get("used", 0) + 1
    return True


searcher = get_searcher()
llm = get_llm()

if "q" not in st.session_state:
    st.session_state["q"] = ""
st.write("Try an example:")
for i, ex in enumerate(EXAMPLES):
    if st.button(ex, key=f"ex{i}"):
        st.session_state["q"] = ex

question = st.text_area("Your question", key="q", height=80, max_chars=MAX_CHARS,
                        placeholder="e.g. Can I extend my kitchen or bedroom into the open space beside my house?")
if st.button("Find the answer", type="primary") and question.strip():
    note, use = "", llm
    if use is not None and not allow_ai():
        use = None
        note = ("> **Demo limit reached.** To keep this free demo available, the AI summary is paused for a while. "
                "The matching original clauses are shown below.\n\n")
    elif use is None:
        note = "> **The AI summary is switched off in this demo.** The matching original clauses are shown below.\n\n"
    with st.spinner("Searching the regulations and checking the answer..."):
        result = answer_question(searcher, question.strip(), llm=use)
    st.markdown(note + to_markdown(result))
