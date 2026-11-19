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

# Questions that the prototype has answered well in live tests (see README for the honest success rate).
EXAMPLES = [
    ("🏠", "Extending the house", "Can I extend my kitchen or bedroom into the open space beside my house?"),
    ("🏗️", "Adding a floor", "Can I add another floor to my existing two-storey house?"),
    ("📏", "Distance from the neighbour", "How much space do I need to leave between my house and my neighbour's property?"),
    ("↔️", "Space at the back", "Do I need to leave open space at the back of my house, or is space on the sides enough?"),
    ("🚿", "Bathroom or servant room", "Can I build a bathroom or servant room in the side open space of my house, and how big can it be?"),
    ("🪜", "Staircase or water tank", "Can I build an open staircase or a water tank in the side open space of my house?"),
]

st.set_page_config(page_title="Clause Finder · Gujarat building rules in plain words", page_icon="📐", layout="centered")

st.markdown(
    """
<style>
:root { --accent:#2f6fed; --accent2:#7c5cff; --line:rgba(128,128,128,.28); --soft:rgba(128,128,128,.08); }
.block-container { max-width: 820px; padding-top: 3.5rem; padding-bottom: 4rem; }
div[data-testid="stButton"] button p { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
div[data-testid="stButton"] button > div { width:100%; }
div[data-testid="stButton"] { width:100%; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

.hero { padding: 1.6rem 1.6rem 1.4rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(47,111,237,.16), rgba(124,92,255,.14));
        border: 1px solid var(--line); }
.hero .pill { display:inline-block; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; font-weight:600;
        padding:.2rem .6rem; border-radius:999px; background:var(--accent); color:#fff; margin-bottom:.7rem; }
.hero h1 { font-size: 2.1rem; line-height:1.15; margin:0 0 .5rem; padding:0; }
.hero p  { font-size: 1.02rem; margin:0; opacity:.9; }

.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; margin:.2rem 0 1.2rem; }
.step { border:1px solid var(--line); background:var(--soft); border-radius:14px; padding:.85rem .95rem; }
.step b { display:block; margin-bottom:.15rem; }
.step span { font-size:.88rem; opacity:.85; }
.num { display:inline-flex; width:1.5rem; height:1.5rem; align-items:center; justify-content:center; border-radius:50%;
       background:var(--accent); color:#fff; font-size:.8rem; font-weight:700; margin-right:.45rem; }
@media (max-width: 640px){ .steps { grid-template-columns:1fr; } .hero h1{font-size:1.7rem;} }

.section-label { font-size:.8rem; letter-spacing:.07em; text-transform:uppercase; font-weight:700; opacity:.65; margin:1.1rem 0 .4rem; }

/* example buttons as cards */
div[data-testid="stButton"] > button { width:100%; text-align:left; border-radius:12px; border:1px solid var(--line);
        padding:.7rem .9rem; min-height:6rem; white-space:normal; line-height:1.3; transition:all .15s; }
div[data-testid="stButton"] > button:hover { border-color:var(--accent); transform:translateY(-1px);
        box-shadow:0 4px 14px rgba(47,111,237,.18); }
div[data-testid="stFormSubmitButton"] > button, div[data-testid="stButton"] > button[kind="primary"] {
        background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff; border:none; font-weight:600;
        min-height:2.8rem; text-align:center; }
textarea { border-radius:12px !important; }

.notice { border:1px solid var(--line); border-left:4px solid #e0a100; background:rgba(224,161,0,.08);
          border-radius:10px; padding:.7rem .9rem; font-size:.9rem; margin:.4rem 0 1rem; }
.foot { text-align:center; font-size:.82rem; opacity:.7; margin-top:2.2rem; }
.foot a { color:inherit; }

/* answer area */
div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size:1.2rem; margin-top:1rem; }
blockquote { border-left:3px solid var(--accent) !important; background:var(--soft); border-radius:0 8px 8px 0; padding:.4rem .9rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <span class="pill">Prototype</span>
  <h1>Clause Finder</h1>
  <p>Ask about Gujarat's building rules (CGDCR 2017, Part II) in everyday words.
  Get a short answer with the exact clause, page and source beside it.</p>
</div>
<div class="steps">
  <div class="step"><b><span class="num">1</span>Find</b><span>Searches 190 passages of the regulation by meaning and by keyword.</span></div>
  <div class="step"><b><span class="num">2</span>Explain</b><span>An AI writes a plain-language answer using only those clauses.</span></div>
  <div class="step"><b><span class="num">3</span>Check</b><span>Code verifies quotes and numbers. If it cannot, it says so instead of guessing.</span></div>
</div>
""",
    unsafe_allow_html=True,
)


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


def pick(text):
    st.session_state["q"] = text
    st.session_state["run"] = True


def ask():
    st.session_state["run"] = True


searcher = get_searcher()
llm = get_llm()
st.session_state.setdefault("q", "")
st.session_state.setdefault("run", False)

st.markdown('<div class="section-label">Try one of these</div>', unsafe_allow_html=True)
cols = st.columns(2)
for i, (icon, label, text) in enumerate(EXAMPLES):
    with cols[i % 2]:
        st.button(f"{icon}  **{label}**  \n{text}", key=f"ex{i}", on_click=pick, args=(text,))

st.markdown('<div class="section-label">Or ask your own</div>', unsafe_allow_html=True)
st.text_area("Your question", key="q", height=90, max_chars=MAX_CHARS, label_visibility="collapsed",
             placeholder="e.g. Can I build a balcony that extends over the open space around my house?")
st.button("Find the answer", type="primary", on_click=ask)

st.markdown(
    '<div class="notice"><b>What it covers today:</b> this early version is built on a limited set of regulations: '
    "<b>Gujarat CGDCR 2017, Part II (Planning)</b> only, about 190 indexed passages. More documents and topics are planned. "
    "If your question is about something that is not loaded yet, the app says it cannot answer instead of guessing, so a "
    "&ldquo;not enough verified information&rdquo; reply usually means that topic has not been added yet. "
    "Because this is a prototype, it can still pick a less relevant clause, so please check the original text shown with every answer. "
    "Not legal or engineering advice. Answers take 20-30 seconds. Your question is sent to Google's Gemini API to write the answer, "
    "so please do not type personal details.</div>",
    unsafe_allow_html=True,
)

question = st.session_state["q"].strip()
if st.session_state.get("run") and question:
    st.session_state["run"] = False
    note, use = "", llm
    if use is not None and not allow_ai():
        use = None
        note = ("> **Demo limit reached.** To keep this free demo available, the AI summary is paused for a while. "
                "The matching original clauses are shown below.\n\n")
    elif use is None:
        note = "> **The AI summary is switched off in this demo.** The matching original clauses are shown below.\n\n"
    with st.spinner("Searching the regulations and checking the answer..."):
        result = answer_question(searcher, question, llm=use)
    with st.container(border=True):
        st.markdown(f"**Your question:** {question}")
        st.markdown(note + to_markdown(result))

foot = "Built by Pritam Chavhan"
if REPO_URL:
    foot += f' · <a href="{REPO_URL}" target="_blank">Code, tests and evaluation on GitHub</a>'
st.markdown(f'<div class="foot">{foot}</div>', unsafe_allow_html=True)
