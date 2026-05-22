import streamlit as st
import pandas as pd
import os

# ── Config ──────────────────────────────────────────────────────────────────
SAMPLES_CSV   = "samples_me_aftersuzan.csv"       # input: merged stories
QUESTIONS_CSV = "questions_sabijn.csv"     # output: annotations

st.set_page_config(page_title="Story Annotator", page_icon="📖", layout="centered")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Lora', Georgia, serif;
    background-color: #0f0e0c;
    color: #e8e0d0;
}

/* Page title */
h1 { font-family: 'Lora', serif; font-weight: 600; color: #f0c060 !important; letter-spacing: -0.5px; }
h2, h3 { font-family: 'Lora', serif; color: #c8b898 !important; }

/* Progress bar */
.stProgress > div > div { background-color: #f0c060; }

/* Story card */
.story-card {
    background: #1a1814;
    border-left: 3px solid #f0c060;
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #d8d0c0;
    max-height: 280px;
    overflow-y: auto;
}

/* Source badge */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    background: #2a2520;
    color: #f0c060;
    border: 1px solid #3a3020;
    border-radius: 2px;
    padding: 2px 8px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Input labels */
label { font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; color: #a09080 !important; letter-spacing: 0.04em !important; }

/* Text inputs & textareas */
.stTextInput input, .stTextArea textarea {
    background: #1a1814 !important;
    border: 1px solid #3a3020 !important;
    border-radius: 4px !important;
    color: #e8e0d0 !important;
    font-family: 'Lora', serif !important;
    font-size: 0.93rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #f0c060 !important;
    box-shadow: 0 0 0 2px rgba(240,192,96,0.15) !important;
}

/* Radio buttons */
.stRadio label { font-family: 'Lora', serif !important; font-size: 0.95rem !important; color: #c8b898 !important; }

/* Buttons */
.stButton > button {
    background: #f0c060 !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.5rem 1.4rem !important;
    text-transform: uppercase !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Secondary / skip button */
.skip-btn .stButton > button {
    background: #2a2520 !important;
    color: #a09080 !important;
    border: 1px solid #3a3020 !important;
}

/* Divider */
hr { border-color: #2a2520 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0c0b09 !important; border-right: 1px solid #2a2520; }
[data-testid="stSidebar"] * { color: #a09080 !important; }

/* Success / warning messages */
.stSuccess { background: #1a2a18 !important; border-left-color: #60c060 !important; color: #90d890 !important; }
.stWarning { background: #2a2010 !important; border-left-color: #f0c060 !important; color: #c8a848 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_samples(path):
    return pd.read_csv(path)


def load_questions(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["id", "source", "story", "question",
                                  "answer_a", "answer_b", "answer_c", "gold_label"])


def save_question(path, row: dict):
    df = load_questions(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def already_annotated_ids(questions_df):
    return set(questions_df["id"].astype(int).tolist())


# ── Load data ────────────────────────────────────────────────────────────────
if not os.path.exists(SAMPLES_CSV):
    st.error(f"Could not find **{SAMPLES_CSV}**. Make sure it is in the same directory as this app.")
    st.stop()

samples_df   = load_samples(SAMPLES_CSV)
questions_df = load_questions(QUESTIONS_CSV)
annotated    = already_annotated_ids(questions_df)

# Find the first unannotated story (by id)
all_ids   = samples_df["id"].astype(int).tolist()
remaining = [i for i in all_ids if i not in annotated]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 Progress")
    st.markdown(f"**{len(annotated)}** / **{len(samples_df)}** annotated")
    pct = len(annotated) / len(samples_df) if len(samples_df) > 0 else 0
    st.progress(pct)
    st.markdown("---")
    st.markdown("**Gold label key**")
    st.markdown("- `A` → first answer\n- `B` → second answer\n- `C` → third answer")
    st.markdown("---")
    st.markdown(f"<span style='font-family:monospace;font-size:0.75rem;color:#605040'>output → {QUESTIONS_CSV}</span>", unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Story Annotator")

if not remaining:
    st.success("🎉 All stories have been annotated!")
    st.dataframe(questions_df, use_container_width=True)
    st.stop()

# Allow jumping to a specific story
current_id = remaining[0]
with st.expander("Jump to story", expanded=False):
    jump = st.number_input("Story id", min_value=min(all_ids),
                            max_value=max(all_ids),
                            value=remaining[0], step=1)
    if jump not in remaining:
        st.warning("This story is already annotated.")
    current_id = int(jump)

row = samples_df[samples_df["id"] == current_id].iloc[0]
source = row.get("source", "unknown") if "source" in samples_df.columns else "unknown"

# Story display
st.markdown(f'<div class="badge">#{current_id} &nbsp;·&nbsp; {source}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="story-card">{row["story"]}</div>', unsafe_allow_html=True)

st.markdown("#### Your question")
question = st.text_area("Question", placeholder="Write a multiple-choice question about this story…", label_visibility="collapsed", height=80, key=f"question_{current_id}")

st.markdown("#### Answer options")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<label>A</label>", unsafe_allow_html=True)
    answer_a = st.text_input("A", placeholder="Answer A", label_visibility="collapsed", key=f"answer_a_{current_id}")
with col2:
    st.markdown("<label>B</label>", unsafe_allow_html=True)
    answer_b = st.text_input("B", placeholder="Answer B", label_visibility="collapsed", key=f"answer_b_{current_id}")
with col3:
    st.markdown("<label>C</label>", unsafe_allow_html=True)
    answer_c = st.text_input("C", placeholder="Answer C", label_visibility="collapsed", key=f"answer_c_{current_id}")

st.markdown("#### Gold label")
gold = st.radio("Gold label", ["A", "B", "C"], horizontal=True, label_visibility="collapsed", key=f"gold_{current_id}")

st.markdown("")
col_save, col_skip = st.columns([1, 5])

with col_save:
    save = st.button("Save & next")

with col_skip:
    with st.container():
        st.markdown('<div class="skip-btn">', unsafe_allow_html=True)
        skip = st.button("Skip →")
        st.markdown('</div>', unsafe_allow_html=True)

if save:
    if not question.strip():
        st.warning("Please enter a question before saving.")
    elif not answer_a.strip() or not answer_b.strip() or not answer_c.strip():
        st.warning("Please fill in all three answer options.")
    else:
        save_question(QUESTIONS_CSV, {
            "id":          current_id,
            "source":      source,
            "story":       row["story"],
            "question":    question.strip(),
            "answer_a":    answer_a.strip(),
            "answer_b":    answer_b.strip(),
            "answer_c":    answer_c.strip(),
            "gold_label":  gold,
        })
        st.success("Saved! Loading next story…")
        st.cache_data.clear()
        st.rerun()

if skip:
    remaining.remove(current_id) if current_id in remaining else None
    st.rerun()