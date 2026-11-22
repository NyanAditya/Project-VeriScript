"""Streamlit web interface (Sec. 6.3.4): paste or upload a document, pick a mode, see the result.

Calls the REST service at $VERISCRIPT_API_URL (default http://127.0.0.1:8000). If the service is
not reachable it falls back to loading the core library in-process.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API = os.environ.get("VERISCRIPT_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEMO = os.environ.get("VERISCRIPT_DEMO", "") == "1"   # public hosted demo (e.g. Hugging Face Spaces)

st.set_page_config(page_title="VeriScript", layout="centered")


@st.cache_resource(show_spinner="Loading models (first visit after a restart takes about a minute) ...")
def local_detector():
    from veriscript.cascade import CascadeDetector

    return CascadeDetector.from_config()


@st.cache_resource
def detector_lock():
    # One detector is shared by all sessions; serialise use so that one visitor's mode
    # cannot change the thresholds while another visitor's document is being scored.
    return threading.Lock()


def api_up() -> bool:
    if not API:
        return False
    try:
        return requests.get(f"{API}/api/v1/health", timeout=2).ok
    except requests.RequestException:
        return False


def run_detection(text: str | None, upload, mode: str) -> dict:
    if api_up():
        if upload is not None:
            r = requests.post(f"{API}/api/v1/detect/file", params={"mode": mode},
                              files={"file": (upload.name, upload.getvalue())}, timeout=300)
        else:
            r = requests.post(f"{API}/api/v1/detect", json={"text": text, "mode": mode}, timeout=300)
        if not r.ok:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()
    from veriscript.io import extract_bytes

    det = local_detector()
    if upload is not None:
        text = extract_bytes(upload.getvalue(), upload.name)
    with detector_lock():
        prev = (det.gate, det.operating_point)
        try:
            det.set_operating_point(mode)
            return det.detect(text).to_json()
        finally:
            det.gate, det.operating_point = prev


st.title("VeriScript")
st.caption("Hybrid heuristic-transformer detector of AI-generated text. "
           "The score is a probability, not proof. Review it before acting on it.")

if DEMO:
    st.warning("**Public demo.** Text you submit is processed on this server and is not stored, "
               "but do not paste confidential documents or real student work. Results are for "
               "demonstration only and must not be used to judge anyone's work.")

mode = st.radio("Mode", ["fast", "balanced", "thorough"], index=1, horizontal=True,
                help="fast: escalates fewest documents to the transformer; thorough: escalates most.")
tab_text, tab_file = st.tabs(["Paste text", "Upload file"])
with tab_text:
    from app.examples import AI as EXAMPLE_AI, HUMAN as EXAMPLE_HUMAN

    e1, e2, _ = st.columns([1, 1, 2])
    if e1.button("Example: human text"):
        st.session_state["doc_text"] = EXAMPLE_HUMAN
    if e2.button("Example: AI text"):
        st.session_state["doc_text"] = EXAMPLE_AI
    text = st.text_area("Document text", height=260, key="doc_text",
                        placeholder="Paste at least 100 words of English prose ...")
    st.caption("Examples: held-out answers from the HC3 dataset (CC BY-SA 4.0).")
with tab_file:
    upload = st.file_uploader("A .txt, .pdf or .docx file", type=["txt", "pdf", "docx"])

if st.button("Check document", type="primary"):
    if upload is None and not (text and text.strip()):
        st.warning("Paste some text or upload a file first.")
        st.stop()
    with st.spinner("Analysing ..."):
        try:
            res = run_detection(text, upload, mode)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.stop()

    if res["probability"] is None:
        st.warning(f"Result: {res['label']}. {res.get('note', '')}")
        st.stop()

    p = res["probability"]
    is_ai = res["label"] == "ai_generated"
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict", "AI-generated" if is_ai else "Human-written")
    c2.metric("P(AI-generated)", f"{p:.1%}")
    c3.metric("Route", "fast path" if res["route"] == "fast_path" else "escalated")
    st.progress(min(max(p, 0.0), 1.0), text="0 = human-written  ·  1 = AI-generated")
    lat = res["latency_ms"]
    st.caption(f"Latency {lat.get('total', 0):.0f} ms (heuristic {lat.get('heuristic', 0):.0f} ms, "
               f"transformer {lat.get('transformer', 0):.0f} ms) · {res['n_words']} words")

    ex = res.get("explanation") or {}
    if ex.get("decided_by") == "transformer":
        st.info(f"The heuristic stage was uncertain (p = {ex.get('heuristic_probability', 0):.3f}), so the document "
                f"was escalated. The fine-tuned transformer decided (p = {ex.get('transformer_probability', 0):.3f}, "
                f"{ex.get('n_windows', 1)} window(s)). The features below are context only.")
    feats = ex.get("top_features") or []
    if feats:
        st.subheader("Most influential features")
        st.dataframe(pd.DataFrame([{
            "feature": f["name"].replace("pos_bigram:", "POS bigram ").replace("_", " "),
            "value": f["value"],
            "pushes towards": "AI" if f["direction"] == "towards_ai" else "human",
            "human median": f["human_median"],
            "AI median": f["ai_median"],
            "group": f["group"],
        } for f in feats]), hide_index=True, use_container_width=True)
    st.caption(ex.get("note", "Probabilistic estimate; review before acting on it."))
