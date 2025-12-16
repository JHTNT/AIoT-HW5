import math
from typing import Dict, Iterable, List

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import logging

logging.set_verbosity_error()

MODEL_NAME = "openai-community/roberta-base-openai-detector"
AI_LABEL_HINTS = {"AI", "FAKE", "MACHINE", "GENERATED", "LABEL_1", "BOT"}
HUMAN_LABEL_HINTS = {"HUMAN", "REAL", "LABEL_0"}


@st.cache_resource(show_spinner=False)
def load_detector():
    """Load the HF pipeline once and cache it for reuse."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # force CPU for broad compatibility
    )


def score_to_percent(ai_score: float, human_score: float) -> Dict[str, float]:
    total = ai_score + human_score
    if total <= 0:
        return {"ai": 50.0, "human": 50.0}
    return {"ai": 100 * ai_score / total, "human": 100 * human_score / total}


def aggregate(scores: Iterable[Dict[str, float]]):
    ai_score = 0.0
    human_score = 0.0
    for row in scores:
        label = row.get("label", "").upper()
        score = float(row.get("score", 0.0))
        if any(hint in label for hint in AI_LABEL_HINTS):
            ai_score += score
        elif any(hint in label for hint in HUMAN_LABEL_HINTS):
            human_score += score
    # fallback: assume the top label represents AI if hints failed
    if ai_score == 0 and human_score == 0 and scores:
        best = max(scores, key=lambda r: r.get("score", 0))
        ai_score = float(best.get("score", 0.5))
        human_score = 1.0 - ai_score
    return ai_score, human_score


def format_scores(scores: List[Dict[str, float]]):
    if not scores:
        return "No scores returned."
    return "\n".join(
        f"{row.get('label', '?')}: {row.get('score', 0):.4f}" for row in scores
    )


st.set_page_config(
    page_title="AI vs Human Detector",
    page_icon="🤖",
    layout="wide",
)

st.title("AI vs Human 文本判斷")
st.caption("使用 Hugging Face roberta-base-openai-detector 進行快速推論。")

pipe = load_detector()

example_text = """
Large language models can draft text quickly, but human writers add nuance and context that models may miss.
""".strip()

input_text = st.text_area(
    "輸入待判斷的文本：",
    value=example_text,
    height=220,
    placeholder="貼上文章或輸入句子...",
)

col_run, col_clear = st.columns([2, 1])

with col_run:
    run_detection = st.button("判斷", type="primary")
with col_clear:
    clear = st.button("清空")

if clear:
    st.experimental_rerun()

if run_detection and input_text.strip():
    with st.spinner("模型推論中..."):
        result = pipe(input_text, top_k=None)

    if isinstance(result, list):
        # If batching is ever used, the pipeline returns a list per input.
        if result and isinstance(result[0], list):
            scores = result[0]
        else:
            scores = result
    else:
        scores = [result]
    ai_score, human_score = aggregate(scores)
    percents = score_to_percent(ai_score, human_score)

    st.subheader("結果")
    cols = st.columns(2)
    with cols[0]:
        st.metric("AI 可能性", f"{percents['ai']:.1f}%")
        st.progress(min(1.0, percents["ai"] / 100.0))
    with cols[1]:
        st.metric("Human 可能性", f"{percents['human']:.1f}%")
        st.progress(min(1.0, percents["human"] / 100.0))

    verdict = "AI 生成" if percents["ai"] >= percents["human"] else "Human 撰寫"
    st.info(f"推測結果：{verdict}")

    with st.expander("查看模型完整分數"):
        st.text(format_scores(scores))

    with st.expander("簡易統計"):
        words = input_text.split()
        st.write(
            {
                "字元數": len(input_text),
                "單詞數": len(words),
                "平均單詞長度": round(
                    sum(len(w) for w in words) / max(len(words), 1), 2
                ),
            }
        )
else:
    st.caption("點擊上方 \"判斷\" 按鈕以取得結果。")

st.sidebar.header("使用說明")
st.sidebar.markdown(
    "- 貼上待分析文本，點擊 **判斷**。\n"
    "- 顯示 AI / Human 概率與判斷。\n"
    "- 模型僅提供參考，長文本效果較佳。"
)

st.sidebar.divider()
st.sidebar.markdown(
    "模型來源: [roberta-base-openai-detector](https://huggingface.co/openai-community/roberta-base-openai-detector)"
)
