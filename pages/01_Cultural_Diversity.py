# pages/01_Cultural_Diversity.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ICON = ROOT / "assets" / "brcb-logo.png"
try:
    icon = Image.open(ICON) if ICON.exists() else "🌍"
except Exception:
    icon = "🌍"

st.set_page_config(page_title="Cultural Diversity · Breaking CCUB", page_icon=icon, layout="wide")

from common_ui import render_global_header
from ui_style import PLOTLY_HEIGHT
render_global_header("Breaking CCUB", "WorldCCUB Analytics", "assets/brcb-logo.png", 120, 2)

df = st.session_state.get("df", pd.DataFrame())
if df.empty:
    st.info("No data yet. Upload CSVs on the Home page.")
    st.stop()

st.subheader("Cultural Diversity Dashboard")

# 안전한 기본 컬럼 보정
for col in ["country", "category"]:
    if col not in df.columns:
        df[col] = ""
df["country"] = df["country"].astype(str)
df["category"] = df["category"].astype(str)

# ===== 지수 함수 =====
def shannon_entropy(counts: pd.Series) -> float:
    arr = counts.values.astype(float)
    arr = arr[arr > 0]
    s = arr.sum()
    if s <= 0 or arr.size == 0:
        return 0.0
    p = arr / s
    return float(-(p * np.log2(p)).sum())

def gini_index(counts: pd.Series) -> float:
    arr = counts.values.astype(float)
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = np.sort(arr / s)
    n = len(p)
    coef = 2 * (np.arange(1, n + 1) - 0.5) / n
    return float(1.0 - 2.0 * np.sum(p * coef))

with st.expander("Settings", expanded=True):
    level = st.radio("Measure diversity for…", ["Global", "Per‑country", "Per‑category"], horizontal=True)
    target = "category" if level in ["Global", "Per‑country"] else "country"
    sample_min = st.number_input("Min group size to include", 1, 10_000, 1, step=1)

st.divider()

# ===== 지표 계산 & 시각화 =====
if level == "Global":
    # 안전한 카운트 테이블 (중복 컬럼명 방지)
    counts_df = (
        df[target]
        .value_counts(dropna=False)
        .rename_axis(target)
        .reset_index(name="n")
    )
    # 최소 샘플 수 필터
    counts_df = counts_df[counts_df["n"] >= sample_min]

    # 지수 계산은 Series로
    H = shannon_entropy(counts_df["n"])
    G = gini_index(counts_df["n"])

    c1, c2 = st.columns(2)
    c1.metric("Shannon Entropy (higher=more diverse)", f"{H:.3f}")
    c2.metric("Gini (higher=more imbalance)", f"{G:.3f}")

    fig = px.bar(
        counts_df,
        x=target, y="n",
        text="n",
        height=PLOTLY_HEIGHT,
        hover_data={target: True, "n": ":,"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=f"Global distribution of {target}",
        xaxis_title="", yaxis_title="Count",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)

elif level == "Per‑country":
    rows = []
    for c, g in df.groupby("country", dropna=False):
        vc = g["category"].value_counts(dropna=False)
        vc = vc[vc >= sample_min]
        rows.append({
            "country": str(c),
            "rows": int(len(g)),
            "entropy": shannon_entropy(vc),
            "gini": gini_index(vc)
        })
    res = pd.DataFrame(rows).sort_values("entropy", ascending=False)
    st.dataframe(res, use_container_width=True)

    st.markdown("**Heatmap: Country × Category**")
    pivot = pd.crosstab(df["country"], df["category"])
    fig = px.imshow(
        pivot.values, x=pivot.columns, y=pivot.index, text_auto=True,
        color_continuous_scale="Blues", height=PLOTLY_HEIGHT
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)

else:  # Per‑category
    rows = []
    for cat, g in df.groupby("category", dropna=False):
        vc = g["country"].value_counts(dropna=False)
        vc = vc[vc >= sample_min]
        rows.append({
            "category": str(cat),
            "rows": int(len(g)),
            "entropy": shannon_entropy(vc),
            "gini": gini_index(vc)
        })
    res = pd.DataFrame(rows).sort_values("entropy", ascending=False)
    st.dataframe(res, use_container_width=True)

    st.markdown("**Heatmap: Category × Country**")
    pivot = pd.crosstab(df["category"], df["country"])
    fig = px.imshow(
        pivot.values, x=pivot.columns, y=pivot.index, text_auto=True,
        color_continuous_scale="Blues", height=PLOTLY_HEIGHT*2
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)