# pages/00_Data_Validation.py
# ===========================
# Data Validation page (read-only; uses centralized upload_manager)
# ===========================

from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ----- Page meta (must be the first Streamlit call) -----
ROOT = Path(__file__).resolve().parent.parent    # project root
ICON_PATH = ROOT / "assets" / "brcb-logo.png"

try:
    icon = Image.open(ICON_PATH) if ICON_PATH.exists() else "🧪"
except Exception:
    icon = "🧪"

st.set_page_config(
    page_title="Data Validation · Breaking CCUB",
    page_icon=icon,
    layout="wide",
)

# ----- Global header on every page -----
from common_ui import render_global_header
render_global_header(
    title="Breaking CCUB",
    subtitle="WorldCCUB Analytics",
    logo_path="assets/brcb-logo.png",
    logo_width=120,
    retina_scale=2,
)

# ----- Use the same upload manager as app.py (so adding/removing files works here too!)
from upload_manager import (
    ensure_upload_state, render_sidebar_manager, get_combined_df, has_files
)
ensure_upload_state()
render_sidebar_manager(show_meta=True)  # ← sidebar: list / add more / delete / clear

# ===========================
# Session inputs (read-only)
# ===========================
if not has_files():
    st.warning("No data found. Go to **Overview** and upload CSVs first.")
    st.stop()

df = get_combined_df()
if df is None or len(df) == 0:
    st.warning("No rows after combining files.")
    st.stop()

# ===========================
# Scope selection (Combined / Per‑file / Per‑country)
# ===========================
st.subheader("Validation Scope")
scope = st.radio(
    "Choose validation scope",
    ["Combined", "Per‑file", "Per‑country"],
    horizontal=True,
)

scoped_df = df
scope_label = "Combined"

if scope == "Per‑file":
    if "source_file" not in df.columns:
        st.error("No `source_file` column found. Make sure your loader adds it.")
        st.stop()
    files = sorted(df["source_file"].dropna().unique().tolist())
    if not files:
        st.error("No files detected in `source_file`.")
        st.stop()
    sel_file = st.selectbox("Select a file", files, key="dv_file")
    scoped_df = df[df["source_file"] == sel_file].copy()
    scope_label = f"File: {sel_file}"

elif scope == "Per‑country":
    if "country" not in df.columns:
        st.error("No `country` column found.")
        st.stop()
    countries = sorted(df["country"].dropna().unique().tolist())
    if not countries:
        st.error("No countries detected.")
        st.stop()
    sel_country = st.selectbox("Select a country", countries, key="dv_country")
    scoped_df = df[df["country"] == sel_country].copy()
    scope_label = f"Country: {sel_country}"

if scoped_df.empty:
    st.warning("No rows under the selected scope.")
    st.stop()

st.caption(f"Validating scope → **{scope_label}** · Rows: **{len(scoped_df):,}**")

# ===========================
# Validation knobs
# ===========================
with st.expander("Validation Settings", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_enh_len = st.number_input("Min enhanced caption words", min_value=1, max_value=200, value=15)
    with c2:
        max_enh_len = st.number_input("Max enhanced caption words", min_value=1, max_value=200, value=20)
    with c3:
        sample_n = st.number_input("Max rows per issue (sample)", min_value=5, max_value=200, value=30, step=5)
    with c4:
        url_required = st.checkbox("Require valid-looking URL in `original_url`", value=True)

# ===========================
# Required columns & helpers
# ===========================
required = [
    "image_id", "filename", "image_path",
    "original_description", "enhanced_caption",
    "category", "original_url", "country"
]
missing = [c for c in required if c not in scoped_df.columns]

st.subheader("Required Columns")
if missing:
    st.error(f"Missing required columns: {missing}")
else:
    st.success("All required columns are present.")

# Make a working copy; ensure key columns exist as strings
tmp = scoped_df.copy()
for col in ["original_description", "enhanced_caption", "country", "category"]:
    if col not in tmp.columns:
        tmp[col] = ""
tmp["original_description"] = tmp["original_description"].astype(str)
tmp["enhanced_caption"]    = tmp["enhanced_caption"].astype(str)
tmp["country"]             = tmp["country"].astype(str)
tmp["category"]            = tmp["category"].astype(str)

# Derived lengths
if "orig_len" not in tmp.columns:
    tmp["orig_len"] = tmp["original_description"].str.split().str.len()
if "enh_len" not in tmp.columns:
    tmp["enh_len"] = tmp["enhanced_caption"].str.split().str.len()

# ===========================
# High-level KPIs (no charts)
# ===========================
st.subheader("Quality KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Rows", f"{len(tmp):,}")
with k2:
    total_null = int(tmp.isna().sum().sum())
    st.metric("Total null cells", f"{total_null:,}")
with k3:
    empties = (
        tmp[["original_description","enhanced_caption","category","country"]]
        .astype(str)
        .applymap(lambda s: s.strip() == "" or s.strip().lower() == "nan")
    ).sum().sum()
    st.metric("Empty strings (key cols)", f"{int(empties):,}")
with k4:
    out_of_range = int(((tmp["enh_len"] < min_enh_len) | (tmp["enh_len"] > max_enh_len)).sum())
    st.metric("Enhanced len out of range", f"{out_of_range:,}")
with k5:
    if "image_path" in tmp.columns:
        missing_files = int((~tmp["image_path"].fillna("").apply(lambda p: os.path.exists(str(p)))).sum())
        st.metric("Missing local image files", f"{missing_files:,}")
    else:
        st.metric("Missing local image files", "N/A")

st.divider()

# ===========================
# Schema summary (nulls/types/uniques)
# ===========================
st.subheader("Nulls / Types / Uniques")
desc = []
for c in tmp.columns:
    s = tmp[c]
    nn = int(s.notnull().sum())
    nu = int(s.isnull().sum())
    uniq = int(s.nunique(dropna=True))
    example = None
    if nn > 0:
        try:
            example = s.dropna().iloc[0]
        except Exception:
            example = None
    desc.append({
        "column": c,
        "dtype": str(s.dtype),
        "non_null": nn,
        "null_count": nu,
        "null_pct": round((nu / max(len(s), 1)) * 100, 2),
        "unique": uniq,
        "example": example,
    })
schema_df = pd.DataFrame(desc).sort_values(["null_pct", "column"], ascending=[False, True])
st.dataframe(schema_df, use_container_width=True)

st.divider()

# ===========================
# Duplicates
# ===========================
st.subheader("Duplicates")

# Exact duplicate rows (all columns)
dup_all_mask = tmp.duplicated(keep=False)
dup_all_count = int(dup_all_mask.sum())
st.write(f"**Exact duplicate rows (all columns):** {dup_all_count:,}")
if dup_all_count:
    st.dataframe(tmp[dup_all_mask].head(sample_n), use_container_width=True)

# Key-based duplicates
keys_checked = []
per_key = {}
for key in ["filename", "original_url", "image_id"]:
    if key in tmp.columns:
        c = int(tmp[key].duplicated(keep=False).sum())
        per_key[key] = c
        keys_checked.append(key)

if keys_checked:
    st.write("**Per-key duplicate counts:**")
    st.write(per_key)
    for k, v in per_key.items():
        if v:
            st.markdown(f"**Examples: duplicates by `{k}`**")
            idx = tmp[tmp[k].duplicated(keep=False)].index
            cols = [col for col in ["filename","image_id","original_url","country","category","enhanced_caption","source_file"] if col in tmp.columns]
            st.dataframe(tmp.loc[idx, cols].head(sample_n), use_container_width=True)

st.divider()

# ===========================
# URL sanity (syntax only)
# ===========================
st.subheader("URL Sanity (`original_url`)")

def looks_like_url(u: str) -> bool:
    if not isinstance(u, str):
        return False
    u = str(u).strip()
    if not u:
        return False
    return bool(re.match(r"^https?://", u))

if "original_url" in tmp.columns:
    if url_required:
        bad_url_mask = ~tmp["original_url"].fillna("").apply(looks_like_url)
    else:
        bad_url_mask = pd.Series([False]*len(tmp), index=tmp.index)

    bad_url_cnt = int(bad_url_mask.sum())
    st.write(f"**Invalid-looking URLs:** {bad_url_cnt:,}")
    if bad_url_cnt:
        show_cols = [c for c in ["original_url","country","category","filename","source_file"] if c in tmp.columns]
        st.dataframe(tmp.loc[bad_url_mask, show_cols].head(sample_n), use_container_width=True)
else:
    st.info("Column `original_url` not found.")

st.divider()

# ===========================
# Local image path existence
# ===========================
st.subheader("Local Image Path Existence (`image_path`)")
if "image_path" in tmp.columns:
    exists_mask = tmp["image_path"].fillna("").apply(lambda p: os.path.exists(str(p)))
    not_exists_mask = ~exists_mask
    missing_n = int(not_exists_mask.sum())
    st.write(f"**Missing files on disk:** {missing_n:,}")
    if missing_n:
        show_cols = [c for c in ["image_path","filename","country","category","source_file"] if c in tmp.columns]
        st.dataframe(tmp.loc[not_exists_mask, show_cols].head(sample_n), use_container_width=True)
else:
    st.info("Column `image_path` not found.")

st.divider()

# ===========================
# Enhanced caption length QA
# ===========================
st.subheader(f"Enhanced Caption Length QA ({min_enh_len}–{max_enh_len} words)")
oor_mask = (tmp["enh_len"] < min_enh_len) | (tmp["enh_len"] > max_enh_len)
oor_n = int(oor_mask.sum())
st.write(f"**Out-of-range rows:** {oor_n:,}")
if oor_n:
    show_cols = [c for c in ["country","category","enh_len","enhanced_caption","filename","source_file"] if c in tmp.columns]
    st.dataframe(tmp.loc[oor_mask, show_cols].sort_values("enh_len").head(sample_n), use_container_width=True)

st.divider()

# ===========================
# Category / Country anomalies
# ===========================
st.subheader("Category / Country Anomalies")
blank_country = (tmp["country"].str.strip() == "") | (tmp["country"].str.lower() == "nan")
blank_category = (tmp["category"].str.strip() == "") | (tmp["category"].str.lower() == "nan")
st.write(f"**Blank `country`:** {int(blank_country.sum()):,}   |   **Blank `category`:** {int(blank_category.sum()):,}")
if int(blank_country.sum()):
    show_cols = [c for c in ["filename","country","category","enhanced_caption","source_file"] if c in tmp.columns]
    st.dataframe(tmp.loc[blank_country, show_cols].head(sample_n), use_container_width=True)
if int(blank_category.sum()):
    show_cols = [c for c in ["filename","country","category","enhanced_caption","source_file"] if c in tmp.columns]
    st.dataframe(tmp.loc[blank_category, show_cols].head(sample_n), use_container_width=True)

# Rare categories (<1% of dataset)
if "category" in tmp.columns:
    cat_counts = tmp["category"].value_counts(dropna=False)
    rare_threshold = max(1, int(0.01 * len(tmp)))
    rare_cats = cat_counts[cat_counts < rare_threshold]
    if len(rare_cats):
        st.markdown("**Rare categories (<1% of rows)** — may indicate typos / inconsistent labeling")
        st.dataframe(rare_cats.to_frame("count"), use_container_width=True)

st.divider()

# ===========================
# Memory footprint
# ===========================
st.subheader("Memory Footprint")
mem_mb = tmp.memory_usage(deep=True).sum() / (1024 * 1024)
st.write(f"Approx. DataFrame memory: **{mem_mb:.2f} MB**")

# ===========================
# Fix‑it hints
# ===========================
with st.expander("Fix‑it Hints", expanded=False):
    st.markdown("""
- **Missing required column** → Fix your preprocessor/export so the column is always present.
- **Blank/Null values** → Backfill programmatically or drop before training; log sources for follow-up.
- **Duplicate keys** (`filename` / `original_url` / `image_id`) → Deduplicate at ingestion; keep latest or highest-quality.
- **Invalid URL** → Ensure proper `https://` scheme or mark as non-downloadable.
- **Missing local image** → Verify root, relative vs. absolute paths, and OS-specific separators.
- **Enhanced caption length out of range** → Re-run the caption enhancer with stricter length control.
- **Rare categories** → Consolidate aliases (e.g., 'han-bok' → 'hanbok'); keep a controlled vocabulary list.
""")
