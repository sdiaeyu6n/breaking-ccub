# app.py (instant render after upload/delete; no rerun needed)
from pathlib import Path
import os, json, hashlib, re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import umap

# --- Page meta MUST be first ---
ROOT = Path(__file__).resolve().parent
ICON_PATH = ROOT / "assets" / "brcb-logo.png"
st.set_page_config(
    page_title="Breaking CCUB",
    page_icon=Image.open(ICON_PATH) if ICON_PATH.exists() else "🌍",
    layout="wide",
)

# Global header (appears on top of every page)
from common_ui import render_global_header
render_global_header(
    title="Breaking CCUB",
    subtitle="WorldCCUB Analytics",
    logo_path="assets/brcb-logo.png",
    logo_width=120,
    retina_scale=2,
)

# Plot style constants
from ui_style import PLOTLY_HEIGHT

# ---- uploader nonce for resetting the file_uploader widget ----
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0


# ---------------- Helpers ----------------
def _norm_text(s):
    if pd.isna(s): return ""
    return str(s)

def file_md5(uploaded_file) -> str:
    """Hash file content to dedupe by content (not by file name)."""
    b = uploaded_file.getbuffer()
    return hashlib.md5(b).hexdigest()

def open_image(row):
    """Return PIL image if local path exists, else original_url string or None."""
    p = _norm_text(row.get("image_path",""))
    if p and os.path.exists(p):
        try:
            return Image.open(p)
        except Exception:
            pass
    url = _norm_text(row.get("original_url",""))
    if url:
        return url
    return None

@st.cache_resource
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=True)
def compute_embeddings(texts, n_neighbors=15, min_dist=0.1, seed=42):
    enc = load_encoder()
    emb = enc.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    xy = reducer.fit_transform(emb)
    return emb, xy

# --- Keyword helpers ---
def _norm_kw(s: str) -> str:
    s = (s or "").strip().lower()
    return s if re.fullmatch(r"[a-z][a-z\-]{1,}", s) else ""

def split_keywords(s: str):
    raw = re.split(r"[,\n]+|\s{2,}", (s or "").strip())
    seen, out = set(), []
    for token in raw:
        nk = _norm_kw(token)
        if nk and nk not in seen:
            seen.add(nk); out.append(nk)
    return out

def build_counts_for_mask(df, mask, keywords, field="enhanced_caption"):
    sub = df[mask]
    text = " ".join(str(x).lower() for x in sub[field].fillna(""))
    row = {"_n": int(len(sub))}
    for kw in keywords:
        row[kw] = text.count(kw)
    return pd.DataFrame([row])

def build_country_keyword_counts(df, countries, keywords, field="enhanced_caption"):
    rows = []
    for c in countries:
        sub = df[df["country"] == c]
        text = " ".join(str(x).lower() for x in sub[field].fillna(""))
        row = {"country": c, "_n": len(sub)}
        for kw in keywords:
            row[kw] = text.count(kw)
        rows.append(row)
    return pd.DataFrame(rows)

def ensure_kw_state(countries, files):
    if "kw_map_country" not in st.session_state:
        st.session_state.kw_map_country = {}
    if "kw_map_file" not in st.session_state:
        st.session_state.kw_map_file = {}
    for c in countries:
        st.session_state.kw_map_country.setdefault(c, [])
    for f in files:
        st.session_state.kw_map_file.setdefault(f, [])

# NEW: single source of truth → always rebuild df from files_list
def _rebuild_combined_df():
    files_list = st.session_state.get("files_list", [])
    if files_list:
        df = pd.concat([item["df"] for item in files_list], ignore_index=True)
    else:
        df = pd.DataFrame()
    st.session_state["df"] = df
    return df

# ---------------- Session helpers ----------------
def _ensure_session_scaffold():
    if "files_list" not in st.session_state:
        st.session_state["files_list"] = []   # [{"name","hash","df","meta"}...]
    if "file_hashes" not in st.session_state:
        st.session_state["file_hashes"] = set()

def _rebuild_combined_df():
    files_list = st.session_state.get("files_list", [])
    if files_list:
        df = pd.concat([item["df"] for item in files_list], ignore_index=True)
    else:
        df = pd.DataFrame()
    st.session_state["df"] = df
    return df

def _ingest_files(uploaded_files):
    """Append newly uploaded CSVs; skip duplicate content by MD5."""
    if not uploaded_files:
        return 0
    added = 0
    for f in uploaded_files:
        try:
            h = file_md5(f)
            if h in st.session_state["file_hashes"]:
                continue

            f.seek(0)
            tmp = pd.read_csv(f)
            tmp["source_file"] = f.name

            # Ensure required columns & dtypes
            for col in ["original_description", "enhanced_caption", "country", "category"]:
                if col not in tmp.columns:
                    tmp[col] = ""
            tmp["original_description"] = tmp["original_description"].astype(str)
            tmp["enhanced_caption"]    = tmp["enhanced_caption"].astype(str)
            tmp["country"]             = tmp["country"].astype(str)
            tmp["category"]            = tmp["category"].astype(str)

            # Derived
            tmp["orig_len"] = tmp["original_description"].str.split().str.len()
            tmp["enh_len"]  = tmp["enhanced_caption"].str.split().str.len()

            meta = {
                "file": f.name,
                "rows": int(len(tmp)),
                "cols": int(tmp.shape[1]),
                "countries": int(tmp["country"].nunique()),
                "categories": int(tmp["category"].nunique()),
                "avg_enh_len": float(tmp["enh_len"].mean()) if len(tmp) else None,
            }

            st.session_state["files_list"].append(
                {"name": f.name, "hash": h, "df": tmp, "meta": meta}
            )
            st.session_state["file_hashes"].add(h)
            added += 1
        except Exception as e:
            st.error(f"Failed to read {getattr(f, 'name', 'file')}: {e}")
    return added

def _delete_file_by_index(idx):
    files_list = st.session_state.get("files_list", [])
    if 0 <= idx < len(files_list):
        item = files_list.pop(idx)
        st.session_state["file_hashes"].discard(item.get("hash"))
        return item["name"]
    return None

# ---------------- Data Manager (MAIN PAGE; not sidebar) ----------------
_ensure_session_scaffold()
with st.container():
    st.subheader("Data Manager")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Upload CSVs**")
        nonce = st.session_state["uploader_nonce"]
        new_files = st.file_uploader(
            "Upload one or more CSVs (you can drop files multiple times)",
            type="csv",
            accept_multiple_files=True,
            key=f"uploader_main_body_{nonce}",  # ← nonce를 섞어서 매번 '새 위젯'으로
        )

        # 업로드 즉시 반영 (드롭 → ingest → 위젯값 초기화 → 재구성 → 리렌더)
        if new_files:
            added = _ingest_files(new_files)
            if added:
                _rebuild_combined_df()
                st.success(f"Added {added} file(s).")
                # 업로더 초기화: nonce 증가 → 다음 렌더에서 '빈' 업로더로 교체
                st.session_state["uploader_nonce"] += 1
                st.rerun()

        # 전체 삭제
        if st.button("Clear all files", type="secondary", help="Remove every loaded file"):
            st.session_state.pop("files_list", None)
            st.session_state.pop("file_hashes", None)
            _ensure_session_scaffold()
            _rebuild_combined_df()
            st.info("All files cleared.")
            st.rerun()

    with right:
        st.markdown("**Loaded files**")
        files_list = st.session_state.get("files_list", [])
        if not files_list:
            st.caption("No files loaded yet.")
        else:
            # 요약 테이블
            summary = pd.DataFrame(
                [
                    {
                        "file": it["meta"]["file"],
                        "rows": it["meta"]["rows"],
                        "countries": it["meta"]["countries"],
                        "categories": it["meta"]["categories"],
                        "avg_enh_len": (
                            f'{it["meta"]["avg_enh_len"]:.1f}'
                            if it["meta"]["avg_enh_len"] is not None else "-"
                        ),
                    }
                    for it in files_list
                ]
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

            # 개별 삭제 (행마다 버튼)
            st.markdown("**Delete a file**")
            for idx, it in enumerate(files_list):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.caption(f"• {it['name']} — {it['meta'].get('rows', 0):,} rows")
                with col2:
                    if st.button("🗑 Delete", key=f"del_{idx}_{it['hash']}", help=f"Remove {it['name']}"):
                        removed = _delete_file_by_index(idx)
                        _rebuild_combined_df()
                        st.info(f"Removed {removed}.")
                        st.rerun()

# ---------------- Build combined df for analysis ----------------
df = _rebuild_combined_df()
if df is None or df.empty:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# ---------------- Tabs ----------------
tab_overview, tab_basic, tab_keywords, tab_embed, tab_gallery = st.tabs(
    ["Overview", "Basic Stats", "Cultural Keywords", "Embeddings", "Gallery"]
)

# ===== Overview =====
with tab_overview:
    st.subheader("Per-file summaries & previews")
    names = [item["name"] for item in st.session_state.get("files_list", [])]
    st.caption(f"Loaded files: {', '.join(names)}" if names else "No files loaded.")
    for item in st.session_state.get("files_list", []):
        meta = item["meta"]; tmp_df = item["df"]
        st.markdown(f"#### Preview: **{meta['file']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{meta['rows']:,}")
        c2.metric("Countries", meta["countries"])
        c3.metric("Categories", meta["categories"])
        c4.metric("Avg enhanced caption len", f"{meta['avg_enh_len']:.1f}" if meta["avg_enh_len"] is not None else "-")
        st.dataframe(tmp_df.head(20), use_container_width=True)
        st.divider()

# ===== Basic Stats (Plotly) =====
with tab_basic:
    st.subheader("Basic Stats")

    mode = st.radio(
        "Mode",
        ["Per‑country (single)", "Per‑file (single)", "Global (combined)"],
        horizontal=True,
        key="basic_mode"
    )

    # ---------- PER‑COUNTRY ----------
    if mode == "Per‑country (single)":
        countries = sorted(df["country"].dropna().unique().tolist())
        if not countries:
            st.info("No countries found in the data.")
            st.stop()
        sel_country = st.selectbox("Select a country", countries, key="bs_country")
        dsub = df[df["country"] == sel_country].copy()
        if dsub.empty:
            st.info(f"No rows for country: {sel_country}")
            st.stop()

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            cat_counts = dsub["category"].value_counts().rename_axis("category").reset_index(name="count")
            fig = px.bar(cat_counts, x="category", y="count", text="count",
                         hover_data={"category": True, "count": ":,"}, height=PLOTLY_HEIGHT)
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(title=f"Category Distribution — {sel_country}",
                              xaxis_title="", yaxis_title="Count",
                              margin=dict(l=10, r=10, t=40, b=10))
            fig.update_xaxes(tickangle=90)
            st.plotly_chart(fig, use_container_width=True)
        with r1c2:
            fig = px.histogram(dsub, x="enh_len", nbins=30, histnorm="probability density",
                               opacity=0.75, height=PLOTLY_HEIGHT, hover_data={"enh_len": True})
            fig.update_layout(title=f"Enhanced Caption Length — {sel_country}",
                              xaxis_title="words", yaxis_title="density",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("**Summary (this country)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(dsub):,}")
            c2.metric("Categories", dsub["category"].nunique())
            c3.metric("Avg enh len", f"{dsub['enh_len'].mean():.1f}")
        with r2c2:
            pivot = pd.crosstab(dsub["category"], dsub["country"])
            if pivot.size > 0 and pivot.shape[0] > 1:
                fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, text_auto=True,
                                color_continuous_scale="Blues", aspect="auto", height=PLOTLY_HEIGHT)
                fig.update_layout(title="Category × Country (this country)",
                                  xaxis_title="", yaxis_title="", coloraxis_showscale=False,
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough variation to draw a heatmap.")

    # ---------- PER‑FILE ----------
    elif mode == "Per‑file (single)":
        if "source_file" not in df.columns:
            st.info("No `source_file` column found. Make sure the loader adds it for per‑file mode.")
            st.stop()
        files = sorted(df["source_file"].dropna().unique().tolist())
        if not files:
            st.info("No files detected in `source_file`.")
            st.stop()
        sel_file = st.selectbox("Select a file", files, key="bs_file")
        fsub = df[df["source_file"] == sel_file].copy()

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            c_counts = fsub["country"].value_counts().rename_axis("country").reset_index(name="count")
            fig = px.bar(c_counts, x="country", y="count", text="count",
                         hover_data={"country": True, "count": ":,"}, height=PLOTLY_HEIGHT)
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(title=f"Samples by Country — {sel_file}",
                              xaxis_title="", yaxis_title="Count",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with r1c2:
            cat_counts = fsub.groupby(["category", "country"]).size().reset_index(name="count")
            cat_order = fsub["category"].value_counts().index.tolist()
            fig = px.bar(cat_counts, x="category", y="count", color="country",
                         category_orders={"category": cat_order},
                         hover_data={"category": True, "country": True, "count": ":,"},
                         barmode="group", height=PLOTLY_HEIGHT)
            fig.update_layout(title=f"Category Distribution — {sel_file}",
                              xaxis_title="", yaxis_title="Count",
                              legend_title_text="Country",
                              margin=dict(l=10, r=10, t=40, b=10))
            fig.update_xaxes(tickangle=90)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig = px.histogram(fsub, x="enh_len", color="country", nbins=30,
                               histnorm="probability density", opacity=0.65,
                               height=PLOTLY_HEIGHT, hover_data={"enh_len": True})
            fig.update_layout(title=f"Enhanced Caption Length — {sel_file}",
                              xaxis_title="words", yaxis_title="density",
                              legend_title_text="Country",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with r2c2:
            pivot = pd.crosstab(fsub["country"], fsub["category"])
            if pivot.size == 0:
                st.info("No data to display.")
            else:
                fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, text_auto=True,
                                color_continuous_scale="Blues", aspect="auto", height=PLOTLY_HEIGHT)
                fig.update_layout(title=f"Country × Category — {sel_file}",
                                  xaxis_title="", yaxis_title="",
                                  coloraxis_showscale=True,
                                  margin=dict(l=10, r=10, t=40, b=10))
                fig.update_xaxes(tickangle=90)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Summary (this file)**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(fsub):,}")
        c2.metric("Countries", fsub["country"].nunique())
        c3.metric("Categories", fsub["category"].nunique())

    # ---------- GLOBAL ----------
    else:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            c_counts = (df.groupby("country").size().reset_index(name="count")
                          .sort_values("count", ascending=False))
            fig = px.bar(c_counts, x="country", y="count", text="count",
                         hover_data={"country": True, "count": ":,"},
                         height=PLOTLY_HEIGHT)
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(title="Samples by Country",
                              xaxis_title="", yaxis_title="Count",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with r1c2:
            cat_counts = df.groupby(["category", "country"]).size().reset_index(name="count")
            cat_order = df["category"].value_counts().index.tolist()
            fig = px.bar(cat_counts, x="category", y="count", color="country",
                         category_orders={"category": cat_order},
                         hover_data={"category": True, "country": True, "count": ":,"},
                         barmode="group", height=PLOTLY_HEIGHT)
            fig.update_layout(title="Category Distribution",
                              xaxis_title="", yaxis_title="Count",
                              legend_title_text="Country",
                              margin=dict(l=10, r=10, t=40, b=10))
            fig.update_xaxes(tickangle=90)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig = px.histogram(df, x="enh_len", color="country", nbins=30,
                               histnorm="probability density", opacity=0.65,
                               height=PLOTLY_HEIGHT, hover_data={"enh_len": True})
            fig.update_layout(title="Enhanced Caption Length",
                              xaxis_title="words", yaxis_title="density",
                              legend_title_text="Country",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with r2c2:
            pivot = pd.crosstab(df["country"], df["category"]).sort_index()
            if pivot.size == 0:
                st.info("No data to display.")
            else:
                fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                                text_auto=True, color_continuous_scale="Blues",
                                aspect="auto", height=PLOTLY_HEIGHT)
                fig.update_layout(title="Country × Category (global)",
                                  xaxis_title="", yaxis_title="",
                                  coloraxis_showscale=True,
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.update_xaxes = fig.update_xaxes(tickangle=90)
                st.plotly_chart(fig, use_container_width=True)

# ===== Cultural Keywords =====
with tab_keywords:
    st.subheader("Cultural Keywords")
    countries = sorted(df["country"].dropna().unique().tolist())
    files = sorted(df["source_file"].dropna().unique().tolist()) if "source_file" in df.columns else []
    ensure_kw_state(countries, files)

    mode = st.radio(
        "Mode",
        ["Per‑country (single)", "Per‑file (single)", "Global search (combined)"],
        horizontal=True
    )

    # ---- PER‑COUNTRY ----
    if mode == "Per‑country (single)":
        left, right = st.columns(2)
        with left:
            sel_country = st.selectbox("Select a country", countries, key="kw_country_select")
            st.markdown("**Current keywords for this country**")
            cur = list(st.session_state.kw_map_country.get(sel_country, []))
            if cur:
                cols = st.columns(min(4, len(cur)))
                for i, kw in enumerate(cur):
                    with cols[i % len(cols)]:
                        if st.button(f"❌ {kw}", key=f"rm_country_{sel_country}_{kw}"):
                            cur = [x for x in cur if x != kw]
                            st.session_state.kw_map_country[sel_country] = cur
            else:
                st.caption("No keywords yet. Add some below.")

            with st.form(key=f"kw_form_country_{sel_country}", clear_on_submit=True):
                new_kw = st.text_input("Add keyword (letters or hyphen, e.g., hanbok)", "")
                submitted = st.form_submit_button("Add (Enter)")
            if submitted:
                nk = _norm_kw(new_kw)
                if not nk:
                    st.warning("Use lowercase letters or hyphen, length ≥ 2.")
                elif nk in cur:
                    st.info(f"'{nk}' already exists.")
                else:
                    cur.append(nk)
                    st.session_state.kw_map_country[sel_country] = cur
                    st.success(f"Added '{nk}'")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear ALL for this country"):
                    st.session_state.kw_map_country[sel_country] = []
            with c2:
                if st.button("Reset all countries to empty"):
                    for c in countries:
                        st.session_state.kw_map_country[c] = []

        with right:
            kws = st.session_state.kw_map_country.get(sel_country, [])
            if not kws:
                st.info("Add one or more keywords to visualize.")
            else:
                out = build_counts_for_mask(df, df["country"] == sel_country, kws, field="enhanced_caption")
                norm = st.checkbox("Normalize per 100 captions", value=True, key="kw_global_norm")
                if norm:
                    for k in kws:
                        out[k + "_per100"] = (out[k] / out["_n"].clip(lower=1)) * 100.0
                    plot_cols, ttl = [k + "_per100" for k in kws], f"{sel_country}: per 100 captions"
                else:
                    plot_cols, ttl = kws, f"{sel_country}: raw counts"

                plot_df = out.melt(id_vars=["_n"], value_vars=plot_cols,
                                   var_name="metric", value_name="value")
                plot_df["keyword"] = plot_df["metric"].str.replace("_per100", "", regex=False)

                fig = px.bar(plot_df, x="keyword", y="value", text="value", height=PLOTLY_HEIGHT)
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(title=ttl, xaxis_title="", yaxis_title=("per100" if norm else "count"),
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Table (this country only)**")
                st.dataframe(out, use_container_width=True)

    # ---- PER‑FILE ----
    elif mode == "Per‑file (single)":
        if not files:
            st.info("No `source_file` column found. Upload CSVs with `source_file` or ensure loader adds it.")
        else:
            left, right = st.columns(2)
            with left:
                sel_file = st.selectbox("Select a file", files, key="kw_file_select")
                st.markdown("**Current keywords for this file**")
                cur = list(st.session_state.kw_map_file.get(sel_file, []))
                if cur:
                    cols = st.columns(min(4, len(cur)))
                    for i, kw in enumerate(cur):
                        with cols[i % len(cols)]:
                            if st.button(f"❌ {kw}", key=f"rm_file_{sel_file}_{kw}"):
                                cur = [x for x in cur if x != kw]
                                st.session_state.kw_map_file[sel_file] = cur
                else:
                    st.caption("No keywords yet. Add some below.")

                with st.form(key=f"kw_form_file_{sel_file}", clear_on_submit=True):
                    new_kw = st.text_input("Add keyword (letters or hyphen)", "")
                    submitted = st.form_submit_button("Add (Enter)")
                if submitted:
                    nk = _norm_kw(new_kw)
                    if not nk:
                        st.warning("Use lowercase letters or hyphen, length ≥ 2.")
                    elif nk in cur:
                        st.info(f"'{nk}' already exists.")
                    else:
                        cur.append(nk)
                        st.session_state.kw_map_file[sel_file] = cur
                        st.success(f"Added '{nk}'")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Clear ALL for this file"):
                        st.session_state.kw_map_file[sel_file] = []
                with c2:
                    if st.button("Reset all files to empty"):
                        for f in files:
                            st.session_state.kw_map_file[f] = []

            with right:
                kws = st.session_state.kw_map_file.get(sel_file, [])
                if not kws:
                    st.info("Add one or more keywords to visualize.")
                else:
                    mask = (df["source_file"] == sel_file)
                    out = build_counts_for_mask(df, mask, kws, field="enhanced_caption")
                    norm = st.checkbox("Normalize per 100 captions", value=True, key="norm_file")
                    if norm:
                        for k in kws:
                            out[k + "_per100"] = (out[k] / out["_n"].clip(lower=1)) * 100.0
                        plot_cols, ttl = [k + "_per100" for k in kws], f"{sel_file}: per 100 captions"
                    else:
                        plot_cols, ttl = kws, f"{sel_file}: raw counts"

                    plot_df = out.melt(id_vars=["_n"], value_vars=plot_cols,
                                       var_name="metric", value_name="value")
                    plot_df["keyword"] = plot_df["metric"].str.replace("_per100", "", regex=False)

                    fig = px.bar(plot_df, x="keyword", y="value", text="value", height=PLOTLY_HEIGHT)
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    fig.update_layout(title=ttl, xaxis_title="", yaxis_title=("per100" if norm else "count"),
                                      margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("**Table (this file only)**")
                    st.dataframe(out, use_container_width=True)

    # ---- GLOBAL SEARCH ----
    else:
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            kw_input = st.text_input(
                "Type keywords (comma/newline/space separated). Press Enter to apply.",
                placeholder="e.g., hanbok, palace, kimchi"
            )
        with c2:
            field = st.selectbox("Text field", ["enhanced_caption", "original_description"], index=0, key="kw_global_field")
        with c3:
            norm = st.checkbox("Normalize per 100 captions", value=True)

        keywords = split_keywords(kw_input)
        if not keywords:
            st.info("Enter one or more keywords to search across countries.")
        else:
            out = build_country_keyword_counts(df, countries, keywords, field=field)
            heat_cols = keywords
            if norm:
                for k in keywords:
                    out[k + "_per100"] = (out[k] / out["_n"].clip(lower=1)) * 100.0
                heat_cols = [k + "_per100" for k in keywords]

            long = out.melt(id_vars=["country", "_n"], value_vars=heat_cols,
                            var_name="metric", value_name="value")
            long["keyword"] = long["metric"].str.replace("_per100", "", regex=False)
            pivot = long.pivot_table(index="country", columns="keyword",
                                     values="value", aggfunc="first", fill_value=0)

            fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, text_auto=True,
                            color_continuous_scale="YlGnBu", aspect="auto", height=PLOTLY_HEIGHT)
            fig.update_layout(title=("Keyword occurrences per 100 captions" if norm else "Keyword occurrences (raw count)"),
                              xaxis_title="", yaxis_title="", coloraxis_showscale=False,
                              margin=dict(l=10, r=10, t=40, b=10))
            fig.update_xaxes(tickangle=90)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Counts table (full width)**")
            st.dataframe(out, use_container_width=True)

# ===== Embeddings =====
with tab_embed:
    st.subheader("Embeddings (UMAP)")

    mode = st.radio(
        "Mode",
        ["Per‑file (single)", "Global (combined)"],
        horizontal=True,
        key="kw_mode"
    )

    # Common controls
    colA, colB = st.columns(2)
    with colA:
        text_field = st.selectbox("Text field", ["enhanced_caption", "original_description"], index=0, key="embed_text_field")
    with colB:
        show_centroids = st.checkbox("Show centroid distance matrix (cosine)", value=False)

    colC, colD, colE = st.columns(3)
    with colC:
        n_neighbors = st.slider("UMAP n_neighbors", 5, 80, 15, step=1)
    with colD:
        min_dist = st.slider("UMAP min_dist", 0.0, 0.99, 0.1, step=0.01)
    with colE:
        sample_max = st.number_input("Max points (sampling)", min_value=200, max_value=20000, value=3000, step=100)

    colF, colG = st.columns(2)
    with colF:
        point_alpha = st.slider("Point opacity", 0.2, 1.0, 0.75)
    with colG:
        point_size = st.slider("Point size", 3, 14, 6)

    run = st.button("Compute / Update Embeddings", type="primary")

    def _prepare_subset(sub_df, text_col, max_n):
        sub_df = sub_df.copy()
        sub_df[text_col] = sub_df[text_col].fillna("").astype(str)
        sub_df = sub_df[sub_df[text_col].str.strip() != ""]
        if len(sub_df) > max_n:
            sub_df = sub_df.sample(n=max_n, random_state=42)
        return sub_df

    # ---- PER‑FILE ----
    if mode == "Per‑file (single)":
        if "source_file" not in df.columns:
            st.info("No `source_file` column found. Make sure the loader adds it for per‑file analysis.")
        else:
            files = sorted(df["source_file"].dropna().unique().tolist())
            if not files:
                st.info("No files detected in `source_file`.")
            else:
                sel_file = st.selectbox("Select a file", files, key="embed_file_simple")
                sub = _prepare_subset(df[df["source_file"] == sel_file], text_field, sample_max)

                if not run:
                    st.info("Set options and click the button to compute.")
                elif sub.empty:
                    st.warning("No non-empty texts in the selected subset.")
                else:
                    color_col = "country" if "country" in sub.columns else None
                    emb, xy = compute_embeddings(sub[text_field].tolist(), n_neighbors, min_dist)
                    plot_df = sub.copy()
                    plot_df["x"], plot_df["y"] = xy[:, 0], xy[:, 1]
                    fig = px.scatter(plot_df, x="x", y="y", color=color_col,
                                     hover_data=["country", "category", "source_file", text_field],
                                     opacity=point_alpha, height=PLOTLY_HEIGHT)
                    fig.update_traces(marker={"size": point_size})
                    fig.update_layout(title=f"UMAP — {sel_file} ({text_field})",
                                      margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    if show_centroids and color_col:
                        groups, cents = [], []
                        for g, gdf in plot_df.groupby(color_col):
                            groups.append(str(g))
                            cents.append(emb[gdf.index.to_numpy()].mean(axis=0))
                        if len(cents) >= 2:
                            C = np.stack(cents)
                            D = cosine_distances(C)
                            st.subheader("Centroid distance (cosine) — by country in this file")
                            st.dataframe(pd.DataFrame(D, index=groups, columns=groups)
                                         .style.background_gradient(cmap="Reds"),
                                         use_container_width=True)

    # ---- GLOBAL ----
    else:
        sub = _prepare_subset(df, text_field, sample_max)
        if not run:
            st.info("Set options and click the button to compute.")
        elif sub.empty:
            st.warning("No non-empty texts in the selected subset.")
        else:
            color_col = "country" if "country" in sub.columns else None
            emb, xy = compute_embeddings(sub[text_field].tolist(), n_neighbors, min_dist)
            plot_df = sub.copy()
            plot_df["x"], plot_df["y"] = xy[:, 0], xy[:, 1]
            fig = px.scatter(plot_df, x="x", y="y", color=color_col,
                             hover_data=["country", "category", "source_file", text_field],
                             opacity=point_alpha, height=PLOTLY_HEIGHT)
            fig.update_traces(marker={"size": point_size})
            fig.update_layout(title=f"UMAP — Global ({text_field})",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            if show_centroids and color_col:
                groups, cents = [], []
                for g, gdf in plot_df.groupby(color_col):
                    groups.append(str(g))
                    cents.append(emb[gdf.index.to_numpy()].mean(axis=0))
                if len(cents) >= 2:
                    C = np.stack(cents)
                    D = cosine_distances(C)
                    st.subheader("Centroid distance (cosine) — by country (global)")
                    st.dataframe(pd.DataFrame(D, index=groups, columns=groups)
                                 .style.background_gradient(cmap="Reds"),
                                 use_container_width=True)

# ===== Gallery =====
with tab_gallery:
    st.subheader("WorldCCUB Gallery")
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_c = st.multiselect(
            "Country",
            sorted(df["country"].dropna().unique()),
            default=list(sorted(df["country"].dropna().unique()))
        )
    with col2:
        sel_cat = st.multiselect(
            "Category",
            sorted(df["category"].dropna().unique()),
            default=list(sorted(df["category"].dropna().unique()))
        )
    with col3:
        contains = st.text_input("Enhanced caption contains (lowercase)")
    ncol = st.slider("Columns", 2, 6, 3)

    sub = df[df["country"].isin(sel_c) & df["category"].isin(sel_cat)].copy()
    if contains:
        sub = sub[sub["enhanced_caption"].str.lower().str.contains(contains.lower())]

    cols = st.columns(ncol)
    for i, row in sub.head(60).iterrows():
        with cols[i % ncol]:
            img = open_image(row)
            if img is None:
                st.image("https://placehold.co/600x600?text=No+Image", use_container_width=True)
            else:
                st.image(img, use_container_width=True)
            st.markdown(f"**{row['country']} · {row['category']}**")
            st.caption(_norm_text(row["enhanced_caption"]))
