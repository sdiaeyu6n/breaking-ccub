# upload_manager.py
import pandas as pd
import streamlit as st

def ensure_upload_state():
    if "files_list" not in st.session_state:
        st.session_state["files_list"] = []   # [{"name","hash","df","meta"}...]
    if "file_hashes" not in st.session_state:
        st.session_state["file_hashes"] = set()
    if "df" not in st.session_state:
        st.session_state["df"] = None

def has_files() -> bool:
    return bool(st.session_state.get("files_list"))

def get_combined_df():
    files = st.session_state.get("files_list", [])
    if not files:
        st.session_state["df"] = None
        return None
    df = pd.concat([x["df"] for x in files], ignore_index=True)
    st.session_state["df"] = df
    return df

def render_landing_uploader(title="Upload CSVs", help_text=""):
    st.header(title)
    if help_text:
        st.caption(help_text)

    upl = st.file_uploader(
        "Drop one or more CSVs",
        type="csv",
        accept_multiple_files=True
    )
    if upl:
        _ingest_files(upl)
        # 바로 결합 df 재계산
        get_combined_df()
        st.success("Files added. You can keep uploading more.")
        st.rerun()

def _ingest_files(files):
    for f in files:
        try:
            # 해시로 중복 방지 (app.py와 동일한 로직이라면 거기에 있는 해시 함수 재사용)
            b = f.getbuffer()
            import hashlib
            h = hashlib.md5(b).hexdigest()
            if h in st.session_state["file_hashes"]:
                continue
            f.seek(0)
            tmp = pd.read_csv(f)
            tmp["source_file"] = f.name

            for col in ["original_description", "enhanced_caption", "country", "category"]:
                if col not in tmp.columns:
                    tmp[col] = ""
            tmp["original_description"] = tmp["original_description"].astype(str)
            tmp["enhanced_caption"]    = tmp["enhanced_caption"].astype(str)
            tmp["country"]             = tmp["country"].astype(str)
            tmp["category"]            = tmp["category"].astype(str)
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

            st.session_state["files_list"].append({
                "name": f.name, "hash": h, "df": tmp, "meta": meta
            })
            st.session_state["file_hashes"].add(h)
        except Exception as e:
            st.error(f"Failed to ingest {getattr(f, 'name', 'file')}: {e}")

def render_sidebar_manager(show_meta=True):
    with st.sidebar:
        st.markdown("### Session files")
        files = st.session_state.get("files_list", [])

        if not files:
            st.info("No files in session yet.")
            return

        # 개별 삭제 버튼
        for i, item in enumerate(list(files)):  # list()로 복사본 순회
            cols = st.columns([5, 1])
            with cols[0]:
                if show_meta:
                    st.caption(f"{item['name']} · {len(item['df']):,} rows · {item['df']['country'].nunique()} countries")
                else:
                    st.caption(item["name"])
            def _delete_one(idx=i, h=item.get("hash")):
                try:
                    removed = st.session_state["files_list"].pop(idx)
                except IndexError:
                    return
                # 해시도 제거 (다시 같은 파일 올릴 수 있도록)
                if h and h in st.session_state.get("file_hashes", set()):
                    st.session_state["file_hashes"].remove(h)
                # 결합 df 재계산 또는 비우기
                if st.session_state["files_list"]:
                    get_combined_df()
                else:
                    st.session_state["df"] = None
                st.toast(f"Removed: {removed['name']}", icon="🗑️")
                st.rerun()

            with cols[1]:
                st.button("🗑️", key=f"del_{i}_{item['name']}", on_click=_delete_one)

        st.divider()

        # 전체 삭제 버튼
        if st.button("Clear ALL files"):
            st.session_state["files_list"].clear()
            st.session_state["file_hashes"].clear()
            st.session_state["df"] = None
            st.toast("Cleared all files", icon="🧹")
            st.rerun()
