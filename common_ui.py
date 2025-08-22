# common_ui.py
from pathlib import Path
import base64
import streamlit as st
from PIL import Image

def _resolve_logo(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (Path(__file__).resolve().parent / p).resolve()

@st.cache_resource
def _load_logo_bytes(path: str) -> bytes:
    p = _resolve_logo(path)
    return p.read_bytes()

def _inject_header_css():
    # Inject once per session
    if st.session_state.get("_hdr_css_injected"):
        return
    st.markdown(
        """
        <style>
          /* No rounded corners or shadows for the header logo */
          .app-header-logo {
            border-radius: 0 !important;
            box-shadow: none !important;
          }
          /* Crisper scaling hints (best effort across browsers) */
          .app-header-logo {
            image-rendering: -webkit-optimize-contrast; /* Safari/Chrome */
            image-rendering: crisp-edges;               /* Firefox/Chromium */
            -ms-interpolation-mode: nearest-neighbor;   /* Legacy IE */
          }
          /* Tighten the header vertical spacing a bit */
          .app-header-wrap { line-height: 1.2; }
          .app-header-title { margin: 0; }
          .app-header-sub { opacity: 0.85; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.session_state["_hdr_css_injected"] = True

def render_global_header(
    title: str = "Breaking CCUB",
    subtitle: str = "WorldCCUB Analytics",
    logo_path: str = "assets/brcb-logo.png",
    logo_width: int = 140,       # displayed CSS width (pixels)
    retina_scale: int = 2        # use 2x intrinsic pixels for sharpness
):
    """
    Renders a global header with a high-DPI logo (retina downscale) and no rounded corners.
    """
    _inject_header_css()

    # Prepare base64 for <img src="data:...">
    try:
        raw = _load_logo_bytes(logo_path)
        b64 = base64.b64encode(raw).decode("ascii")
        src = f"data:image/png;base64,{b64}"
    except Exception:
        # graceful fallback to emoji if the image fails to load
        src = None

    col_logo, col_text = st.columns([1, 8], vertical_alignment="center")

    with col_logo:
        if src:
            # Use a bigger intrinsic width for retina (e.g., 2x) while displaying at logo_width
            display_w = int(logo_width)
            intrinsic_w = int(logo_width * max(1, retina_scale))
            st.markdown(
                f"""
                <img class="app-header-logo"
                     src="{src}"
                     width="{display_w}"
                     style="max-width:{display_w}px;"
                     srcset="{src} {intrinsic_w}w"
                     sizes="{display_w}px" />
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"<div style='font-size:{int(logo_width*0.9)}px;'>🌍</div>", unsafe_allow_html=True)

    with col_text:
        st.markdown(
            f"""
            <div class="app-header-wrap">
              <h1 class="app-header-title">{title}</h1>
              <span class="app-header-sub">{subtitle}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.divider()