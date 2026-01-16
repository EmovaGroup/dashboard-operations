from pathlib import Path
import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "meteo"

def _read_svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")

@st.cache_data(show_spinner=False)
def get_svg(asset_file_meteo: str) -> str:
    fallback = ASSETS_DIR / "unknown.svg"

    if not asset_file_meteo:
        return _read_svg(fallback) if fallback.exists() else ""

    safe_name = Path(asset_file_meteo).name
    p = ASSETS_DIR / safe_name

    if p.exists():
        return _read_svg(p)

    return _read_svg(fallback) if fallback.exists() else ""

def render_meteo_icon(asset_file_meteo: str, size: int = 28):
    svg = get_svg(asset_file_meteo)
    if not svg:
        st.write("🌤️")
        return
    st.markdown(
        f"<div style='width:{size}px;height:{size}px;display:flex;align-items:center'>{svg}</div>",
        unsafe_allow_html=True,
    )
