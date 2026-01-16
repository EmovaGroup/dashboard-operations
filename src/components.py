# src/components.py
# -----------------------------------------------------------------------------
# Composants UI réutilisables (KPI cards, Store card, Assets météo, helpers...)
#
# Utilisé dans :
# - pages/1_Global.py  -> KPI cards + fiche magasin + météo
# - pages/3_Achats.py  -> KPI cards + normalisation fournisseurs
# -----------------------------------------------------------------------------
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components
import re

ACCENT_COLOR = "#b00000"

# =============================================================================
# KPI CSS (utilisé dans pages/1_Global.py et pages/3_Achats.py)
# =============================================================================
def inject_kpi_css():
    """
    Important :
    - Avant : kpi-main / kpi-sub forcés en blanc -> illisible en thème clair
    - Maintenant : on se base sur les variables Streamlit (var(--text-color))
      donc ça marche en light + dark sans hack.
    """
    st.markdown(
        f"""
        <style>
        .kpi-card {{
          border: 3px solid {ACCENT_COLOR};
          border-radius: 14px;
          padding: 14px;
          background: rgba(255,255,255,0.04);
          text-align: center;
          min-height: 130px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
        }}

        .kpi-title {{
          font-size: 13px;
          font-weight: 900;
          color: {ACCENT_COLOR};
          margin-bottom: 6px;
          text-transform: uppercase;
          letter-spacing: .4px;
        }}

        /* ✅ On utilise le thème Streamlit */
        .kpi-main {{
          font-size: 22px;
          font-weight: 900;
          color: var(--text-color);
          margin: 6px 0;
        }}

        .kpi-sub {{
          font-size: 12px;
          color: var(--text-color);
          opacity: 0.70;
        }}

        .kpi-row {{
          margin-bottom: 22px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# KPI rouge “simple” (si utilisé encore dans certaines pages)
def kpi_card_red(title: str, main_value: str, sublabel: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-main">{main_value}</div>
          <div class="kpi-sub">{sublabel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Helpers de format (utilisés partout)
# =============================================================================
def fmt_money(x, digits=0):
    try:
        x = float(x or 0)
    except Exception:
        x = 0
    s = f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", " ")
    return s + " €"


def fmt_int(x):
    try:
        n = int(round(float(x or 0)))
    except Exception:
        n = 0
    return f"{n:,}".replace(",", " ")


# =============================================================================
# KPI Compare (Année N vs N-1) (utilisé pages/1_Global.py et pages/3_Achats.py)
# =============================================================================
def inject_kpi_compare_css():
    st.markdown(
        """
<style>
  .kpi-compare {
    color: var(--text-color);
    border-radius: 14px;
    padding: 14px 14px 12px 14px;
    border: 1px solid rgba(128,128,128,0.25);
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    margin-bottom: 10px;
  }

  .kpi-compare-title {
    font-size: 14px;
    font-weight: 800;
    opacity: 0.95;
    margin-bottom: 10px;
    letter-spacing: .2px;
  }

  .kpi-compare-main {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .kpi-compare-values {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }

  .kpi-line {
    font-size: 15px;
    display: flex;
    gap: 8px;
    align-items: baseline;
    color: var(--text-color);
    opacity: 0.88;
  }

  .kpi-line-main {
    font-weight: 900;
    opacity: 1;
  }

  .kpi-label { opacity: 0.72; }
  .kpi-value { white-space: nowrap; }

  .kpi-compare-delta {
    font-size: 14px;
    font-weight: 950;
    padding: 8px 12px;
    border-radius: 999px;
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 10px 18px rgba(0,0,0,0.22);
  }

  .kpi-delta-up { background: #12a058; color: #fff; }
  .kpi-delta-down { background: #dc3c3c; color: #fff; }
  .kpi-delta-neutral { background: #777; color: #fff; }

  .kpi-bg-green {
    background: linear-gradient(180deg, rgba(18,160,88,0.26) 0%, rgba(18,160,88,0.10) 100%);
  }
  .kpi-bg-red {
    background: linear-gradient(180deg, rgba(220,60,60,0.26) 0%, rgba(220,60,60,0.10) 100%);
  }
  .kpi-bg-neutral {
    background: linear-gradient(180deg, rgba(120,120,120,0.18) 0%, rgba(120,120,120,0.08) 100%);
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card_compare(
    title: str,
    value_n: float,
    value_n1: float,
    label_n: str,
    label_n1: str,
    formatter=None,
):
    if formatter is None:
        formatter = lambda x: f"{x:,.0f}".replace(",", " ")

    n = float(value_n or 0)
    n1 = float(value_n1 or 0)

    if n1 == 0:
        if n == 0:
            delta_txt = "0.0%"
            bg = "kpi-bg-neutral"
            delta_cls = "kpi-delta-neutral"
            arrow = "→"
        else:
            delta_txt = "+∞"
            bg = "kpi-bg-green"
            delta_cls = "kpi-delta-up"
            arrow = "↑"
    else:
        delta = (n - n1) / abs(n1) * 100.0
        delta_txt = f"{delta:+.1f}%"
        if delta > 0:
            bg = "kpi-bg-green"
            delta_cls = "kpi-delta-up"
            arrow = "↑"
        elif delta < 0:
            bg = "kpi-bg-red"
            delta_cls = "kpi-delta-down"
            arrow = "↓"
        else:
            bg = "kpi-bg-neutral"
            delta_cls = "kpi-delta-neutral"
            arrow = "→"

    html = (
        f'<div class="kpi-compare {bg}">'
        f'  <div class="kpi-compare-title">{title}</div>'
        f'  <div class="kpi-compare-main">'
        f'    <div class="kpi-compare-values">'
        f'      <div class="kpi-line kpi-line-main">'
        f'        <span class="kpi-label">{label_n}</span>'
        f'        <span class="kpi-value">{formatter(n)}</span>'
        f'      </div>'
        f'      <div class="kpi-line">'
        f'        <span class="kpi-label">{label_n1}</span>'
        f'        <span class="kpi-value">{formatter(n1)}</span>'
        f'      </div>'
        f'    </div>'
        f'    <div class="kpi-compare-delta {delta_cls}">{arrow} {delta_txt}</div>'
        f'  </div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# Store card (fiche magasin) (utilisé dans pages/1_Global.py)
# =============================================================================
def inject_store_css():
    st.markdown(
        """
<style>
  .store-card {
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.22);
    box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    padding: 14px 16px;
    color: var(--text-color);
    background: rgba(255,255,255,0.04);
    margin-bottom: 12px;
  }

  .store-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 14px;
  }

  .store-title {
    font-size: 18px;
    font-weight: 900;
    line-height: 1.15;
    margin-bottom: 6px;
  }

  .store-subtitle {
    font-size: 12.5px;
    opacity: 0.9;
    line-height: 1.35;
  }

  .store-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
  }

  .badge {
    font-size: 12px;
    font-weight: 900;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(0,0,0,0.12);
    background: rgba(0,0,0,0.10);
    white-space: nowrap;
  }

  .badge-ok { background: rgba(18,160,88,0.18); border-color: rgba(18,160,88,0.35); }
  .badge-warn { background: rgba(255,170,0,0.18); border-color: rgba(255,170,0,0.35); }
  .badge-neutral { background: rgba(120,120,120,0.14); border-color: rgba(120,120,120,0.30); }

  .store-3col {
    margin-top: 12px;
    display: grid;
    grid-template-columns: 1.1fr 1.4fr 1.2fr;
    gap: 10px 14px;
  }

  .store-col {
    border: 1px solid rgba(128,128,128,0.14);
    border-radius: 14px;
    padding: 10px 12px;
    background: rgba(0,0,0,0.06);
  }

  .store-col-title {
    font-size: 12px;
    font-weight: 900;
    opacity: 0.85;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: .35px;
  }

  .store-item {
    font-size: 13px;
    line-height: 1.35;
    margin-bottom: 6px;
  }
  .store-item:last-child { margin-bottom: 0; }

  .store-item .lbl {
    opacity: 0.70;
    margin-right: 6px;
  }

  .store-item a {
    color: inherit;
    text-decoration: underline;
    opacity: 0.95;
  }

  @media (max-width: 900px) {
    .store-3col { grid-template-columns: 1fr; }
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def store_card_3col_html(
    title: str,
    subtitle: str,
    badges: list[tuple[str, str]],
    col_left: list[tuple[str, str]],
    col_mid: list[tuple[str, str]],
    col_right: list[tuple[str, str]],
    left_title: str = "Identité",
    mid_title: str = "Adresse",
    right_title: str = "Infos",
):
    def _items_html(items: list[tuple[str, str]]) -> str:
        out = []
        for lbl, val in items:
            if val in (None, "", "—"):
                continue
            out.append(f'<div class="store-item"><span class="lbl">{lbl}</span>{val}</div>')
        return "".join(out)

    badges_html = "".join([f'<span class="badge {cls}">{txt}</span>' for txt, cls in badges if txt])

    html = (
        f'<div class="store-card">'
        f'  <div class="store-head">'
        f'    <div>'
        f'      <div class="store-title">{title}</div>'
        f'      <div class="store-subtitle">{subtitle or ""}</div>'
        f'    </div>'
        f'    <div class="store-badges">{badges_html}</div>'
        f'  </div>'
        f'  <div class="store-3col">'
        f'    <div class="store-col">'
        f'      <div class="store-col-title">{left_title}</div>'
        f'      {_items_html(col_left)}'
        f'    </div>'
        f'    <div class="store-col">'
        f'      <div class="store-col-title">{mid_title}</div>'
        f'      {_items_html(col_mid)}'
        f'    </div>'
        f'    <div class="store-col">'
        f'      <div class="store-col-title">{right_title}</div>'
        f'      {_items_html(col_right)}'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# METEO assets (utilisé dans pages/1_Global.py)
# =============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ASSETS_DIR = _PROJECT_ROOT / "assets"

@st.cache_data(ttl=3600)
def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()

@st.cache_data(ttl=3600)
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _candidate_paths(filename: str) -> list[Path]:
    raw = str(filename).strip().replace("\\", "/").lstrip("/")
    if raw.lower().startswith("assets/"):
        raw = raw[7:]

    if raw.lower().startswith("meteo/"):
        base_rel = raw
    else:
        base_rel = f"meteo/{raw}"

    base_path = (_ASSETS_DIR / base_rel)

    if base_path.suffix:
        return [base_path]

    return [
        base_path.with_suffix(".svg"),
        base_path.with_suffix(".png"),
        base_path.with_suffix(".jpg"),
        base_path.with_suffix(".jpeg"),
        base_path.with_suffix(".webp"),
    ]

def st_meteo_asset(filename: str | None, width: int = 72, height: int = 110):
    if not filename:
        st.caption("—")
        return

    for p in _candidate_paths(filename):
        if p.exists():
            ext = p.suffix.lower()
            if ext == ".svg":
                svg = _read_text(p)
                components.html(
                    f'<div style="display:flex;align-items:center;justify-content:center;">{svg}</div>',
                    height=height,
                )
                return
            else:
                data = _read_bytes(p)
                st.image(data, width=width)
                return

    st.caption(f"Asset météo introuvable: {filename}")


# =============================================================================
# Fournisseurs - Normalisation (utilisé dans pages/3_Achats.py)
# =============================================================================
DEFAULT_FOURNISSEUR_ALIASES = {
    # VdV
    "EMP / VDV": "VdV",
    "EMP/VDV": "VdV",
    "EMP - VDV": "VdV",
    "EMP VDV": "VdV",
    "VDV": "VdV",
}

def normalize_fournisseur_name(name: str | None, aliases: dict[str, str] | None = None) -> str:
    if not name:
        return "Fournisseur inconnu"

    s = str(name).strip()
    if not s:
        return "Fournisseur inconnu"

    up = re.sub(r"\s+", " ", s).strip().upper()

    amap = aliases or DEFAULT_FOURNISSEUR_ALIASES
    return amap.get(up, s)
