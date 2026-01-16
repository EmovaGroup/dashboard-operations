# src/ui.py
import base64
import re
from pathlib import Path

import streamlit as st
from .auth import logout

# -----------------------------------------------------------------------------
# Assets
# -----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ASSETS_DIR = _PROJECT_ROOT / "assets"

# Mets ton logo ici (dans /assets) :
#   assets/logo_emova_group.png
_LOGO_FILENAME = "logo_emova_group.png"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _extract_lastname_from_email(email: str | None) -> str:

    if not email:
        return "!"

    local = email.split("@")[0].strip().lower()
    if not local:
        return "!"

    # si on a "sa.ouni" -> prend "ouni"
    parts = [p for p in local.split(".") if p]
    candidate = parts[-1] if parts else local

    # nettoie (au cas où) : enlève chiffres / caractères chelous
    candidate = re.sub(r"[^a-z\-]", "", candidate).strip()
    if not candidate:
        return "!"

    return candidate.upper()


@st.cache_data(ttl=3600)
def _logo_data_uri() -> str | None:
    """
    Retourne une data URI (base64) pour afficher le logo en HTML,
    ce qui permet un alignement propre (logo + bienvenue sur la même ligne).
    """
    p = _ASSETS_DIR / _LOGO_FILENAME
    if not p.exists():
        return None

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = p.suffix.lower().lstrip(".") or "png"
    return f"data:image/{ext};base64,{b64}"


def _inject_topbar_css():
    st.markdown(
        """
<style>
  .topbar-wrap {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:14px;
    margin-top: 4px;
    margin-bottom: 6px;
  }

  .topbar-left {
    display:flex;
    align-items:center;
    gap:14px;
    min-width: 0;
  }

  .topbar-logo {
    height: 42px;
    width: auto;
    display:block;
  }

  .topbar-welcome {
    font-weight: 900;
    font-size: 22px;
    line-height: 1;
    color: #1f6feb; /* bleu lisible UX */
    white-space: nowrap;
  }

  .topbar-spacer {
    height: 6px;
  }

  /* petite adaptation mobile */
  @media (max-width: 720px) {
    .topbar-welcome { font-size: 18px; }
    .topbar-logo { height: 36px; }
  }
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def top_bar(title: str):
    """
    Header commun :
    - Ligne 1 : Logo + "👋 Bienvenue NOM !" (bleu, gros, bien aligné) + Déconnexion tout à droite
    - Ligne 2 : Titre "Dashboard – ..."
    - Pas d'affichage du mail
    - Compatible light/dark (aucun fond/couleur imposé côté conteneur)
    """
    _inject_topbar_css()

    user = st.session_state.get("sb_user", {})
    email = user.get("email", "")
    lastname = _extract_lastname_from_email(email)

    logo_uri = _logo_data_uri()

    # on fait 2 colonnes : gauche (logo + bienvenue), droite (déconnexion)
    left, right = st.columns([6, 2], vertical_alignment="center")

    with left:
        if logo_uri:
            st.markdown(
                f"""
                <div class="topbar-wrap" style="justify-content:flex-start;">
                  <div class="topbar-left">
                    <img class="topbar-logo" src="{logo_uri}" alt="logo" />
                    <div class="topbar-welcome">👋&nbsp;Bienvenue&nbsp;{lastname}&nbsp;!</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # fallback si logo absent
            st.markdown(
                f"""
                <div class="topbar-wrap" style="justify-content:flex-start;">
                  <div class="topbar-left">
                    <div class="topbar-welcome">👋&nbsp;Bienvenue&nbsp;{lastname}&nbsp;!</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        # bouton à droite, bien aligné
        if st.button("Déconnexion", use_container_width=True):
            logout()

    # Titre natif (s’adapte au thème)
    st.title(title)


def tabs_nav():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Global", use_container_width=True):
            st.switch_page("pages/1_Global.py")
    with c2:
        if st.button("Commerce", use_container_width=True):
            st.switch_page("pages/2_Commerce.py")
    with c3:
        if st.button("Achats", use_container_width=True):
            st.switch_page("pages/3_Achats.py")
    with c4:
        if st.button("Marketing", use_container_width=True):
            st.switch_page("pages/4_Marketing.py")


def under_construction(page_name: str):
    """
    Bloc "en construction" lisible en dark ET en light.
    -> pas de texte blanc forcé
    -> on utilise 'color: inherit' et on gère juste des fonds neutres
    """
    st.divider()
    st.markdown(
        f"""
        <style>
          .uc-wrap {{
            text-align:center;
            padding:70px 20px;
            background: rgba(127,127,127,0.08);
            border: 1px solid rgba(127,127,127,0.22);
            border-radius:16px;
            margin-top:30px;
            color: inherit;
          }}
          .uc-title {{
            font-size:34px;
            font-weight:900;
            margin-top:10px;
            color: inherit;
          }}
          .uc-sub {{
            font-size:20px;
            margin-top:8px;
            color: inherit;
            opacity: 0.75;
          }}
          .uc-body {{
            font-size:16px;
            margin-top:18px;
            color: inherit;
            opacity: 0.65;
          }}
        </style>

        <div class="uc-wrap">
          <div style="font-size:44px; line-height:1;">🚧</div>
          <div class="uc-title">{page_name}</div>
          <div class="uc-sub">Page en cours de construction</div>
          <div class="uc-body">
            Cette section sera disponible prochainement.<br/>
            Merci de votre patience.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
