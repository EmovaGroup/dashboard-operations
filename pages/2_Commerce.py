# pages/Commerce.py
# -----------------------------------------------------------------------------
# Page : Commerce
# - Auth + top bar + tabs
# - Placeholder "en construction" qui s'adapte automatiquement au thème Streamlit
# -----------------------------------------------------------------------------
import streamlit as st

from src.auth import require_auth
from src.ui import top_bar, tabs_nav

st.set_page_config(page_title="Commerce", layout="wide")
require_auth()

top_bar("Dashboard – Commerce")
tabs_nav()
st.divider()

# Placeholder robuste (light/dark) : composant natif Streamlit
st.markdown("## 🚧 Commerce")
st.info(
    "Cette page est en cours de construction.\n\n"
    "➡️ L’objectif : piloter les KPI commerce (CA, tickets, panier, top magasins, perf OP, etc.)."
)

st.caption("💡 Astuce : en attendant, utilise l’onglet **Global** / **Achats** pour suivre les opérations.")
