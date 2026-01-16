# pages/Marketing.py
# -----------------------------------------------------------------------------
# Page : Marketing
# - Auth + top bar + tabs
# - Placeholder "en construction" qui s'adapte automatiquement au thème Streamlit
# -----------------------------------------------------------------------------
import streamlit as st

from src.auth import require_auth
from src.ui import top_bar, tabs_nav

st.set_page_config(page_title="Marketing", layout="wide")
require_auth()

top_bar("Dashboard – Marketing")
tabs_nav()
st.divider()

# Placeholder robuste (light/dark) : composant natif Streamlit
st.markdown("## 🚧 Marketing")
st.info(
    "Cette page est en cours de construction.\n\n"
    "➡️ L’objectif : centraliser les KPI marketing (budget, canaux, perf OP, etc.)."
)

# Petit espace pour éviter l’effet “vide”
st.caption("💡 Astuce : en attendant, utilise l’onglet **Global** / **Achats** pour suivre les opérations.")
