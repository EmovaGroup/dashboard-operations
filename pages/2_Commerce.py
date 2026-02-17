# pages/2_Commerce.py
# -----------------------------------------------------------------------------
# COMMERCE — Dashboard (A vs B)
# ✅ Comparable / Non comparable / Tous appliqué partout
# ✅ Spinners custom partout ("Données en cours de chargement… merci de patienter.")
# ✅ FIX TIMEOUT : on calcule les magasins (mags) UNE SEULE FOIS -> puis ANY(%s)
# ✅ FIX CARTE : plus de hovertemplate basé sur customdata (erreur "customdata not defined")
# ✅ Tables “plus marketing” : € + % + intitulés plus propres
# ✅ FIX A=B (ex: Saint Valentin 2026 vs Saint Valentin 2026) :
#    - labels display distincts (A)/(B) pour éviter colonnes dupliquées
# -----------------------------------------------------------------------------

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

from src.auth import require_auth
from src.ui import top_bar, tabs_nav
from src.db import read_df
from src.filters import render_filters, fetch_selected_mags
from src.components import inject_kpi_compare_css, kpi_card_compare, fmt_money, fmt_int


# =============================================================================
# CONFIG — mapping code_operation -> MV OP (période) + MV POIDS OP (période)
# =============================================================================
OP_MV_PERIODE = {
    "poinsettia_2025": "public.mv_poinsettia_2025_periode_op_magasin",
    "noel_2025": "public.mv_noel_2025_periode_op_magasin",
    "noel_2024": "public.mv_noel_2024_periode_op_magasin",
    "anniversaire_2025": "public.mv_anniversaire_2025_periode_op_magasin",
    "anniversaire_2024": "public.mv_anniversaire_2024_periode_op_magasin",
    "tulipe_2026": "public.mv_tulipe_2026_periode_op_magasin",
    "tulipe_2025": "public.mv_tulipe_2025_periode_op_magasin",
}

OP_MV_POIDS = {
    "poinsettia_2025": "public.mv_poinsettia_2025_poids_op_periode_op_magasin",
    "noel_2025": "public.mv_noel_2025_poids_op_periode_op_magasin",
    "noel_2024": "public.mv_noel_2024_poids_op_periode_op_magasin",
    "anniversaire_2025": "public.mv_anniversaire_2025_poids_op_periode_op_magasin",
    "anniversaire_2024": "public.mv_anniversaire_2024_poids_op_periode_op_magasin",
    "tulipe_2026": "public.mv_tulipe_2026_poids_op_periode_op_magasin",
    "tulipe_2025": "public.mv_tulipe_2025_poids_op_periode_op_magasin",
}

# ⚠️ Ops "non produit / non nationale produit" : pas de poids OP / pas d'indicateurs produits OP
OPS_SANS_PRODUIT = {
    "st_valentin_2026",
    "st_valentin_2025",
    # ajoute ici d’autres opérations locales si besoin
}


# =============================================================================
# HELPERS
# =============================================================================
SPINNER_TXT = "Données en cours de chargement… merci de patienter."

def _safe_float(x) -> float:
    try:
        return float(x or 0)
    except Exception:
        return 0.0


def _fmt_pct(x, decimals=1) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return ""
        return f"{v:.{decimals}f} %"
    except Exception:
        return ""


def _distinct_labels_for_display(label_A: str, label_B: str):
    """
    Evite les colonnes dupliquées dans les tables quand A == B (ex: même libellé).
    """
    a = str(label_A or "").strip()
    b = str(label_B or "").strip()
    if a and (a == b):
        return f"{a} (A)", f"{b} (B)"
    return a, b


@st.cache_data(ttl=3600, show_spinner=False)
def load_france_regions_geojson():
    url = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    return r.json()


def plot_regions_plotly(df_region: pd.DataFrame, geojson: dict, label_A: str, label_B: str):
    """
    IMPORTANT:
    - Ne PAS utiliser %{customdata[...]}, car customdata n’est pas défini par défaut sur px.choropleth
    - On laisse hover_data faire le boulot (stable)
    """
    if df_region is None or df_region.empty:
        fig = px.choropleth()
        fig.update_layout(height=520)
        return fig

    if df_region["variation_CA"].notna().sum() >= 2:
        q5, q95 = np.nanpercentile(df_region["variation_CA"].dropna(), [5, 95])
    else:
        q5, q95 = -30, 30

    fig = px.choropleth(
        df_region,
        geojson=geojson,
        featureidkey="properties.nom",
        locations="region_admin",
        color="variation_CA",
        color_continuous_scale=["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9850"],
        range_color=[q5, q95],
        hover_name="region_admin",
        hover_data={
            "ca_A": ":,.0f",
            "ca_B": ":,.0f",
            "variation_CA": ":.1f",
            "region_admin": False,
        },
        labels={
            "variation_CA": "Δ CA (A vs B) (%)",
            "ca_A": f"CA {label_A} (€)",
            "ca_B": f"CA {label_B} (€)",
        },
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        height=520,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _superposed_line_by_dayindex(dfA, dfB, y_col, title, y_label, label_A, label_B):
    if dfA is None:
        dfA = pd.DataFrame(columns=["day_index", y_col])
    if dfB is None:
        dfB = pd.DataFrame(columns=["day_index", y_col])

    dA = dfA[["day_index", y_col]].copy()
    dA["Opération"] = label_A
    dB = dfB[["day_index", y_col]].copy()
    dB["Opération"] = label_B

    d = pd.concat([dA, dB], ignore_index=True)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["day_index"])

    if d.empty:
        st.info("Aucune donnée à afficher.")
        return

    fig = px.line(
        d,
        x="day_index",
        y=y_col,
        color="Opération",
        markers=True,
        title=title,
        labels={"day_index": "Jour de l’opération", y_col: y_label},
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="Opération",
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_poids_bar(poidsA: dict, poidsB: dict, label_A: str, label_B: str):
    vA = _safe_float(poidsA.get("poids_ca", 0)) * 100.0
    vB = _safe_float(poidsB.get("poids_ca", 0)) * 100.0
    qA = _safe_float(poidsA.get("poids_volume", 0)) * 100.0
    qB = _safe_float(poidsB.get("poids_volume", 0)) * 100.0

    df = pd.DataFrame(
        {
            "Opération": [label_A, label_B, label_A, label_B],
            "Type": ["Poids valeur (CA)", "Poids valeur (CA)", "Poids volume", "Poids volume"],
            "Poids (%)": [vA, vB, qA, qB],
        }
    )

    fig = px.bar(
        df,
        x="Type",
        y="Poids (%)",
        color="Opération",
        barmode="group",
        text="Poids (%)",
        title="Poids de l’opération (période) — Valeur & Volume (A vs B)",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=55, b=10),
        yaxis=dict(range=[0, max(5, df["Poids (%)"].max() * 1.25)]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _apply_comparable_scope_to_store_dfs(dfA_store: pd.DataFrame, dfB_store: pd.DataFrame, comparable_choice: str):
    comparable_choice = (comparable_choice or "Tous").upper()

    A_codes = set(dfA_store["code_magasin"].astype(str).str.strip().str.upper().tolist()) if not dfA_store.empty else set()
    B_codes = set(dfB_store["code_magasin"].astype(str).str.strip().str.upper().tolist()) if not dfB_store.empty else set()

    inter = A_codes.intersection(B_codes)
    symdiff = A_codes.symmetric_difference(B_codes)

    if comparable_choice == "C":
        dfA_store = dfA_store[dfA_store["code_magasin"].isin(list(inter))].copy()
        dfB_store = dfB_store[dfB_store["code_magasin"].isin(list(inter))].copy()
    elif comparable_choice == "NC":
        dfA_store = dfA_store[dfA_store["code_magasin"].isin(list(symdiff))].copy()
        dfB_store = dfB_store[dfB_store["code_magasin"].isin(list(symdiff))].copy()

    return dfA_store, dfB_store


def _prettify_store_perf(df: pd.DataFrame, libA: str, libB: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out = out.rename(
        columns={
            "code_magasin": "Code magasin",
            "region_admin": "Région (admin)",
            "ca_A": f"CA {libA} (€)",
            "ca_B": f"CA {libB} (€)",
            "variation_CA_pct": "Δ CA (A vs B) (%)",
        }
    )

    if f"CA {libA} (€)" in out.columns:
        out[f"CA {libA} (€)"] = out[f"CA {libA} (€)"].apply(lambda x: fmt_money(x, 0))
    if f"CA {libB} (€)" in out.columns:
        out[f"CA {libB} (€)"] = out[f"CA {libB} (€)"].apply(lambda x: fmt_money(x, 0))
    if "Δ CA (A vs B) (%)" in out.columns:
        out["Δ CA (A vs B) (%)"] = out["Δ CA (A vs B) (%)"].apply(lambda x: _fmt_pct(x, 1))

    return out


def _prettify_region_table(df_region: pd.DataFrame, libA: str, libB: str) -> pd.DataFrame:
    if df_region is None or df_region.empty:
        return df_region

    out = df_region.copy()
    out = out.rename(
        columns={
            "region_admin": "Région (admin)",
            "ca_A": f"CA {libA} (€)",
            "tickets_A": f"Tickets {libA}",
            "ca_B": f"CA {libB} (€)",
            "variation_CA": "Δ CA (A vs B) (%)",
        }
    )

    if f"CA {libA} (€)" in out.columns:
        out[f"CA {libA} (€)"] = out[f"CA {libA} (€)"].apply(lambda x: fmt_money(x, 0))
    if f"CA {libB} (€)" in out.columns:
        out[f"CA {libB} (€)"] = out[f"CA {libB} (€)"].apply(lambda x: fmt_money(x, 0))
    if f"Tickets {libA}" in out.columns:
        out[f"Tickets {libA}"] = out[f"Tickets {libA}"].apply(lambda x: fmt_int(x))
    if "Δ CA (A vs B) (%)" in out.columns:
        out["Δ CA (A vs B) (%)"] = out["Δ CA (A vs B) (%)"].apply(lambda x: _fmt_pct(x, 1))

    return out


# =============================================================================
# LOADERS (FAST) — base ANY(%s) sur liste magasins
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_totaux_commerce_from_codes(codes: list[str], date_debut: str, date_fin: str) -> dict:
    if not codes:
        return {"ca_total": 0.0, "tickets_total": 0.0, "articles_total": 0.0, "panier_moyen": 0.0, "indice_vente": 0.0}

    sql = """
select
  round(coalesce(sum(st.total_ttc_net),0), 2) as ca_total,
  round(coalesce(sum(st.nb_tickets),0), 0) as tickets_total,
  round(coalesce(sum(st.qte_article),0), 0) as articles_total,
  round(coalesce(sum(st.total_ttc_net),0) / nullif(coalesce(sum(st.nb_tickets),0),0), 2) as panier_moyen,
  round(coalesce(sum(st.qte_article),0) / nullif(coalesce(sum(st.nb_tickets),0),0), 2) as indice_vente
from public.vw_gold_tickets_jour_clean_op st
where st.ticket_date >= %s::date
  and st.ticket_date <= %s::date
  and upper(trim(st.code_magasin::text)) = any(%s);
"""
    df = read_df(sql, (date_debut, date_fin, codes))
    if df.empty:
        return {"ca_total": 0.0, "tickets_total": 0.0, "articles_total": 0.0, "panier_moyen": 0.0, "indice_vente": 0.0}

    r = df.iloc[0].to_dict()
    return {
        "ca_total": _safe_float(r.get("ca_total")),
        "tickets_total": _safe_float(r.get("tickets_total")),
        "articles_total": _safe_float(r.get("articles_total")),
        "panier_moyen": _safe_float(r.get("panier_moyen")),
        "indice_vente": _safe_float(r.get("indice_vente")),
    }


@st.cache_data(ttl=600, show_spinner=False)
def load_store_totals_for_map_from_codes(codes: list[str], date_debut: str, date_fin: str) -> pd.DataFrame:
    if not codes:
        return pd.DataFrame(columns=["code_magasin", "ca", "tickets"])

    sql = """
select
  upper(trim(st.code_magasin::text)) as code_magasin,
  coalesce(sum(st.total_ttc_net),0)::numeric as ca,
  coalesce(sum(st.nb_tickets),0)::numeric as tickets
from public.vw_gold_tickets_jour_clean_op st
where st.ticket_date >= %s::date
  and st.ticket_date <= %s::date
  and upper(trim(st.code_magasin::text)) = any(%s)
group by 1;
"""
    return read_df(sql, (date_debut, date_fin, codes))


@st.cache_data(ttl=600, show_spinner=False)
def load_series_jour_relative_from_codes(codes: list[str], date_debut: str, date_fin: str) -> pd.DataFrame:
    if not codes:
        return pd.DataFrame(columns=["day_index", "ticket_date", "ca_ttc_net", "nb_tickets", "qte_article", "panier_moyen", "indice_vente"])

    sql = """
with base as (
  select
    st.ticket_date,
    ((st.ticket_date - %s::date) + 1)::int as day_index,
    coalesce(sum(st.nb_tickets),0)::numeric as nb_tickets,
    coalesce(sum(st.qte_article),0)::numeric as qte_article,
    coalesce(sum(st.total_ttc_net),0)::numeric as ca_ttc_net
  from public.vw_gold_tickets_jour_clean_op st
  where st.ticket_date >= %s::date
    and st.ticket_date <= %s::date
    and upper(trim(st.code_magasin::text)) = any(%s)
  group by st.ticket_date, day_index
)
select
  day_index,
  ticket_date,
  ca_ttc_net,
  nb_tickets,
  qte_article,
  round(ca_ttc_net / nullif(nb_tickets,0), 2) as panier_moyen,
  round(qte_article / nullif(nb_tickets,0), 2) as indice_vente
from base
order by day_index;
"""
    return read_df(sql, (date_debut, date_debut, date_fin, codes))


@st.cache_data(ttl=600, show_spinner=False)
def load_mag_regions_from_codes(codes: list[str]) -> pd.DataFrame:
    if not codes:
        return pd.DataFrame(columns=["code_magasin", "region_admin"])

    sql = """
select
  upper(trim(rm.code_magasin::text)) as code_magasin,
  coalesce(rm."crp_:_region_nationale_d_affectation", 'Non renseigné') as region_admin
from public.ref_magasin rm
where upper(trim(rm.code_magasin::text)) = any(%s);
"""
    return read_df(sql, (codes,))


@st.cache_data(ttl=600, show_spinner=False)
def load_op_totaux_depuis_mv_periode_codes(codes: list[str], mv_periode: str) -> dict:
    if not codes:
        return {"ca_op_total": 0.0, "qte_op_total": 0.0, "tickets_op_total": 0.0}

    sql = f"""
select
  round(coalesce(sum(coalesce(total_ttc_net,0)),0), 2) as ca_op_total,
  round(coalesce(sum(coalesce(quantite_totale,0)),0), 0) as qte_op_total,
  round(coalesce(sum(coalesce(nb_tickets,0)),0), 0) as tickets_op_total
from {mv_periode}
where upper(trim(code_magasin::text)) = any(%s);
"""
    df = read_df(sql, (codes,))
    if df.empty:
        return {"ca_op_total": 0.0, "qte_op_total": 0.0, "tickets_op_total": 0.0}
    r = df.iloc[0].to_dict()
    return {
        "ca_op_total": _safe_float(r.get("ca_op_total")),
        "qte_op_total": _safe_float(r.get("qte_op_total")),
        "tickets_op_total": _safe_float(r.get("tickets_op_total")),
    }


@st.cache_data(ttl=600, show_spinner=False)
def load_poids_op_global_depuis_mv_poids_codes(codes: list[str], mv_poids: str) -> dict:
    if not codes:
        return {"poids_ca": 0.0, "poids_volume": 0.0}

    sql = f"""
select
  round(coalesce(sum(coalesce(ca_op,0)),0), 2) as ca_op_sum,
  round(coalesce(sum(coalesce(ca_total_magasin,0)),0), 2) as ca_tot_sum,
  round(coalesce(sum(coalesce(qte_op,0)),0), 0) as qte_op_sum,
  round(coalesce(sum(coalesce(qte_total_magasin,0)),0), 0) as qte_tot_sum
from {mv_poids}
where upper(trim(code_magasin::text)) = any(%s);
"""
    df = read_df(sql, (codes,))
    if df.empty:
        return {"poids_ca": 0.0, "poids_volume": 0.0}

    r = df.iloc[0].to_dict()
    ca_op = _safe_float(r.get("ca_op_sum"))
    ca_tot = _safe_float(r.get("ca_tot_sum"))
    qte_op = _safe_float(r.get("qte_op_sum"))
    qte_tot = _safe_float(r.get("qte_tot_sum"))

    return {
        "poids_ca": 0.0 if ca_tot == 0 else ca_op / ca_tot,
        "poids_volume": 0.0 if qte_tot == 0 else qte_op / qte_tot,
    }


# =============================================================================
# EXPORT HELPERS
# =============================================================================
def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep=";")
    return buf.getvalue().encode("utf-8")


# =============================================================================
# PAGE
# =============================================================================
st.set_page_config(page_title="Commerce", layout="wide")
require_auth()
top_bar("Dashboard – Commerce")
tabs_nav()
st.divider()
inject_kpi_compare_css()

ctx = render_filters()
comparable_choice = ctx.get("comparable", "Tous")

lib_opA = ctx["opA"]["lib"]
dateA0 = str(ctx["opA"]["date_debut"])
dateA1 = str(ctx["opA"]["date_fin"])
code_opA = ctx["opA"]["code"]

lib_opB = ctx["opB"]["lib"]
dateB0 = str(ctx["opB"]["date_debut"])
dateB1 = str(ctx["opB"]["date_fin"])
code_opB = ctx["opB"]["code"]

# ✅ labels d’affichage distincts si A == B (évite colonnes dupliquées)
libA_disp, libB_disp = _distinct_labels_for_display(lib_opA, lib_opB)

mags_cte_sql_A = ctx["mags_cte_sql_A"]
mags_cte_params_A = ctx["mags_cte_params_A"]
mags_cte_sql_B = ctx["mags_cte_sql_B"]
mags_cte_params_B = ctx["mags_cte_params_B"]

mvA_periode = OP_MV_PERIODE.get(code_opA)
mvB_periode = OP_MV_PERIODE.get(code_opB)
mvA_poids = OP_MV_POIDS.get(code_opA)
mvB_poids = OP_MV_POIDS.get(code_opB)

isA_sans_produit = code_opA in OPS_SANS_PRODUIT
isB_sans_produit = code_opB in OPS_SANS_PRODUIT

# ✅ FIX TIMEOUT : on récupère UNE fois la liste magasins A/B
with st.spinner(SPINNER_TXT):
    magsA_codes = fetch_selected_mags(mags_cte_sql_A, mags_cte_params_A)
    magsB_codes = fetch_selected_mags(mags_cte_sql_B, mags_cte_params_B)

nb_mag_selected_A = len(magsA_codes)
nb_mag_selected_B = len(magsB_codes)

st.markdown("## 🧩 Commerce — KPI (A vs B)")
st.caption(f"Magasins sélectionnés (parc + filtres) : **{nb_mag_selected_A}** vs **{nb_mag_selected_B}**")

# message global (une seule fois)
if (isA_sans_produit or isB_sans_produit):
    st.info("ℹ️ Cette opération n’est pas une opération nationale produit : les indicateurs 'produits OP' et les 'poids OP' ne sont pas calculés.")

st.divider()

with st.spinner(SPINNER_TXT):
    totA = load_totaux_commerce_from_codes(magsA_codes, dateA0, dateA1)
    totB = load_totaux_commerce_from_codes(magsB_codes, dateB0, dateB1)

opA = {"ca_op_total": 0.0, "qte_op_total": 0.0, "tickets_op_total": 0.0}
opB = {"ca_op_total": 0.0, "qte_op_total": 0.0, "tickets_op_total": 0.0}

# --- MV PERIODE
if mvA_periode:
    with st.spinner(SPINNER_TXT):
        opA = load_op_totaux_depuis_mv_periode_codes(magsA_codes, mvA_periode)
elif not isA_sans_produit:
    st.warning(f"MV période OP manquante pour {code_opA}. (mapping OP_MV_PERIODE)")

if mvB_periode:
    with st.spinner(SPINNER_TXT):
        opB = load_op_totaux_depuis_mv_periode_codes(magsB_codes, mvB_periode)
elif not isB_sans_produit:
    st.warning(f"MV période OP manquante pour {code_opB}. (mapping OP_MV_PERIODE)")

poidsA = {"poids_ca": 0.0, "poids_volume": 0.0}
poidsB = {"poids_ca": 0.0, "poids_volume": 0.0}

# --- MV POIDS
if mvA_poids:
    with st.spinner(SPINNER_TXT):
        poidsA = load_poids_op_global_depuis_mv_poids_codes(magsA_codes, mvA_poids)
elif not isA_sans_produit:
    st.warning(f"MV poids OP manquante pour {code_opA}. (mapping OP_MV_POIDS)")

if mvB_poids:
    with st.spinner(SPINNER_TXT):
        poidsB = load_poids_op_global_depuis_mv_poids_codes(magsB_codes, mvB_poids)
elif not isB_sans_produit:
    st.warning(f"MV poids OP manquante pour {code_opB}. (mapping OP_MV_POIDS)")


# =============================================================================
# KPI — 3 lignes de 4 KPI
# =============================================================================
pma_A = 0.0 if float(totA["articles_total"] or 0) == 0 else float(totA["ca_total"] or 0) / float(totA["articles_total"] or 1)
pma_B = 0.0 if float(totB["articles_total"] or 0) == 0 else float(totB["ca_total"] or 0) / float(totB["articles_total"] or 1)

r1 = st.columns(4)
with r1[0]:
    kpi_card_compare("CA total (€)", totA["ca_total"], totB["ca_total"], libA_disp, libB_disp, formatter=lambda x: fmt_money(x, 0))
with r1[1]:
    kpi_card_compare("Tickets total", totA["tickets_total"], totB["tickets_total"], libA_disp, libB_disp, formatter=lambda x: fmt_int(x))
with r1[2]:
    kpi_card_compare("Articles vendus", totA["articles_total"], totB["articles_total"], libA_disp, libB_disp, formatter=lambda x: fmt_int(x))
with r1[3]:
    kpi_card_compare("Panier moyen (€)", totA["panier_moyen"], totB["panier_moyen"], libA_disp, libB_disp, formatter=lambda x: fmt_money(x, 2))

r2 = st.columns(4)
with r2[0]:
    kpi_card_compare("Indice de vente (articles/ticket)", totA["indice_vente"], totB["indice_vente"], libA_disp, libB_disp, formatter=lambda x: f"{float(x or 0):.2f}")
with r2[1]:
    kpi_card_compare("Prix moyen article (€)", pma_A, pma_B, libA_disp, libB_disp, formatter=lambda x: fmt_money(x, 2))
with r2[2]:
    kpi_card_compare("CA produits OP (€)", opA["ca_op_total"], opB["ca_op_total"], libA_disp, libB_disp, formatter=lambda x: fmt_money(x, 0))
with r2[3]:
    kpi_card_compare("Tickets produits OP", opA["tickets_op_total"], opB["tickets_op_total"], libA_disp, libB_disp, formatter=lambda x: fmt_int(x))

r3 = st.columns(4)
with r3[0]:
    kpi_card_compare("Articles OP vendus", opA["qte_op_total"], opB["qte_op_total"], libA_disp, libB_disp, formatter=lambda x: fmt_int(x))
with r3[1]:
    kpi_card_compare("Poids OP valeur (%)", poidsA["poids_ca"] * 100.0, poidsB["poids_ca"] * 100.0, libA_disp, libB_disp, formatter=lambda x: f"{float(x or 0):.2f} %")
with r3[2]:
    kpi_card_compare("Poids OP volume (%)", poidsA["poids_volume"] * 100.0, poidsB["poids_volume"] * 100.0, libA_disp, libB_disp, formatter=lambda x: f"{float(x or 0):.2f} %")
with r3[3]:
    kpi_card_compare("Nb magasins (parc)", nb_mag_selected_A, nb_mag_selected_B, libA_disp, libB_disp, formatter=lambda x: fmt_int(x))

st.divider()

# =============================================================================
# COURBES — SUPERPOSITION JOUR 1..N
# =============================================================================
st.markdown("## 📈 Évolution jour (Jour 1..N) — superposition A vs B")

with st.spinner(SPINNER_TXT):
    sA = load_series_jour_relative_from_codes(magsA_codes, dateA0, dateA1)
    sB = load_series_jour_relative_from_codes(magsB_codes, dateB0, dateB1)

_superposed_line_by_dayindex(sA, sB, "ca_ttc_net", "CA TTC net / jour", "CA (€)", libA_disp, libB_disp)
_superposed_line_by_dayindex(sA, sB, "nb_tickets", "Tickets / jour", "Tickets", libA_disp, libB_disp)
_superposed_line_by_dayindex(sA, sB, "qte_article", "Articles / jour", "Articles", libA_disp, libB_disp)
_superposed_line_by_dayindex(sA, sB, "panier_moyen", "Panier moyen / jour", "Panier moyen (€)", libA_disp, libB_disp)
_superposed_line_by_dayindex(sA, sB, "indice_vente", "Indice de vente / jour", "Articles / ticket", libA_disp, libB_disp)

st.divider()

# =============================================================================
# GRAPHE POIDS OP (période) — A vs B
# =============================================================================
st.markdown("## ⚖️ Poids de l’opération (période) — Valeur & Volume")
if isA_sans_produit and isB_sans_produit:
    st.info("ℹ️ Pas d’analyse 'Poids OP' sur cette opération (pas de périmètre produit national).")
else:
    _plot_poids_bar(poidsA, poidsB, libA_disp, libB_disp)

st.divider()

# =============================================================================
# CARTE FRANCE PAR RÉGION ADMIN — variation CA (A vs B)
# =============================================================================
st.subheader("🗺️ Carte des performances régionales — Δ CA (A vs B)")

with st.spinner(SPINNER_TXT):
    dfA_store = load_store_totals_for_map_from_codes(magsA_codes, dateA0, dateA1)
    dfB_store = load_store_totals_for_map_from_codes(magsB_codes, dateB0, dateB1)

    dfA_store, dfB_store = _apply_comparable_scope_to_store_dfs(dfA_store, dfB_store, comparable_choice)

    codes_union = sorted(
        set(dfA_store["code_magasin"].astype(str).tolist()).union(set(dfB_store["code_magasin"].astype(str).tolist()))
    )
    df_mag = load_mag_regions_from_codes(codes_union)

df_region_A = (
    dfA_store.merge(df_mag, on="code_magasin", how="left")
    .groupby("region_admin", as_index=False)
    .agg(ca_A=("ca", "sum"), tickets_A=("tickets", "sum"))
)

df_region_B = (
    dfB_store.merge(df_mag, on="code_magasin", how="left")
    .groupby("region_admin", as_index=False)
    .agg(ca_B=("ca", "sum"))
)

df_region = df_region_A.merge(df_region_B, on="region_admin", how="outer").fillna(0)
df_region["variation_CA"] = np.where(
    df_region["ca_B"] > 0,
    (df_region["ca_A"] - df_region["ca_B"]) / df_region["ca_B"] * 100,
    np.nan,
)

try:
    with st.spinner(SPINNER_TXT):
        geojson = load_france_regions_geojson()
    fig_map_regions = plot_regions_plotly(df_region, geojson, libA_disp, libB_disp)
    st.plotly_chart(fig_map_regions, use_container_width=True)
    st.caption("Vert = progression / Rouge = baisse. Survolez pour le détail.")
except Exception as e:
    st.warning(f"Impossible d'afficher la carte (GeoJSON): {e}")
    st.dataframe(_prettify_region_table(df_region, libA_disp, libB_disp), use_container_width=True, hide_index=True)

with st.expander("📋 Détail par région (table)"):
    st.dataframe(_prettify_region_table(df_region, libA_disp, libB_disp), use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# TOP 10 magasins — meilleurs / pires perf (variation CA %)
# =============================================================================
st.subheader("🏪 Top magasins — meilleurs & pires Δ CA (A vs B)")

join_mode = "inner" if (comparable_choice or "Tous").upper() == "C" else "outer"

df_store_perf = (
    dfA_store.rename(columns={"ca": "ca_A", "tickets": "tickets_A"})
    .merge(
        dfB_store.rename(columns={"ca": "ca_B", "tickets": "tickets_B"}),
        on="code_magasin",
        how=join_mode,
    )
    .merge(df_mag, on="code_magasin", how="left")
    .fillna(0)
)

df_store_perf["variation_CA_pct"] = np.where(
    df_store_perf["ca_B"] > 0,
    (df_store_perf["ca_A"] - df_store_perf["ca_B"]) / df_store_perf["ca_B"] * 100,
    np.nan,
)

df_store_perf["ca_A"] = df_store_perf["ca_A"].astype(float)
df_store_perf["ca_B"] = df_store_perf["ca_B"].astype(float)
df_store_perf["variation_CA_pct"] = df_store_perf["variation_CA_pct"].astype(float)

df_store_perf = df_store_perf[~((df_store_perf["ca_A"] == 0) & (df_store_perf["ca_B"] == 0))].copy()
df_valid = df_store_perf.dropna(subset=["variation_CA_pct"]).copy()

cols_show = ["code_magasin", "region_admin", "ca_A", "ca_B", "variation_CA_pct"]

df_best = df_valid.sort_values("variation_CA_pct", ascending=False).head(10)[cols_show]
df_worst = df_valid.sort_values("variation_CA_pct", ascending=True).head(10)[cols_show]

c_top = st.columns(2)
with c_top[0]:
    st.markdown("### ✅ Top 10 (meilleurs)")
    st.dataframe(_prettify_store_perf(df_best, libA_disp, libB_disp), use_container_width=True, hide_index=True)
with c_top[1]:
    st.markdown("### ❌ Top 10 (pires)")
    st.dataframe(_prettify_store_perf(df_worst, libA_disp, libB_disp), use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# EXPORT CSV
# =============================================================================
st.markdown("## 📤 Export (CSV) — données affichées")

df_kpi_export = pd.DataFrame(
    [
        {"bloc": "kpi", "kpi": "ca_total", "op": "A", "value": totA["ca_total"]},
        {"bloc": "kpi", "kpi": "ca_total", "op": "B", "value": totB["ca_total"]},
        {"bloc": "kpi", "kpi": "tickets_total", "op": "A", "value": totA["tickets_total"]},
        {"bloc": "kpi", "kpi": "tickets_total", "op": "B", "value": totB["tickets_total"]},
        {"bloc": "kpi", "kpi": "articles_total", "op": "A", "value": totA["articles_total"]},
        {"bloc": "kpi", "kpi": "articles_total", "op": "B", "value": totB["articles_total"]},
        {"bloc": "kpi", "kpi": "panier_moyen", "op": "A", "value": totA["panier_moyen"]},
        {"bloc": "kpi", "kpi": "panier_moyen", "op": "B", "value": totB["panier_moyen"]},
        {"bloc": "kpi", "kpi": "indice_vente", "op": "A", "value": totA["indice_vente"]},
        {"bloc": "kpi", "kpi": "indice_vente", "op": "B", "value": totB["indice_vente"]},
        {"bloc": "kpi", "kpi": "prix_moyen_article", "op": "A", "value": pma_A},
        {"bloc": "kpi", "kpi": "prix_moyen_article", "op": "B", "value": pma_B},
        {"bloc": "kpi_op", "kpi": "ca_op_total", "op": "A", "value": opA["ca_op_total"]},
        {"bloc": "kpi_op", "kpi": "ca_op_total", "op": "B", "value": opB["ca_op_total"]},
        {"bloc": "kpi_op", "kpi": "tickets_op_total", "op": "A", "value": opA["tickets_op_total"]},
        {"bloc": "kpi_op", "kpi": "tickets_op_total", "op": "B", "value": opB["tickets_op_total"]},
        {"bloc": "kpi_op", "kpi": "qte_op_total", "op": "A", "value": opA["qte_op_total"]},
        {"bloc": "kpi_op", "kpi": "qte_op_total", "op": "B", "value": opB["qte_op_total"]},
        {"bloc": "poids", "kpi": "poids_ca", "op": "A", "value": poidsA["poids_ca"]},
        {"bloc": "poids", "kpi": "poids_ca", "op": "B", "value": poidsB["poids_ca"]},
        {"bloc": "poids", "kpi": "poids_volume", "op": "A", "value": poidsA["poids_volume"]},
        {"bloc": "poids", "kpi": "poids_volume", "op": "B", "value": poidsB["poids_volume"]},
        {"bloc": "parc", "kpi": "nb_mag_selected", "op": "A", "value": nb_mag_selected_A},
        {"bloc": "parc", "kpi": "nb_mag_selected", "op": "B", "value": nb_mag_selected_B},
        {"bloc": "filtre", "kpi": "comparable", "op": "ALL", "value": comparable_choice},
    ]
)

df_series_export = pd.concat([sA.assign(operation=lib_opA), sB.assign(operation=lib_opB)], ignore_index=True)

df_region_export = df_region.copy()
df_region_export["opA"] = lib_opA
df_region_export["opB"] = lib_opB
df_region_export["comparable"] = comparable_choice

df_store_export = df_store_perf.copy()
df_store_export["opA"] = lib_opA
df_store_export["opB"] = lib_opB
df_store_export["comparable"] = comparable_choice

c_exp = st.columns(4)
with c_exp[0]:
    st.download_button("⬇️ KPI (CSV)", data=_df_to_csv_bytes(df_kpi_export), file_name="commerce_kpi.csv", mime="text/csv", use_container_width=True)
with c_exp[1]:
    st.download_button("⬇️ Séries jour (CSV)", data=_df_to_csv_bytes(df_series_export), file_name="commerce_series_jour.csv", mime="text/csv", use_container_width=True)
with c_exp[2]:
    st.download_button("⬇️ Régions (CSV)", data=_df_to_csv_bytes(df_region_export), file_name="commerce_regions.csv", mime="text/csv", use_container_width=True)
with c_exp[3]:
    st.download_button("⬇️ Magasins (CSV)", data=_df_to_csv_bytes(df_store_export), file_name="commerce_magasins.csv", mime="text/csv", use_container_width=True)

with st.expander("Voir les tables exportées"):
    st.markdown("### KPI")
    st.dataframe(df_kpi_export, use_container_width=True, hide_index=True)
    st.markdown("### Séries jour")
    st.dataframe(df_series_export, use_container_width=True, hide_index=True)
    st.markdown("### Régions")
    st.dataframe(df_region_export, use_container_width=True, hide_index=True)
    st.markdown("### Magasins (avec perf)")
    st.dataframe(df_store_export, use_container_width=True, hide_index=True)
