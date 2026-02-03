# pages/2_Commerce.py
# -----------------------------------------------------------------------------
# COMMERCE — Dashboard (A vs B)
#
# ✅ Aligné avec la “même manip” que Global :
# - UN SEUL parc filtré (ctx["mags_cte_sql"] / ctx["mags_cte_params"]) pour A et B
# - KPI calculés sur vw_gold_tickets_jour_clean_op sur la période de l’opération
#
# ✅ Ajouts demandés :
# - KPI “Nb magasins sélectionnés” (sur le parc filtré)
# - Export CSV tout en bas : toutes les données affichées (KPI + séries + poids + carte)
#
# ✅ Tulipe (2025 / 2026) :
# - Branché sur tes MV :
#   - public.mv_tulipe_2025_periode_op_magasin
#   - public.mv_tulipe_2026_periode_op_magasin
#   - public.mv_tulipe_2025_poids_op_periode_op_magasin
#   - public.mv_tulipe_2026_poids_op_periode_op_magasin
#
# ✅ Modifs KPI demandées :
# - Afficher les formules entre parenthèses
# - Ajouter “Prix moyen article” = CA total / Nb articles vendus
# - 3 lignes de 4 KPI
# - “Nb magasins sélectionnés” tout à droite dans la 3ème ligne
#
# ✅ Ajout :
# - Top 10 meilleurs magasins (perf) + Top 10 pires magasins (perf)
#   (perf = variation CA % calculée comme sur la carte : (CA A - CA B) / CA B * 100)
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
from src.filters import render_filters
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

    # ✅ Tulipe
    "tulipe_2026": "public.mv_tulipe_2026_periode_op_magasin",
    "tulipe_2025": "public.mv_tulipe_2025_periode_op_magasin",
}

OP_MV_POIDS = {
    "poinsettia_2025": "public.mv_poinsettia_2025_poids_op_periode_op_magasin",
    "noel_2025": "public.mv_noel_2025_poids_op_periode_op_magasin",
    "noel_2024": "public.mv_noel_2024_poids_op_periode_op_magasin",
    "anniversaire_2025": "public.mv_anniversaire_2025_poids_op_periode_op_magasin",
    "anniversaire_2024": "public.mv_anniversaire_2024_poids_op_periode_op_magasin",

    # ✅ Tulipe
    "tulipe_2026": "public.mv_tulipe_2026_poids_op_periode_op_magasin",
    "tulipe_2025": "public.mv_tulipe_2025_poids_op_periode_op_magasin",
}


# =============================================================================
# HELPERS
# =============================================================================
def _safe_float(x) -> float:
    try:
        return float(x or 0)
    except Exception:
        return 0.0


@st.cache_data(ttl=3600)
def load_france_regions_geojson():
    url = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    return r.json()


def plot_regions_plotly(df_region: pd.DataFrame, geojson: dict):
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
        hover_data={"ca_A": True, "ca_B": True, "variation_CA": True},
        labels={"variation_CA": "Δ CA (A vs B) (%)", "ca_A": "CA A (€)", "ca_B": "CA B (€)"},
    )

    fig.update_traces(
        hovertemplate=(
            "<div style='background-color:#f5f5f5; padding:8px 10px; border-radius:8px; "
            "border:1px solid #ddd; color:#111; font-size:13px;'>"
            "<b>%{location}</b><br>"
            "CA A = %{customdata[0]:,.0f} €<br>"
            "CA B = %{customdata[1]:,.0f} €<br>"
            "Variation (A vs B) = %{customdata[2]:+.1f} %"
            "</div><extra></extra>"
        )
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        height=520,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _superposed_line_by_dayindex(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    label_A: str,
    label_B: str,
):
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


# =============================================================================
# LOADERS (même logique Global : 1 parc filtré)
# =============================================================================
@st.cache_data(ttl=600)
def count_selected_mags(mags_cte_sql_: str, mags_cte_params_: tuple) -> int:
    sql = f"""
{mags_cte_sql_}
select count(*)::int as nb_mag
from mags;
"""
    df = read_df(sql, mags_cte_params_)
    if df.empty:
        return 0
    return int(df.iloc[0]["nb_mag"] or 0)


@st.cache_data(ttl=600)
def load_totaux_commerce(mags_cte_sql_: str, mags_cte_params_: tuple, date_debut: str, date_fin: str) -> dict:
    sql = f"""
{mags_cte_sql_},

base as (
  select
    trim(st.code_magasin::text) as code_magasin,
    st.ticket_date,
    coalesce(st.nb_tickets, 0)::numeric as nb_tickets,
    coalesce(st.qte_article, 0)::numeric as qte_article,
    coalesce(st.total_ttc_net, 0)::numeric as ca_ttc_net
  from public.vw_gold_tickets_jour_clean_op st
  join mags m on m.code_magasin = trim(st.code_magasin::text)
  where st.ticket_date >= %s::date
    and st.ticket_date <= %s::date
)
select
  round(coalesce(sum(ca_ttc_net),0), 2) as ca_total,
  round(coalesce(sum(nb_tickets),0), 0) as tickets_total,
  round(coalesce(sum(qte_article),0), 0) as articles_total,
  round(coalesce(sum(ca_ttc_net),0) / nullif(coalesce(sum(nb_tickets),0),0), 2) as panier_moyen,
  round(coalesce(sum(qte_article),0) / nullif(coalesce(sum(nb_tickets),0),0), 2) as indice_vente
from base;
"""
    df = read_df(sql, tuple(list(mags_cte_params_) + [date_debut, date_fin]))
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


@st.cache_data(ttl=600)
def load_op_totaux_depuis_mv_periode(mags_cte_sql_: str, mags_cte_params_: tuple, mv_periode: str) -> dict:
    sql = f"""
{mags_cte_sql_},

op as (
  select
    trim(code_magasin::text) as code_magasin,
    coalesce(total_ttc_net, 0)::numeric as ca_op,
    coalesce(quantite_totale, 0)::numeric as qte_op,
    coalesce(nb_tickets, 0)::numeric as tickets_op
  from {mv_periode}
),
op_m as (
  select o.* from op o join mags m using(code_magasin)
)
select
  round(coalesce(sum(ca_op),0), 2) as ca_op_total,
  round(coalesce(sum(qte_op),0), 0) as qte_op_total,
  round(coalesce(sum(tickets_op),0), 0) as tickets_op_total
from op_m;
"""
    df = read_df(sql, mags_cte_params_)
    if df.empty:
        return {"ca_op_total": 0.0, "qte_op_total": 0.0, "tickets_op_total": 0.0}

    r = df.iloc[0].to_dict()
    return {
        "ca_op_total": _safe_float(r.get("ca_op_total")),
        "qte_op_total": _safe_float(r.get("qte_op_total")),
        "tickets_op_total": _safe_float(r.get("tickets_op_total")),
    }


@st.cache_data(ttl=600)
def load_poids_op_global_depuis_mv_poids(mags_cte_sql_: str, mags_cte_params_: tuple, mv_poids: str) -> dict:
    sql = f"""
{mags_cte_sql_},

p as (
  select
    trim(code_magasin::text) as code_magasin,
    coalesce(ca_op, 0)::numeric as ca_op,
    coalesce(qte_op, 0)::numeric as qte_op,
    coalesce(ca_total_magasin, 0)::numeric as ca_total_magasin,
    coalesce(qte_total_magasin, 0)::numeric as qte_total_magasin
  from {mv_poids}
),
p_m as (
  select p.* from p join mags m using(code_magasin)
)
select
  round(coalesce(sum(ca_op),0), 2) as ca_op_sum,
  round(coalesce(sum(ca_total_magasin),0), 2) as ca_tot_sum,
  round(coalesce(sum(qte_op),0), 0) as qte_op_sum,
  round(coalesce(sum(qte_total_magasin),0), 0) as qte_tot_sum
from p_m;
"""
    df = read_df(sql, mags_cte_params_)
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


@st.cache_data(ttl=600)
def load_series_jour_relative(mags_cte_sql_: str, mags_cte_params_: tuple, date_debut: str, date_fin: str) -> pd.DataFrame:
    sql = f"""
{mags_cte_sql_},

base as (
  select
    st.ticket_date,
    ((st.ticket_date - %s::date) + 1)::int as day_index,
    coalesce(sum(st.nb_tickets),0)::numeric as nb_tickets,
    coalesce(sum(st.qte_article),0)::numeric as qte_article,
    coalesce(sum(st.total_ttc_net),0)::numeric as ca_ttc_net
  from public.vw_gold_tickets_jour_clean_op st
  join mags m on m.code_magasin = trim(st.code_magasin::text)
  where st.ticket_date >= %s::date
    and st.ticket_date <= %s::date
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
    params = tuple(list(mags_cte_params_) + [date_debut, date_debut, date_fin])
    return read_df(sql, params)


@st.cache_data(ttl=600)
def load_store_totals_for_map(mags_cte_sql_: str, mags_cte_params_: tuple, date_debut: str, date_fin: str) -> pd.DataFrame:
    sql = f"""
{mags_cte_sql_},

base as (
  select
    trim(st.code_magasin::text) as code_magasin,
    coalesce(sum(st.total_ttc_net),0)::numeric as ca,
    coalesce(sum(st.nb_tickets),0)::numeric as tickets
  from public.vw_gold_tickets_jour_clean_op st
  join mags m on m.code_magasin = trim(st.code_magasin::text)
  where st.ticket_date >= %s::date
    and st.ticket_date <= %s::date
  group by trim(st.code_magasin::text)
)
select * from base;
"""
    return read_df(sql, tuple(list(mags_cte_params_) + [date_debut, date_fin]))


@st.cache_data(ttl=600)
def load_mag_regions(mags_cte_sql_: str, mags_cte_params_: tuple) -> pd.DataFrame:
    sql = f"""
{mags_cte_sql_}
select
  m.code_magasin,
  coalesce(rm."crp_:_region_nationale_d_affectation", 'Non renseigné') as region_admin
from mags m
left join public.ref_magasin rm on trim(rm.code_magasin::text) = m.code_magasin;
"""
    return read_df(sql, mags_cte_params_)


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

code_opA = ctx["opA"]["code"]
lib_opA = ctx["opA"]["lib"]
dateA0 = str(ctx["opA"]["date_debut"])
dateA1 = str(ctx["opA"]["date_fin"])

code_opB = ctx["opB"]["code"]
lib_opB = ctx["opB"]["lib"]
dateB0 = str(ctx["opB"]["date_debut"])
dateB1 = str(ctx["opB"]["date_fin"])

# ✅ MÊME MANIP GLOBAL : UN SEUL parc filtré pour A et B
mags_cte_sql = ctx["mags_cte_sql"]
mags_cte_params = ctx["mags_cte_params"]

mvA_periode = OP_MV_PERIODE.get(code_opA)
mvB_periode = OP_MV_PERIODE.get(code_opB)
mvA_poids = OP_MV_POIDS.get(code_opA)
mvB_poids = OP_MV_POIDS.get(code_opB)

# ✅ nb magasins sélectionnés (parc filtré)
nb_mag_selected = count_selected_mags(mags_cte_sql, mags_cte_params)

st.markdown("## 🛒 Commerce — KPI (A vs B)")
st.caption(f"Magasins sélectionnés (parc filtré) : **{nb_mag_selected}**")
st.divider()

# ✅ totaux calculés sur le même parc (Global-like)
totA = load_totaux_commerce(mags_cte_sql, mags_cte_params, dateA0, dateA1)
totB = load_totaux_commerce(mags_cte_sql, mags_cte_params, dateB0, dateB1)

opA = {"ca_op_total": 0.0, "qte_op_total": 0.0, "tickets_op_total": 0.0}
opB = {"ca_op_total": 0.0, "qte_op_total": 0.0, "tickets_op_total": 0.0}

if mvA_periode:
    opA = load_op_totaux_depuis_mv_periode(mags_cte_sql, mags_cte_params, mvA_periode)
else:
    st.warning(f"MV période OP manquante pour {code_opA}. Ajoute-la dans OP_MV_PERIODE.")

if mvB_periode:
    opB = load_op_totaux_depuis_mv_periode(mags_cte_sql, mags_cte_params, mvB_periode)
else:
    st.warning(f"MV période OP manquante pour {code_opB}. Ajoute-la dans OP_MV_PERIODE.")

poidsA = {"poids_ca": 0.0, "poids_volume": 0.0}
poidsB = {"poids_ca": 0.0, "poids_volume": 0.0}

if mvA_poids:
    poidsA = load_poids_op_global_depuis_mv_poids(mags_cte_sql, mags_cte_params, mvA_poids)
else:
    st.warning(f"MV poids OP manquante pour {code_opA}. Ajoute-la dans OP_MV_POIDS.")

if mvB_poids:
    poidsB = load_poids_op_global_depuis_mv_poids(mags_cte_sql, mags_cte_params, mvB_poids)
else:
    st.warning(f"MV poids OP manquante pour {code_opB}. Ajoute-la dans OP_MV_POIDS.")


# =============================================================================
# KPI — 3 lignes de 4 KPI (avec formules + prix moyen article)
# =============================================================================
KPI_CA = "CA total"
KPI_TICKETS = "Tickets total"
KPI_ARTICLES = "Nb articles vendus"
KPI_PANIER = "Panier moyen (CA total / Tickets total)"

KPI_INDICE = "Indice de vente (Nb articles / Tickets total)"
KPI_PRIX_ART = "Prix moyen article (CA total / Nb articles vendus)"
KPI_CA_OP = "CA produits OP"
KPI_TICKETS_OP = "Tickets produits OP"

KPI_ARTICLES_OP = "Nb articles OP vendus"
KPI_POIDS_CA = "Poids OP valeur (CA OP / CA total magasin)"
KPI_POIDS_VOL = "Poids OP volume (Qte OP / Qte total magasin)"
KPI_NB_MAG = "Nb magasins sélectionnés"

# Prix moyen article = CA total / Nb articles vendus (calculé côté Python)
pma_A = 0.0 if float(totA["articles_total"] or 0) == 0 else float(totA["ca_total"] or 0) / float(totA["articles_total"] or 1)
pma_B = 0.0 if float(totB["articles_total"] or 0) == 0 else float(totB["ca_total"] or 0) / float(totB["articles_total"] or 1)

# Ligne 1 (4)
r1 = st.columns(4)
with r1[0]:
    kpi_card_compare(KPI_CA, totA["ca_total"], totB["ca_total"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 0))
with r1[1]:
    kpi_card_compare(KPI_TICKETS, totA["tickets_total"], totB["tickets_total"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
with r1[2]:
    kpi_card_compare(KPI_ARTICLES, totA["articles_total"], totB["articles_total"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
with r1[3]:
    kpi_card_compare(KPI_PANIER, totA["panier_moyen"], totB["panier_moyen"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))

# Ligne 2 (4)
r2 = st.columns(4)
with r2[0]:
    kpi_card_compare(KPI_INDICE, totA["indice_vente"], totB["indice_vente"], lib_opA, lib_opB, formatter=lambda x: f"{float(x or 0):.2f}")
with r2[1]:
    kpi_card_compare(KPI_PRIX_ART, pma_A, pma_B, lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))
with r2[2]:
    kpi_card_compare(KPI_CA_OP, opA["ca_op_total"], opB["ca_op_total"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 0))
with r2[3]:
    kpi_card_compare(KPI_TICKETS_OP, opA["tickets_op_total"], opB["tickets_op_total"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))

# Ligne 3 (4) — Nb magasins sélectionnés tout à droite
r3 = st.columns(4)
with r3[0]:
    kpi_card_compare(KPI_ARTICLES_OP, opA["qte_op_total"], opB["qte_op_total"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
with r3[1]:
    kpi_card_compare(KPI_POIDS_CA, poidsA["poids_ca"] * 100.0, poidsB["poids_ca"] * 100.0, lib_opA, lib_opB, formatter=lambda x: f"{float(x or 0):.2f} %")
with r3[2]:
    kpi_card_compare(KPI_POIDS_VOL, poidsA["poids_volume"] * 100.0, poidsB["poids_volume"] * 100.0, lib_opA, lib_opB, formatter=lambda x: f"{float(x or 0):.2f} %")
with r3[3]:
    kpi_card_compare(KPI_NB_MAG, nb_mag_selected, nb_mag_selected, lib_opA, lib_opB, formatter=lambda x: fmt_int(x))

st.divider()

# =============================================================================
# COURBES — SUPERPOSITION JOUR 1..N (même parc)
# =============================================================================
st.markdown("## 📈 Évolution jour — superposition (Jour 1..N)")

sA = load_series_jour_relative(mags_cte_sql, mags_cte_params, dateA0, dateA1)
sB = load_series_jour_relative(mags_cte_sql, mags_cte_params, dateB0, dateB1)

_superposed_line_by_dayindex(sA, sB, "ca_ttc_net", "CA TTC net / jour — superposition A vs B", "CA (€)", lib_opA, lib_opB)
_superposed_line_by_dayindex(sA, sB, "nb_tickets", "Tickets / jour — superposition A vs B", "Tickets", lib_opA, lib_opB)
_superposed_line_by_dayindex(sA, sB, "qte_article", "Nb articles / jour — superposition A vs B", "Articles", lib_opA, lib_opB)
_superposed_line_by_dayindex(sA, sB, "panier_moyen", "Panier moyen / jour — superposition A vs B", "Panier moyen (€)", lib_opA, lib_opB)
_superposed_line_by_dayindex(sA, sB, "indice_vente", "Indice de vente / jour — superposition A vs B", "Articles / ticket", lib_opA, lib_opB)

st.divider()

# =============================================================================
# GRAPHE POIDS OP (période) — A vs B
# =============================================================================
st.markdown("## ⚖️ Poids de l’opération (période) — A vs B")
_plot_poids_bar(poidsA, poidsB, lib_opA, lib_opB)

st.divider()

# =============================================================================
# CARTE FRANCE PAR RÉGION ADMIN (A vs B) — variation CA (même parc)
# =============================================================================
st.subheader("🗺️ Carte des performances régionales – variation CA (A vs B)")

dfA_store = load_store_totals_for_map(mags_cte_sql, mags_cte_params, dateA0, dateA1)
dfB_store = load_store_totals_for_map(mags_cte_sql, mags_cte_params, dateB0, dateB1)
df_mag = load_mag_regions(mags_cte_sql, mags_cte_params)

df_region_A = (
    dfA_store.merge(df_mag[["code_magasin", "region_admin"]], on="code_magasin", how="left")
    .groupby("region_admin", as_index=False)
    .agg(ca_A=("ca", "sum"), tickets_A=("tickets", "sum"))
)

df_region_B = (
    dfB_store.merge(df_mag[["code_magasin", "region_admin"]], on="code_magasin", how="left")
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
    geojson = load_france_regions_geojson()
    fig_map_regions = plot_regions_plotly(df_region, geojson)
    st.plotly_chart(fig_map_regions, use_container_width=True)
    st.caption("Vert = progression positive / Rouge = baisse. Survolez pour les valeurs.")
except Exception as e:
    st.warning(f"Impossible d'afficher la carte (GeoJSON): {e}")
    show = df_region.copy()
    show = show.rename(
        columns={
            "region_admin": "Région (admin)",
            "ca_A": f"CA {lib_opA}",
            "ca_B": f"CA {lib_opB}",
            "tickets_A": f"Tickets {lib_opA}",
            "variation_CA": "Variation CA (A vs B) %",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# TOP 10 magasins — meilleurs / pires perf (variation CA %)
# =============================================================================
st.subheader("🏪 Top magasins — meilleurs & pires perf (A vs B)")

df_store_perf = (
    dfA_store.rename(columns={"ca": "ca_A", "tickets": "tickets_A"})
    .merge(dfB_store.rename(columns={"ca": "ca_B", "tickets": "tickets_B"}), on="code_magasin", how="outer")
    .merge(df_mag, on="code_magasin", how="left")
    .fillna(0)
)

df_store_perf["variation_CA_pct"] = np.where(
    df_store_perf["ca_B"] > 0,
    (df_store_perf["ca_A"] - df_store_perf["ca_B"]) / df_store_perf["ca_B"] * 100,
    np.nan,
)

# Optionnel : arrondis pour lecture
df_store_perf["ca_A"] = df_store_perf["ca_A"].astype(float)
df_store_perf["ca_B"] = df_store_perf["ca_B"].astype(float)
df_store_perf["variation_CA_pct"] = df_store_perf["variation_CA_pct"].astype(float)

cols_show = ["code_magasin", "region_admin", "ca_A", "ca_B", "variation_CA_pct"]

df_best = df_store_perf.dropna(subset=["variation_CA_pct"]).sort_values("variation_CA_pct", ascending=False).head(10)[cols_show]
df_worst = df_store_perf.dropna(subset=["variation_CA_pct"]).sort_values("variation_CA_pct", ascending=True).head(10)[cols_show]

c_top = st.columns(2)
with c_top[0]:
    st.markdown("### ✅ Top 10 meilleurs")
    st.dataframe(df_best, use_container_width=True, hide_index=True)
with c_top[1]:
    st.markdown("### ❌ Top 10 pires")
    st.dataframe(df_worst, use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# EXPORT CSV (tout ce qui est affiché)
# =============================================================================
st.markdown("## 📤 Export (CSV) — données affichées")

# 1) KPI export
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

        {"bloc": "parc", "kpi": "nb_mag_selected", "op": "A", "value": nb_mag_selected},
        {"bloc": "parc", "kpi": "nb_mag_selected", "op": "B", "value": nb_mag_selected},
    ]
)

# 2) Séries export (jour)
df_series_export = pd.concat(
    [
        sA.assign(operation=lib_opA),
        sB.assign(operation=lib_opB),
    ],
    ignore_index=True,
)

# 3) Carte export (région)
df_region_export = df_region.copy()
df_region_export["opA"] = lib_opA
df_region_export["opB"] = lib_opB

# 4) Stores map export (magasin) + perf
df_store_export = df_store_perf.copy()
df_store_export["opA"] = lib_opA
df_store_export["opB"] = lib_opB

c_exp = st.columns(4)
with c_exp[0]:
    st.download_button(
        "⬇️ KPI (CSV)",
        data=_df_to_csv_bytes(df_kpi_export),
        file_name="commerce_kpi.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c_exp[1]:
    st.download_button(
        "⬇️ Séries jour (CSV)",
        data=_df_to_csv_bytes(df_series_export),
        file_name="commerce_series_jour.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c_exp[2]:
    st.download_button(
        "⬇️ Régions (CSV)",
        data=_df_to_csv_bytes(df_region_export),
        file_name="commerce_regions.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c_exp[3]:
    st.download_button(
        "⬇️ Magasins (CSV)",
        data=_df_to_csv_bytes(df_store_export),
        file_name="commerce_magasins.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("Voir les tables exportées"):
    st.markdown("### KPI")
    st.dataframe(df_kpi_export, use_container_width=True, hide_index=True)
    st.markdown("### Séries jour")
    st.dataframe(df_series_export, use_container_width=True, hide_index=True)
    st.markdown("### Régions")
    st.dataframe(df_region_export, use_container_width=True, hide_index=True)
    st.markdown("### Magasins (avec perf)")
    st.dataframe(df_store_export, use_container_width=True, hide_index=True)
