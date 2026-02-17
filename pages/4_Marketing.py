# pages/4_Marketing.py
# -----------------------------------------------------------------------------
# MARKETING — Dashboard (A vs B)
#
# ✅ Objectif : réduire le temps de chargement + spinner unique (comme Commerce)
# - Tous les @st.cache_data : show_spinner=False (évite le “Running …” Streamlit)
# - Spinner custom autour des blocs lourds
#
# ✅ IMPORTANT (FIX) :
# Le CTE généré par filters.py peut déjà contenir un CTE "map".
# Donc dans cette page, on utilise TOUJOURS "map2" pour éviter :
#   WITH query name "map" specified more than once
#
# ✅ FIX A=B (même opération A et B) :
# - labels d’affichage distincts (A)/(B)
#
# ✅ FIX StreamlitDuplicateElementId (plotly_chart) :
# - ajouter un `key=` unique sur CHAQUE st.plotly_chart
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.auth import require_auth
from src.ui import top_bar, tabs_nav
from src.db import read_df
from src.filters import render_filters
from src.components import (
    inject_kpi_compare_css,
    kpi_card_compare,
    fmt_money,
    fmt_int,
)

# =============================================================================
# CONFIG — mapping code_operation -> MV poids OP
# =============================================================================
POIDS_MV_MAP = {
    "poinsettia_2025": "public.mv_poinsettia_2025_poids_op_periode_op_magasin",
    "noel_2025": "public.mv_noel_2025_poids_op_periode_op_magasin",
    "noel_2024": "public.mv_noel_2024_poids_op_periode_op_magasin",
    "anniversaire_2025": "public.mv_anniversaire_2025_poids_op_periode_op_magasin",
    "anniversaire_2024": "public.mv_anniversaire_2024_poids_op_periode_op_magasin",
    "tulipe_2026": "public.mv_tulipe_2026_poids_op_periode_op_magasin",
    "tulipe_2025": "public.mv_tulipe_2025_poids_op_periode_op_magasin",
}

SPINNER_TXT = "Données en cours de chargement… merci de patienter."

# =============================================================================
# Helpers
# =============================================================================
def _safe_float(x) -> float:
    try:
        return float(x or 0)
    except Exception:
        return 0.0


def _as_rate01(v: float) -> float:
    if v is None:
        return 0.0
    v = float(v)
    if v > 1.0:
        return v / 100.0
    return v


def _distinct_labels_for_display(label_A: str, label_B: str):
    """
    Evite les soucis quand A == B (même libellé).
    """
    a = str(label_A or "").strip()
    b = str(label_B or "").strip()
    if a and (a == b):
        return f"{a} (A)", f"{b} (B)"
    return a, b


def _info_no_poids_generic(code_opA: str, code_opB: str):
    st.info(
        "ℹ️ **Poids / analyses produit non disponibles pour cette sélection.**\n\n"
        "Cette comparaison concerne une opération qui n’est pas une opération nationale basée sur un ou plusieurs produits.\n"
        "➡️ Dans ce cas, on ne calcule pas les indicateurs liés aux produits (poids valeur / poids volume, régressions, etc.).\n"
        "Le reste des indicateurs marketing (investissements, parc, répartition packs, etc.) reste disponible."
    )


# =============================================================================
# Régression (numpy polyfit) + hover code magasin + R² en légende
# =============================================================================
def _fit_line_xy(df: pd.DataFrame, x_col: str, y_col: str):
    d = df[[x_col, y_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna()

    if d.shape[0] < 2:
        return None

    x = d[x_col].astype(float).to_numpy()
    y = d[y_col].astype(float).to_numpy()

    if np.allclose(np.std(x), 0):
        return None

    m, b = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 80)
    y_line = m * x_line + b

    y_pred = m * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot

    return m, b, r2, x_line, y_line


def _scatter_plus_line(
    fig: go.Figure,
    df: pd.DataFrame,
    x: str,
    y: str,
    name: str,
    legendgroup: str,
    y_decimals: int,
):
    if df is None or df.empty:
        return

    cols_needed = ["code_magasin", x, y]
    for c in cols_needed:
        if c not in df.columns:
            return

    d = df[cols_needed].copy()
    d = d.replace([np.inf, -np.inf], np.nan)

    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])

    if d.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=d[x],
            y=d[y],
            mode="markers",
            name=f"{name} (points)",
            legendgroup=legendgroup,
            customdata=d[["code_magasin"]].to_numpy(),
            hovertemplate=(
                "<b>Opération :</b> " + name + "<br>"
                "<b>Code magasin :</b> %{customdata[0]}<br>"
                "<b>Investissement :</b> %{x:,.0f} €<br>"
                f"<b>Poids OP :</b> %{{y:.{y_decimals}f}}"
                "<extra></extra>"
            ),
        )
    )

    fit = _fit_line_xy(d, x, y)
    if fit is None:
        return

    m, b, r2, x_line, y_line = fit
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"{name} (fit) — R²={r2:.2f}",
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )


def _plot_reg(
    title: str,
    x_label: str,
    y_col: str,
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    label_A: str,
    label_B: str,
    key: str,  # ✅ clé unique Streamlit
):
    fig = go.Figure()
    y_decimals = 4 if y_col in ("poids_ca", "poids_volume") else 2

    _scatter_plus_line(fig, dfA, "invest_eur", y_col, name=label_A, legendgroup="A", y_decimals=y_decimals)
    _scatter_plus_line(fig, dfB, "invest_eur", y_col, name=label_B, legendgroup="B", y_decimals=y_decimals)

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Poids OP",
        legend_title="Opération",
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# =============================================================================
# Loader — “Commerce light” (CA / tickets / panier moyen) sur la période
# ✅ map2 (pas map) pour éviter collision avec filters.py
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_commerce_light(mags_cte_sql_: str, mags_cte_params_: tuple, date_debut: str, date_fin: str) -> dict:
    sql = f"""
{mags_cte_sql_},

map2 as (
  select
    upper(trim(ancien_code::text)) as ancien_code,
    upper(trim(code_magasin::text)) as code_magasin
  from public.vw_param_magasin_ancien_code
),

base as (
  select
    coalesce(m2.code_magasin, upper(trim(st.code_magasin::text))) as code_magasin,
    st.ticket_date,
    coalesce(st.nb_tickets, 0)::numeric as nb_tickets,
    coalesce(st.total_ttc_net, 0)::numeric as ca_ttc_net
  from public.vw_gold_tickets_jour_clean_op st
  left join map2 m2
    on m2.ancien_code = upper(trim(st.code_magasin::text))
  join mags m
    on m.code_magasin = coalesce(m2.code_magasin, upper(trim(st.code_magasin::text)))
  where st.ticket_date >= %s::date
    and st.ticket_date <= %s::date
)

select
  round(coalesce(sum(ca_ttc_net),0), 2) as ca_total,
  round(coalesce(sum(nb_tickets),0), 0) as tickets_total,
  round(coalesce(sum(ca_ttc_net),0) / nullif(coalesce(sum(nb_tickets),0),0), 2) as panier_moyen
from base;
"""
    df = read_df(sql, tuple(list(mags_cte_params_) + [date_debut, date_fin]))
    if df.empty:
        return {"ca_total": 0.0, "tickets_total": 0.0, "panier_moyen": 0.0}

    r = df.iloc[0].to_dict()
    return {
        "ca_total": _safe_float(r.get("ca_total")),
        "tickets_total": _safe_float(r.get("tickets_total")),
        "panier_moyen": _safe_float(r.get("panier_moyen")),
    }


# =============================================================================
# Loader — métriques marketing (A ou B) sur le parc filtré
# ✅ map2 (pas map)
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_marketing_metrics(mags_cte_sql_: str, mags_cte_params_: tuple, code_operation: str) -> dict:
    sql = f"""
{mags_cte_sql_},

map2 as (
  select
    upper(trim(ancien_code::text)) as ancien_code,
    upper(trim(code_magasin::text)) as code_magasin
  from public.vw_param_magasin_ancien_code
),

parc as (
  select count(*)::numeric as parc_magasin
  from mags
),

ermes as (
  select
    coalesce(m2.code_magasin, upper(trim(e.code_magasin::text))) as code_magasin,
    coalesce(e.pack, 0)::numeric as pack_eur
  from public.ori_ermes e
  left join map2 m2
    on m2.ancien_code = upper(trim(e.code_magasin::text))
  where e.code_operation = %s
    and e.code_magasin is not null
),
ermes_mags as (
  select e.* from ermes e join mags m using(code_magasin)
),

ktb_cost as (
  select
    coalesce(m2.code_magasin, upper(trim(k.code_magasin::text))) as code_magasin,
    coalesce(k.nb_sms_envoyes, 0)::numeric as nb_sms,
    coalesce(k.cout_sms_total_eur, 0)::numeric as cout_sms_total_eur,
    coalesce(k.tx_actifs, 0)::numeric as tx_actifs
  from public.vw_ktb_with_sms_cost k
  left join map2 m2
    on m2.ancien_code = upper(trim(k.code_magasin::text))
  where k.code_operation = %s
    and k.code_magasin is not null
),
ktb_cost_mags as (
  select k.* from ktb_cost k join mags m using(code_magasin)
),

ktb_ori as (
  select
    coalesce(m2.code_magasin, upper(trim(o.code_magasin::text))) as code_magasin,
    coalesce(o.ca_actifs, 0)::numeric as ca_actifs
  from public.ori_ktb o
  left join map2 m2
    on m2.ancien_code = upper(trim(o.code_magasin::text))
  where o.code_operation = %s
    and o.code_magasin is not null
),
ktb_ori_mags as (
  select o.* from ktb_ori o join mags m using(code_magasin)
),

agg as (
  select
    (select parc_magasin from parc) as parc_magasin,

    (select count(distinct code_magasin) from ermes_mags) as nb_mag_ermes,
    (select count(distinct code_magasin) from ktb_cost_mags) as nb_mag_ktb,

    (select coalesce(sum(pack_eur), 0) from ermes_mags) as invest_ermes_total_eur,
    (select coalesce(sum(pack_eur), 0) / nullif(count(distinct code_magasin), 0) from ermes_mags) as invest_ermes_moy_eur,

    (select coalesce(sum(nb_sms), 0) from ktb_cost_mags) as nb_sms_total,
    (select coalesce(sum(cout_sms_total_eur), 0) from ktb_cost_mags) as invest_ktb_total_eur,
    (select coalesce(sum(cout_sms_total_eur), 0) / nullif(count(distinct code_magasin), 0) from ktb_cost_mags) as invest_ktb_moy_eur,

    (select
        coalesce(avg(
            case
              when tx_actifs is null then 0
              when tx_actifs > 1 then tx_actifs / 100.0
              else tx_actifs
            end
        ), 0)
      from ktb_cost_mags
    ) as tx_activation_moyen,

    (select coalesce(sum(ca_actifs), 0) from ktb_ori_mags) as ca_actifs_total,

    (select count(*) from (
        select code_magasin from ermes_mags
        union
        select code_magasin from ktb_cost_mags
    ) u) as nb_mag_marketing
)

select
  parc_magasin,

  nb_mag_marketing,
  round(100 * nb_mag_marketing::numeric / nullif(parc_magasin, 0), 2) as pct_mag_marketing,

  nb_mag_ermes,
  round(invest_ermes_total_eur, 2) as invest_ermes_total_eur,
  round(coalesce(invest_ermes_moy_eur, 0), 2) as invest_ermes_moy_eur,
  round(100 * nb_mag_ermes::numeric / nullif(parc_magasin, 0), 2) as pct_mag_ermes,

  nb_mag_ktb,
  round(nb_sms_total, 0) as nb_sms_total,
  round(invest_ktb_total_eur, 2) as invest_ktb_total_eur,
  round(coalesce(invest_ktb_moy_eur, 0), 2) as invest_ktb_moy_eur,
  round(100 * nb_mag_ktb::numeric / nullif(parc_magasin, 0), 2) as pct_mag_ktb,

  round(coalesce(invest_ermes_total_eur,0) + coalesce(invest_ktb_total_eur,0), 2) as invest_total_eur,
  round(
    (coalesce(invest_ermes_total_eur,0) + coalesce(invest_ktb_total_eur,0))
    / nullif(nb_mag_marketing, 0),
    2
  ) as invest_total_moy_eur,

  round(coalesce(tx_activation_moyen, 0), 6) as tx_activation_moyen,
  round(coalesce(ca_actifs_total, 0), 2) as ca_actifs_total
from agg;
"""
    df = read_df(sql, tuple(list(mags_cte_params_) + [code_operation, code_operation, code_operation]))

    keys = [
        "parc_magasin",
        "nb_mag_marketing",
        "pct_mag_marketing",
        "nb_mag_ermes",
        "invest_ermes_total_eur",
        "invest_ermes_moy_eur",
        "pct_mag_ermes",
        "nb_mag_ktb",
        "nb_sms_total",
        "invest_ktb_total_eur",
        "invest_ktb_moy_eur",
        "pct_mag_ktb",
        "invest_total_eur",
        "invest_total_moy_eur",
        "tx_activation_moyen",
        "ca_actifs_total",
    ]
    if df.empty:
        return {k: 0.0 for k in keys}

    r = df.iloc[0].to_dict()
    return {k: _safe_float(r.get(k)) for k in keys}


# =============================================================================
# Région — tableaux "beaux"
# =============================================================================
def _prettify_region_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    rename = {
        "region": "Région",
        "nb_magasin": "Nombre de magasins",
        "invest_total_eur": "Investissement total (€)",
        "invest_ermes_eur": "Investissement ERMES (€)",
        "invest_fid_eur": "Investissement FID (€)",
        "invest_moy_total_magasin": "Moyenne total / magasin (€)",
        "invest_moy_ermes_magasin": "Moyenne ERMES / magasin (€)",
        "invest_moy_fid_magasin": "Moyenne FID / magasin (€)",
    }
    out = out.rename(columns=rename)

    int_cols = ["Nombre de magasins"]
    money_cols = [
        "Investissement total (€)",
        "Investissement ERMES (€)",
        "Investissement FID (€)",
        "Moyenne total / magasin (€)",
        "Moyenne ERMES / magasin (€)",
        "Moyenne FID / magasin (€)",
    ]

    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: fmt_int(x))

    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: fmt_money(x, 0))

    return out


def _render_pretty_table(df: pd.DataFrame):
    st.dataframe(df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=600, show_spinner=False)
def _load_region_table_df(
    region_field_sql: str,
    mags_cte_sql_: str,
    mags_cte_params_: tuple,
    code_operation: str,
) -> pd.DataFrame:
    sql_region = f"""
{mags_cte_sql_},

map2 as (
  select
    upper(trim(ancien_code::text)) as ancien_code,
    upper(trim(code_magasin::text)) as code_magasin
  from public.vw_param_magasin_ancien_code
),

ermes as (
  select
    coalesce(m2.code_magasin, upper(trim(e.code_magasin::text))) as code_magasin,
    coalesce(e.pack, 0)::numeric as invest_ermes_eur
  from public.ori_ermes e
  left join map2 m2
    on m2.ancien_code = upper(trim(e.code_magasin::text))
  where e.code_operation = %s
    and e.code_magasin is not null
),

ktb as (
  select
    coalesce(m2.code_magasin, upper(trim(k.code_magasin::text))) as code_magasin,
    coalesce(k.cout_sms_total_eur, 0)::numeric as invest_fid_eur
  from public.vw_ktb_with_sms_cost k
  left join map2 m2
    on m2.ancien_code = upper(trim(k.code_magasin::text))
  where k.code_operation = %s
    and k.code_magasin is not null
),

base as (
  select
    m.code_magasin,
    coalesce({region_field_sql}, 'Non renseigné') as region,
    coalesce(e.invest_ermes_eur, 0) as invest_ermes_eur,
    coalesce(k.invest_fid_eur, 0) as invest_fid_eur,
    coalesce(e.invest_ermes_eur, 0) + coalesce(k.invest_fid_eur, 0) as invest_total_eur
  from mags m
  left join ermes e using(code_magasin)
  left join ktb   k using(code_magasin)
  left join public.ref_magasin rm
    on upper(trim(rm.code_magasin::text)) = m.code_magasin
)

select
  region,
  count(*) as nb_magasin,
  round(sum(invest_total_eur), 2) as invest_total_eur,
  round(sum(invest_ermes_eur), 2) as invest_ermes_eur,
  round(sum(invest_fid_eur), 2) as invest_fid_eur,
  round(avg(invest_total_eur), 2) as invest_moy_total_magasin,
  round(avg(invest_ermes_eur), 2) as invest_moy_ermes_magasin,
  round(avg(invest_fid_eur), 2) as invest_moy_fid_magasin
from base
group by region
order by invest_total_eur desc;
"""
    return read_df(sql_region, tuple(list(mags_cte_params_) + [code_operation, code_operation]))


def _render_region_table(
    region_field_sql: str,
    title: str,
    mags_cte_sql_: str,
    mags_cte_params_: tuple,
    code_operation: str,
):
    df = _load_region_table_df(region_field_sql, mags_cte_sql_, mags_cte_params_, code_operation)
    st.markdown(f"### {title}")
    _render_pretty_table(_prettify_region_df(df))


# =============================================================================
# Packs ERMES — distribution nb magasins
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_pack_distribution(mags_cte_sql_: str, mags_cte_params_: tuple, code_operation: str) -> pd.DataFrame:
    sql = f"""
{mags_cte_sql_},

map2 as (
  select
    upper(trim(ancien_code::text)) as ancien_code,
    upper(trim(code_magasin::text)) as code_magasin
  from public.vw_param_magasin_ancien_code
),

ermes as (
  select
    coalesce(m2.code_magasin, upper(trim(e.code_magasin::text))) as code_magasin,
    round(coalesce(e.pack, 0)::numeric, 2) as pack_eur
  from public.ori_ermes e
  left join map2 m2
    on m2.ancien_code = upper(trim(e.code_magasin::text))
  where e.code_operation = %s
    and e.code_magasin is not null
),
ermes_mags as (
  select e.*
  from ermes e
  join mags m using(code_magasin)
)
select
  pack_eur,
  count(*) as nb_magasin
from ermes_mags
group by pack_eur
order by pack_eur;
"""
    return read_df(sql, tuple(list(mags_cte_params_) + [code_operation]))


# =============================================================================
# Loader — données régression (A ou B)
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def _load_reg_data(
    code_operation: str,
    poids_mv: str,
    invest_mode: str,
    mags_cte_sql_: str,
    mags_cte_params_: tuple,
) -> pd.DataFrame:
    map2_cte = """
map2 as (
  select
    upper(trim(ancien_code::text)) as ancien_code,
    upper(trim(code_magasin::text)) as code_magasin
  from public.vw_param_magasin_ancien_code
)
""".strip()

    if invest_mode == "ermes":
        invest_cte = """
invest as (
  select
    coalesce(m2.code_magasin, upper(trim(e.code_magasin::text))) as code_magasin,
    coalesce(e.pack, 0)::numeric as invest_eur
  from public.ori_ermes e
  left join map2 m2
    on m2.ancien_code = upper(trim(e.code_magasin::text))
  where e.code_operation = %s
    and e.code_magasin is not null
)
""".strip()
        params_extra = [code_operation]

    elif invest_mode == "fid":
        invest_cte = """
invest as (
  select
    coalesce(m2.code_magasin, upper(trim(k.code_magasin::text))) as code_magasin,
    coalesce(k.cout_sms_total_eur, 0)::numeric as invest_eur
  from public.vw_ktb_with_sms_cost k
  left join map2 m2
    on m2.ancien_code = upper(trim(k.code_magasin::text))
  where k.code_operation = %s
    and k.code_magasin is not null
)
""".strip()
        params_extra = [code_operation]

    else:
        invest_cte = """
ermes as (
  select
    coalesce(m2.code_magasin, upper(trim(e.code_magasin::text))) as code_magasin,
    coalesce(e.pack, 0)::numeric as invest_ermes_eur
  from public.ori_ermes e
  left join map2 m2
    on m2.ancien_code = upper(trim(e.code_magasin::text))
  where e.code_operation = %s
    and e.code_magasin is not null
),
fid as (
  select
    coalesce(m2.code_magasin, upper(trim(k.code_magasin::text))) as code_magasin,
    coalesce(k.cout_sms_total_eur, 0)::numeric as invest_fid_eur
  from public.vw_ktb_with_sms_cost k
  left join map2 m2
    on m2.ancien_code = upper(trim(k.code_magasin::text))
  where k.code_operation = %s
    and k.code_magasin is not null
),
invest as (
  select
    m.code_magasin,
    (coalesce(e.invest_ermes_eur, 0) + coalesce(f.invest_fid_eur, 0))::numeric as invest_eur
  from mags m
  left join ermes e using(code_magasin)
  left join fid   f using(code_magasin)
)
""".strip()
        params_extra = [code_operation, code_operation]

    sql = f"""
{mags_cte_sql_},
{map2_cte},
{invest_cte},

poids as (
  select
    coalesce(m2.code_magasin, upper(trim(p0.code_magasin::text))) as code_magasin,
    coalesce(p0.poids_ca, 0)::numeric as poids_ca,
    coalesce(p0.poids_volume, 0)::numeric as poids_volume
  from {poids_mv} p0
  left join map2 m2
    on m2.ancien_code = upper(trim(p0.code_magasin::text))
)

select
  m.code_magasin,
  coalesce(i.invest_eur, 0)::numeric as invest_eur,
  coalesce(p.poids_ca, 0)::numeric as poids_ca,
  coalesce(p.poids_volume, 0)::numeric as poids_volume
from mags m
left join invest i using(code_magasin)
left join poids  p using(code_magasin)
where p.code_magasin is not null;
"""
    params = tuple(list(mags_cte_params_) + params_extra)
    df = read_df(sql, params)

    if df is None or df.empty:
        return pd.DataFrame(columns=["code_magasin", "invest_eur", "poids_ca", "poids_volume"])

    df["invest_eur"] = pd.to_numeric(df["invest_eur"], errors="coerce").fillna(0.0)
    df["poids_ca"] = pd.to_numeric(df["poids_ca"], errors="coerce").fillna(0.0)
    df["poids_volume"] = pd.to_numeric(df["poids_volume"], errors="coerce").fillna(0.0)
    df["code_magasin"] = df["code_magasin"].astype(str)
    return df


# =============================================================================
# PAGE — UI
# =============================================================================
st.set_page_config(page_title="Marketing", layout="wide")
require_auth()

top_bar("Dashboard – Marketing")
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

mags_cte_sql_A = ctx["mags_cte_sql_A"]
mags_cte_params_A = ctx["mags_cte_params_A"]
mags_cte_sql_B = ctx["mags_cte_sql_B"]
mags_cte_params_B = ctx["mags_cte_params_B"]

# ✅ labels display distincts si A == B
label_A, label_B = _distinct_labels_for_display(lib_opA, lib_opB)

# =============================================================================
# KPI COMMERCE (au-dessus)
# =============================================================================
st.markdown("## 🧾 Contexte commerce (sur la période) — A vs B")
with st.spinner(SPINNER_TXT):
    cA = load_commerce_light(mags_cte_sql_A, mags_cte_params_A, dateA0, dateA1)
    cB = load_commerce_light(mags_cte_sql_B, mags_cte_params_B, dateB0, dateB1)

kc = st.columns(3)
with kc[0]:
    kpi_card_compare("CA total", cA["ca_total"], cB["ca_total"], label_A, label_B, formatter=lambda x: fmt_money(x, 0))
with kc[1]:
    kpi_card_compare("Tickets total", cA["tickets_total"], cB["tickets_total"], label_A, label_B, formatter=lambda x: fmt_int(x))
with kc[2]:
    kpi_card_compare("Panier moyen", cA["panier_moyen"], cB["panier_moyen"], label_A, label_B, formatter=lambda x: fmt_money(x, 2))

st.divider()

# =============================================================================
# KPI MARKETING
# =============================================================================
st.markdown("## 📣 Marketing — Investissement (ERMES + FID)")
st.divider()

with st.spinner(SPINNER_TXT):
    mA = load_marketing_metrics(mags_cte_sql_A, mags_cte_params_A, code_opA)
    mB = load_marketing_metrics(mags_cte_sql_B, mags_cte_params_B, code_opB)

# --- TOTAL
rt1, rt2, rt3 = st.columns(3)
with rt1:
    kpi_card_compare("TOTAL – Investissement (ERMES + FID)", mA["invest_total_eur"], mB["invest_total_eur"], label_A, label_B, formatter=lambda x: fmt_money(x, 0))
with rt2:
    kpi_card_compare("TOTAL – Moyen / magasin (marketing)", mA["invest_total_moy_eur"], mB["invest_total_moy_eur"], label_A, label_B, formatter=lambda x: fmt_money(x, 2))
with rt3:
    kpi_card_compare("CA actifs (FID) — ori_ktb.ca_actifs", mA["ca_actifs_total"], mB["ca_actifs_total"], label_A, label_B, formatter=lambda x: fmt_money(x, 0))

# --- ERMES
re1, re2, re3 = st.columns(3)
with re1:
    kpi_card_compare("ERMES – Nb magasins pack", mA["nb_mag_ermes"], mB["nb_mag_ermes"], label_A, label_B, formatter=lambda x: fmt_int(x))
with re2:
    kpi_card_compare("ERMES – Investissement PACK (Total)", mA["invest_ermes_total_eur"], mB["invest_ermes_total_eur"], label_A, label_B, formatter=lambda x: fmt_money(x, 0))
with re3:
    kpi_card_compare("ERMES – % magasins (sur parc)", mA["pct_mag_ermes"], mB["pct_mag_ermes"], label_A, label_B, formatter=lambda x: f"{float(x or 0):.1f} %")

# --- FID / KTB
rf1, rf2, rf3, rf4 = st.columns(4)
with rf1:
    kpi_card_compare("KTB – Nb magasins", mA["nb_mag_ktb"], mB["nb_mag_ktb"], label_A, label_B, formatter=lambda x: fmt_int(x))
with rf2:
    kpi_card_compare("KTB – Investissement SMS (Total)", mA["invest_ktb_total_eur"], mB["invest_ktb_total_eur"], label_A, label_B, formatter=lambda x: fmt_money(x, 0))
with rf3:
    kpi_card_compare("KTB – Nb SMS envoyés", mA["nb_sms_total"], mB["nb_sms_total"], label_A, label_B, formatter=lambda x: fmt_int(x))
with rf4:
    kpi_card_compare(
        "Taux d’activation fidèles (KTB)",
        _as_rate01(mA["tx_activation_moyen"]) * 100.0,
        _as_rate01(mB["tx_activation_moyen"]) * 100.0,
        label_A,
        label_B,
        formatter=lambda x: f"{float(x or 0):.1f} %",
    )

# --- Parc & marketing
rp1, rp2 = st.columns(2)
with rp1:
    kpi_card_compare("Parc magasins", mA["parc_magasin"], mB["parc_magasin"], label_A, label_B, formatter=lambda x: fmt_int(x))
with rp2:
    kpi_card_compare("% magasins marketing (ERMES ou KTB)", mA["pct_mag_marketing"], mB["pct_mag_marketing"], label_A, label_B, formatter=lambda x: f"{float(x or 0):.1f} %")

st.divider()

# =============================================================================
# TABLEAUX PAR REGION (Op A)
# =============================================================================
st.markdown("## 🌍 Investissement marketing par région (Opération A)")
with st.spinner(SPINNER_TXT):
    _render_region_table('rm."crp_:_region_elargie"', "Par région élargie", mags_cte_sql_A, mags_cte_params_A, code_opA)
    _render_region_table('rm."crp_:_region_nationale_d_affectation"', "Par région administrative", mags_cte_sql_A, mags_cte_params_A, code_opA)

st.divider()

# =============================================================================
# PACK ERMES : nb magasins A vs B
# =============================================================================
st.markdown("## 🍰 Répartition des packs — nb magasins (A vs B)")
with st.spinner(SPINNER_TXT):
    df_pack_A = load_pack_distribution(mags_cte_sql_A, mags_cte_params_A, code_opA)
    df_pack_B = load_pack_distribution(mags_cte_sql_B, mags_cte_params_B, code_opB)

c1, c2 = st.columns(2)
with c1:
    if df_pack_A.empty:
        st.info(f"Aucune donnée pack ERMES pour {label_A}.")
    else:
        figA = px.pie(df_pack_A, names="pack_eur", values="nb_magasin", hole=0.45, title=f"Répartition des packs — nb magasins ({label_A})")
        figA.update_traces(textinfo="percent+label")
        st.plotly_chart(figA, use_container_width=True, key=f"pie_pack_A_{code_opA}")

with c2:
    if df_pack_B.empty:
        st.info(f"Aucune donnée pack ERMES pour {label_B}.")
    else:
        figB = px.pie(df_pack_B, names="pack_eur", values="nb_magasin", hole=0.45, title=f"Répartition des packs — nb magasins ({label_B})")
        figB.update_traces(textinfo="percent+label")
        st.plotly_chart(figB, use_container_width=True, key=f"pie_pack_B_{code_opB}")

st.divider()

# =============================================================================
# REGRESSIONS — Poids OP vs investissement (A vs B)
# =============================================================================
st.markdown("## 📈 Régressions linéaires — Poids OP vs investissement (A vs B)")

poids_mv_A = POIDS_MV_MAP.get(code_opA)
poids_mv_B = POIDS_MV_MAP.get(code_opB)

# ✅ message générique + skip (pas d’arrêt de page)
if not poids_mv_A or not poids_mv_B:
    _info_no_poids_generic(code_opA, code_opB)
else:
    with st.spinner(SPINNER_TXT):
        dfA_ermes = _load_reg_data(code_opA, poids_mv_A, "ermes", mags_cte_sql_A, mags_cte_params_A)
        dfB_ermes = _load_reg_data(code_opB, poids_mv_B, "ermes", mags_cte_sql_B, mags_cte_params_B)

        dfA_fid = _load_reg_data(code_opA, poids_mv_A, "fid", mags_cte_sql_A, mags_cte_params_A)
        dfB_fid = _load_reg_data(code_opB, poids_mv_B, "fid", mags_cte_sql_B, mags_cte_params_B)

        dfA_total = _load_reg_data(code_opA, poids_mv_A, "total", mags_cte_sql_A, mags_cte_params_A)
        dfB_total = _load_reg_data(code_opB, poids_mv_B, "total", mags_cte_sql_B, mags_cte_params_B)

    st.markdown("### Poids OP en VALEUR (CA)")
    _plot_reg(
        "ERMES : Investissement vs Poids OP (CA) — A vs B",
        "Investissement ERMES (€)",
        "poids_ca",
        dfA_ermes,
        dfB_ermes,
        label_A,
        label_B,
        key=f"reg_ca_ermes_{code_opA}_{code_opB}",
    )
    _plot_reg(
        "FID : Investissement vs Poids OP (CA) — A vs B",
        "Investissement FID (€)",
        "poids_ca",
        dfA_fid,
        dfB_fid,
        label_A,
        label_B,
        key=f"reg_ca_fid_{code_opA}_{code_opB}",
    )
    _plot_reg(
        "TOTAL : Investissement vs Poids OP (CA) — A vs B",
        "Investissement total (€) = ERMES + FID",
        "poids_ca",
        dfA_total,
        dfB_total,
        label_A,
        label_B,
        key=f"reg_ca_total_{code_opA}_{code_opB}",
    )

    st.markdown("### Poids OP en VOLUME")
    _plot_reg(
        "ERMES : Investissement vs Poids OP (Volume) — A vs B",
        "Investissement ERMES (€)",
        "poids_volume",
        dfA_ermes,
        dfB_ermes,
        label_A,
        label_B,
        key=f"reg_vol_ermes_{code_opA}_{code_opB}",
    )
    _plot_reg(
        "FID : Investissement vs Poids OP (Volume) — A vs B",
        "Investissement FID (€)",
        "poids_volume",
        dfA_fid,
        dfB_fid,
        label_A,
        label_B,
        key=f"reg_vol_fid_{code_opA}_{code_opB}",
    )
    _plot_reg(
        "TOTAL : Investissement vs Poids OP (Volume) — A vs B",
        "Investissement total (€) = ERMES + FID",
        "poids_volume",
        dfA_total,
        dfB_total,
        label_A,
        label_B,
        key=f"reg_vol_total_{code_opA}_{code_opB}",
    )
