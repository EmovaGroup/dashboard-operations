# pages/3_Achats.py
# =============================================================================
# PAGE : ACHATS (A vs B) — parcs séparés A/B + normalisation ancien_code -> code_magasin
# =============================================================================
# Objectif : même logique que Commerce/Marketing
# ✅ On se fit AU PARC VENTES (tickets) via `mags` (déjà filtré comparable/parc/ermes/fid/etc.)
# ✅ Puis on affiche la data Achats UNIQUEMENT sur ce parc
# ❌ On n'ajoute PLUS les magasins acheteurs hors parc ventes
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from src.auth import require_auth
from src.ui import top_bar, tabs_nav
from src.db import read_df
from src.filters import render_filters
from src.components import (
    inject_kpi_css,
    inject_kpi_compare_css,
    kpi_card_compare,
    fmt_money,
    fmt_int,
    inject_store_css,
    store_card_3col_html,
)

SPINNER_TXT = "Données en cours de chargement… merci de patienter."

# -----------------------------------------------------------------------------
# Config page + auth
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Achats", layout="wide")
require_auth()

top_bar("Dashboard – Achats")
tabs_nav()
st.divider()

inject_kpi_css()
inject_kpi_compare_css()
inject_store_css()

# -----------------------------------------------------------------------------
# Contexte filtres
# -----------------------------------------------------------------------------
ctx = render_filters()

code_opA = ctx["opA"]["code"]
lib_opA = ctx["opA"]["lib"]
dateA0 = str(ctx["opA"]["date_debut"])
dateA1 = str(ctx["opA"]["date_fin"])

code_opB = ctx["opB"]["code"]
lib_opB = ctx["opB"]["lib"]
dateB0 = str(ctx["opB"]["date_debut"])
dateB1 = str(ctx["opB"]["date_fin"])

# ✅ IMPORTANT : parcs séparés A / B
mags_cte_sql_A = ctx["mags_cte_sql_A"]
mags_cte_params_A = ctx["mags_cte_params_A"]

mags_cte_sql_B = ctx["mags_cte_sql_B"]
mags_cte_params_B = ctx["mags_cte_params_B"]

code_magasin_selected = ctx["filters"]["code_magasin"]

st.caption(f"Opération A : **{lib_opA}**  |  Opération B : **{lib_opB}**")

# -----------------------------------------------------------------------------
# Source achats normalisée
# -----------------------------------------------------------------------------
ACHATS_SRC = "public.vw_op_achats_norm"  # contient fournisseur_norm

# -----------------------------------------------------------------------------
# Mapping ancien code -> nouveau code (canon)
# -----------------------------------------------------------------------------
CODE_MAP_VIEW = "public.vw_param_magasin_ancien_code"


# =============================================================================
# Helpers Python (safe casts)
# =============================================================================
def _f0(x) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0


# =============================================================================
# Palette couleurs (UX-friendly)
# =============================================================================
BASE_PALETTE = [
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]
COLOR_AUTRES_NON_CAPTES = "#6B7280"


def build_color_map(labels: list[str]) -> dict:
    cmap = {}
    i = 0
    for lab in labels:
        if lab.strip().lower() == "autres fournisseurs (non captés)":
            cmap[lab] = COLOR_AUTRES_NON_CAPTES
        else:
            cmap[lab] = BASE_PALETTE[i % len(BASE_PALETTE)]
            i += 1
    return cmap


# =============================================================================
# COMMERCE LIGHT — CA / Tickets / PM (sur la période)
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_commerce_light(mags_cte_sql_: str, mags_cte_params_: tuple, date_debut: str, date_fin: str) -> dict:
    sql = f"""
{mags_cte_sql_},

base as (
  select
    trim(st.code_magasin::text) as code_magasin,
    st.ticket_date,
    coalesce(st.nb_tickets, 0)::numeric as nb_tickets,
    coalesce(st.total_ttc_net, 0)::numeric as ca_ttc_net
  from public.vw_gold_tickets_jour_clean_op st
  join mags m on m.code_magasin = trim(st.code_magasin::text)
  where st.ticket_date >= %s::date
    and st.ticket_date <= %s::date
)
select
  round(coalesce(sum(ca_ttc_net),0), 2) as ca_total,
  round(coalesce(sum(nb_tickets),0), 0) as tickets_total,
  round(coalesce(sum(ca_ttc_net),0) / nullif(coalesce(sum(nb_tickets),0),0), 2) as panier_moyen
from base;
"""
    df = read_df(sql, params=tuple(list(mags_cte_params_) + [date_debut, date_fin]))
    if df.empty:
        return {"ca_total": 0.0, "tickets_total": 0.0, "panier_moyen": 0.0}

    r = df.iloc[0].to_dict()
    return {
        "ca_total": _f0(r.get("ca_total")),
        "tickets_total": _f0(r.get("tickets_total")),
        "panier_moyen": _f0(r.get("panier_moyen")),
    }


# =============================================================================
# FICHE MAGASIN
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def magasin_info_query(code_magasin: str) -> pd.Series:
    sql = """
    select
      trim(rm.code_magasin::text) as code_magasin,
      rm.nom_magasin,
      rm.telephone,
      rm.e_mail,
      rm."adresses_:_adresse_de_livraison_:_adresse_1" as adresse,
      rm."adresses_:_adresse_de_livraison_:_cp" as cp,
      rm."adresses_:_adresse_de_livraison_:_ville" as ville,
      rm."adresses_:_adresse_de_livraison_:_pays" as pays,
      rm.type,
      rm.statut,
      rm.rcr,
      rm.rdr,
      rm.nom_franchise,
      rm.prenom_franchise,
      rm.telephone_franchise
    from public.ref_magasin rm
    where trim(rm.code_magasin::text) = %s
    limit 1;
    """
    df = read_df(sql, params=(code_magasin,))
    if df.empty:
        return pd.Series({})
    return df.iloc[0]


def render_magasin_fiche(code_magasin: str):
    info = magasin_info_query(code_magasin)
    if info.empty:
        st.warning("Magasin introuvable dans ref_magasin.")
        return

    title = f"{info.get('code_magasin','')} — {info.get('nom_magasin','')}"
    tel = info.get("telephone")
    mail = info.get("e_mail")

    subtitle_lines = []
    if tel:
        subtitle_lines.append(f"📞 {tel}")
    if mail:
        subtitle_lines.append(f"✉️ {mail}")
    subtitle = "  •  ".join(subtitle_lines)

    badges = []
    statut = (info.get("statut") or "").strip()
    if statut:
        cls = "badge-ok" if "actif" in statut.lower() or "ouvert" in statut.lower() else "badge-warn"
        badges.append((statut, cls))

    col_left = [
        ("Type :", info.get("type") or "—"),
        ("RCR :", info.get("rcr") or "—"),
        ("RDR :", info.get("rdr") or "—"),
    ]

    adr = info.get("adresse") or ""
    cp = info.get("cp") or ""
    ville = info.get("ville") or ""
    pays = info.get("pays") or ""
    col_mid = [
        ("Adresse :", adr or "—"),
        ("CP / Ville :", f"{cp} {ville}".strip() or "—"),
        ("Pays :", pays or "—"),
    ]

    nom_f = info.get("nom_franchise")
    prenom_f = info.get("prenom_franchise")
    tel_f = info.get("telephone_franchise")

    col_right = []
    if nom_f or prenom_f:
        col_right.append(("Franchisé :", f"{prenom_f or ''} {nom_f or ''}".strip()))
    if tel_f:
        col_right.append(("Tél. franchisé :", tel_f))

    store_card_3col_html(
        title=title,
        subtitle=subtitle,
        badges=badges,
        col_left=col_left,
        col_mid=col_mid,
        col_right=col_right,
        left_title="Infos magasin",
        mid_title="Adresse",
        right_title="Franchise",
    )


# =============================================================================
# Helper SQL — normalisation code magasin achats (canon)
# =============================================================================
def _achats_norm_cte(code_op: str) -> tuple[str, list]:
    cte = f"""
a_norm as (
  select
    coalesce(vp.code_magasin, upper(trim(a.code_magasin::text))) as code_magasin_canon,
    a.*
  from {ACHATS_SRC} a
  left join {CODE_MAP_VIEW} vp
    on upper(trim(a.code_magasin::text)) = vp.ancien_code
  where a.code_operation = %s
    and a.code_magasin is not null
)
""".strip()
    return cte, [code_op]


# =============================================================================
# PARC VENTES STRICT (comme Commerce/Marketing)
# =============================================================================
def _parc_sales_cte(mags_cte_sql_: str, mags_cte_params_: tuple) -> tuple[str, tuple]:
    sql = f"""
{mags_cte_sql_},

parc_sales as (
  select distinct code_magasin
  from mags
  where code_magasin is not null
)
"""
    return sql, mags_cte_params_


# =============================================================================
# NO DATA — bloc "stop"
# =============================================================================
def no_data_block(msg: str):
    st.warning(
        f"📭 **Je n’ai pas de données pour cette période / ces filtres.**\n\n"
        f"{msg}\n\n"
        f"👉 Merci de choisir une autre opération (ou d’élargir les filtres)."
    )
    st.stop()


# =============================================================================
# NO DATA — message générique Achats
# =============================================================================
def _info_no_achats_generic():
    st.warning(
        "🧾 **Aucune data Achats disponible pour cette opération.**\n\n"
        "Cela signifie qu’il n’y a pas de lignes d’achats référencées pour l’opération sélectionnée."
    )


# =============================================================================
# Check data Achats — magasin & parc ventes
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def has_data_for_magasin(code_op: str, code_magasin: str) -> bool:
    sql_cte, _ = _achats_norm_cte(code_op)
    sql = f"""
with
{sql_cte}
select 1
from a_norm a
where a.code_magasin_canon = upper(trim(%s::text))
limit 1;
"""
    df = read_df(sql, params=(code_op, code_magasin))
    return not df.empty


@st.cache_data(ttl=600, show_spinner=False)
def has_sales_parc(mags_cte_sql_: str, mags_cte_params_: tuple) -> bool:
    parc_sql, parc_params = _parc_sales_cte(mags_cte_sql_, mags_cte_params_)
    sql = f"""
{parc_sql}
select 1
from parc_sales
limit 1;
"""
    df = read_df(sql, params=parc_params)
    return not df.empty


@st.cache_data(ttl=600, show_spinner=False)
def has_achats_for_sales_parc(mags_cte_sql_: str, mags_cte_params_: tuple, code_op: str) -> bool:
    parc_sql, parc_params = _parc_sales_cte(mags_cte_sql_, mags_cte_params_)
    achats_cte, achats_params = _achats_norm_cte(code_op)

    sql = f"""
{parc_sql},
{achats_cte}
select 1
from a_norm a
join parc_sales p
  on p.code_magasin = a.code_magasin_canon
limit 1;
"""
    df = read_df(sql, params=tuple(list(parc_params) + achats_params))
    return not df.empty


# =============================================================================
# MODE MAGASIN : KPI + table PUM
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def achats_kpi_magasin(code_op: str, code_magasin: str) -> pd.Series:
    sql_cte, _ = _achats_norm_cte(code_op)
    sql = f"""
with
{sql_cte}
select
  coalesce(sum(a."total achat"),0) as valeur_achats_captee,
  coalesce(sum(a.quantite),0) as volume_achats_capte,
  coalesce(sum(a."total achat") / nullif(sum(a.quantite)::numeric,0), 0) as pum
from a_norm a
where a.code_magasin_canon = upper(trim(%s::text));
"""
    df = read_df(sql, params=(code_op, code_magasin))
    if df.empty:
        return pd.Series({"valeur_achats_captee": 0, "volume_achats_capte": 0, "pum": 0})
    return df.iloc[0]


@st.cache_data(ttl=600, show_spinner=False)
def pum_par_fournisseur_magasin(code_op: str, code_magasin: str) -> pd.DataFrame:
    sql_cte, _ = _achats_norm_cte(code_op)
    sql = f"""
with
{sql_cte}
select
  coalesce(nullif(trim(a.fournisseur_norm::text),''),'Fournisseur inconnu') as fournisseur,
  coalesce(sum(a.quantite),0) as qte,
  coalesce(sum(a."total achat"),0) as valeur,
  coalesce(sum(a."total achat") / nullif(sum(a.quantite)::numeric,0), 0) as pum
from a_norm a
where a.code_magasin_canon = upper(trim(%s::text))
group by 1
order by valeur desc;
"""
    return read_df(sql, params=(code_op, code_magasin))


# =============================================================================
# MODE PARC (VENTES STRICT) : KPI + camemberts + table PUM
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def achats_kpi_parc_sales(mags_cte_sql_: str, mags_cte_params_: tuple, code_op: str) -> pd.Series:
    parc_sql, parc_params = _parc_sales_cte(mags_cte_sql_, mags_cte_params_)
    achats_cte, achats_params = _achats_norm_cte(code_op)

    sql = f"""
{parc_sql},
{achats_cte},

achats_capt as (
  select
    a.code_magasin_canon as code_magasin,
    coalesce(sum(a."total achat"),0) as valeur_achats,
    coalesce(sum(a.quantite),0) as volume_achats
  from a_norm a
  join parc_sales p on p.code_magasin = a.code_magasin_canon
  group by 1
),

parc as (select count(*) as nb_mag_parc from parc_sales),
acheteurs as (select count(*) as nb_mag_acheteurs from achats_capt where valeur_achats > 0),

agg as (
  select
    coalesce(sum(ac.valeur_achats),0) as valeur_achats_captee,
    coalesce(sum(ac.volume_achats),0) as volume_achats_capte,
    (select nb_mag_parc from parc) as nb_mag_parc,
    (select nb_mag_acheteurs from acheteurs) as nb_mag_acheteurs
  from achats_capt ac
)
select
  coalesce(valeur_achats_captee,0) as valeur_achats_captee,
  coalesce(volume_achats_capte,0)  as volume_achats_capte,
  coalesce(nb_mag_parc,0)          as nb_mag_parc,
  coalesce(nb_mag_acheteurs,0)     as nb_mag_acheteurs,
  coalesce(round(nb_mag_acheteurs::numeric / nullif(nb_mag_parc,0) * 100, 1), 0) as pct_mag_acheteurs
from agg;
"""
    df = read_df(sql, params=tuple(list(parc_params) + achats_params))
    if df.empty:
        return pd.Series(
            {
                "valeur_achats_captee": 0,
                "volume_achats_capte": 0,
                "nb_mag_parc": 0,
                "nb_mag_acheteurs": 0,
                "pct_mag_acheteurs": 0,
            }
        )
    return df.iloc[0]


@st.cache_data(ttl=600, show_spinner=False)
def camembert_fusion_magasin_parc_sales(
    mags_cte_sql_: str,
    mags_cte_params_: tuple,
    code_op: str,
    top_n_fournisseurs: int = 12
) -> pd.DataFrame:
    parc_sql, parc_params = _parc_sales_cte(mags_cte_sql_, mags_cte_params_)
    achats_cte, achats_params = _achats_norm_cte(code_op)

    sql = f"""
{parc_sql},
{achats_cte},

achats_mag_fourn as (
  select
    a.code_magasin_canon as code_magasin,
    coalesce(nullif(trim(a.fournisseur_norm::text),''),'Fournisseur inconnu') as fournisseur,
    coalesce(sum(a."total achat"),0) as valeur
  from a_norm a
  join parc_sales p on p.code_magasin = a.code_magasin_canon
  group by 1,2
),

best_fourn as (
  select code_magasin, fournisseur
  from (
    select
      code_magasin,
      fournisseur,
      valeur,
      row_number() over (partition by code_magasin order by valeur desc) as rn
    from achats_mag_fourn
  ) t
  where rn = 1
),

capt_counts as (
  select fournisseur, count(*)::int as nb_mag
  from best_fourn
  group by 1
  order by nb_mag desc
),

capt_top as (
  select fournisseur, nb_mag
  from capt_counts
  order by nb_mag desc
  limit {int(top_n_fournisseurs)}
),

capt_rest as (
  select 'Autres fournisseurs (captés)'::text as fournisseur, coalesce(sum(nb_mag),0)::int as nb_mag
  from capt_counts
  where fournisseur not in (select fournisseur from capt_top)
),

parc_sans_achats as (
  select p.code_magasin
  from parc_sales p
  left join best_fourn bf on bf.code_magasin = p.code_magasin
  where bf.code_magasin is null
),

autres_non_captes as (
  select 'Autres fournisseurs (non captés)'::text as fournisseur, count(*)::int as nb_mag
  from parc_sans_achats
)

select fournisseur, nb_mag from capt_top
union all
select fournisseur, nb_mag from capt_rest where nb_mag > 0
union all
select fournisseur, nb_mag from autres_non_captes;
"""
    return read_df(sql, params=tuple(list(parc_params) + achats_params))


@st.cache_data(ttl=600, show_spinner=False)
def camembert_fusion_franchise_parc_sales(
    mags_cte_sql_: str,
    mags_cte_params_: tuple,
    code_op: str,
    top_n_fournisseurs: int = 12
) -> pd.DataFrame:
    parc_sql, parc_params = _parc_sales_cte(mags_cte_sql_, mags_cte_params_)
    achats_cte, achats_params = _achats_norm_cte(code_op)

    sql = f"""
{parc_sql},
{achats_cte},

ref_franchise as (
  select
    trim(rm.code_magasin::text) as code_magasin,
    upper(trim(coalesce(rm.prenom_franchise,''))) || '||' ||
    upper(trim(coalesce(rm.nom_franchise,''))) as franchise_key
  from public.ref_magasin rm
  where trim(coalesce(rm.prenom_franchise,'')) <> ''
     or trim(coalesce(rm.nom_franchise,'')) <> ''
),

franchises_parc as (
  select distinct rf.franchise_key
  from parc_sales p
  join ref_franchise rf
    on rf.code_magasin = p.code_magasin
),

franchise_fournisseurs as (
  select distinct
    rf.franchise_key,
    coalesce(nullif(trim(a.fournisseur_norm::text),''), 'Fournisseur inconnu') as fournisseur
  from a_norm a
  join parc_sales p
    on p.code_magasin = a.code_magasin_canon
  join ref_franchise rf
    on rf.code_magasin = a.code_magasin_canon
),

capt_counts as (
  select
    fournisseur,
    count(distinct franchise_key)::int as nb_franchises
  from franchise_fournisseurs
  group by 1
  order by nb_franchises desc
),

capt_top as (
  select fournisseur, nb_franchises
  from capt_counts
  order by nb_franchises desc, fournisseur asc
  limit {int(top_n_fournisseurs)}
),

capt_rest as (
  select
    'Autres fournisseurs (captés)'::text as fournisseur,
    coalesce(sum(nb_franchises),0)::int as nb_franchises
  from capt_counts
  where fournisseur not in (select fournisseur from capt_top)
),

franchises_avec_achats as (
  select distinct franchise_key
  from franchise_fournisseurs
),

autres_non_captes as (
  select
    'Autres fournisseurs (non captés)'::text as fournisseur,
    count(*)::int as nb_franchises
  from franchises_parc fp
  where fp.franchise_key not in (
    select franchise_key from franchises_avec_achats
  )
)

select fournisseur, nb_franchises as nb_mag from capt_top
union all
select fournisseur, nb_franchises as nb_mag from capt_rest where nb_franchises > 0
union all
select fournisseur, nb_franchises as nb_mag from autres_non_captes;
"""
    return read_df(sql, params=tuple(list(parc_params) + achats_params))


@st.cache_data(ttl=600, show_spinner=False)
def pum_par_fournisseur_parc_sales(mags_cte_sql_: str, mags_cte_params_: tuple, code_op: str) -> pd.DataFrame:
    parc_sql, parc_params = _parc_sales_cte(mags_cte_sql_, mags_cte_params_)
    achats_cte, achats_params = _achats_norm_cte(code_op)

    sql = f"""
{parc_sql},
{achats_cte}
select
  coalesce(nullif(trim(a.fournisseur_norm::text),''),'Fournisseur inconnu') as fournisseur,
  coalesce(sum(a.quantite),0) as qte,
  coalesce(sum(a."total achat"),0) as valeur,
  coalesce(sum(a."total achat") / nullif(sum(a.quantite)::numeric,0), 0) as pum
from a_norm a
join parc_sales p
  on p.code_magasin = a.code_magasin_canon
group by 1
order by valeur desc;
"""
    return read_df(sql, params=tuple(list(parc_params) + achats_params))


def render_pie_with_shared_palette(
    df: pd.DataFrame,
    all_labels: list[str],
    color_map: dict,
    value_label: str = "Nb magasins",
):
    if df.empty or df["nb_mag"].sum() == 0:
        st.info("Aucune donnée")
        return

    df2 = df.copy()
    df2["__order"] = df2["fournisseur"].apply(lambda x: all_labels.index(x) if x in all_labels else 9999)
    df2 = df2.sort_values("__order").drop(columns="__order")

    fig = px.pie(
        df2,
        names="fournisseur",
        values="nb_mag",
        hole=0.45,
        color="fournisseur",
        color_discrete_map=color_map,
        labels={"nb_mag": value_label},
    )
    fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        hovertemplate=f"<b>%{{label}}</b><br>{value_label} : %{{value}}<br>%{{percent}}<extra></extra>",
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(t=10, b=10, l=10, r=240),
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# HELPERS — récupérer les codes magasin des KPI
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def get_codes_parc_sales(mags_cte_sql_: str, mags_cte_params_: tuple) -> list[str]:
    parc_sql, parc_params = _parc_sales_cte(mags_cte_sql_, mags_cte_params_)

    sql = f"""
{parc_sql}
select distinct trim(code_magasin::text) as code_magasin
from parc_sales
where code_magasin is not null
order by 1
"""
    df = read_df(sql, params=parc_params)

    if df.empty:
        return []

    return df["code_magasin"].astype(str).str.strip().tolist()


@st.cache_data(ttl=600, show_spinner=False)
def get_codes_mag_achats_references(
    mags_cte_sql_: str,
    mags_cte_params_: tuple,
    code_op: str
) -> list[str]:
    parc_sql, parc_params = _parc_sales_cte(mags_cte_sql_, mags_cte_params_)
    achats_cte, achats_params = _achats_norm_cte(code_op)

    sql = f"""
{parc_sql},
{achats_cte}
select distinct a.code_magasin_canon as code_magasin
from a_norm a
join parc_sales p
  on p.code_magasin = a.code_magasin_canon
where a.code_magasin_canon is not null
order by 1
"""

    df = read_df(sql, params=tuple(list(parc_params) + achats_params))

    if df.empty:
        return []

    return df["code_magasin"].astype(str).str.strip().tolist()


@st.cache_data(ttl=600, show_spinner=False)
def count_distinct_franchises_from_codes(code_magasin_list: tuple[str, ...]) -> int:
    if not code_magasin_list:
        return 0

    sql = """
    select count(*) as nb_franchises
    from (
        select distinct
            upper(trim(coalesce(prenom_franchise,''))) || '||' ||
            upper(trim(coalesce(nom_franchise,''))) as franchise_key
        from public.ref_magasin
        where trim(code_magasin::text) = any(%s::text[])
          and (
            trim(coalesce(prenom_franchise, '')) <> ''
            or trim(coalesce(nom_franchise, '')) <> ''
          )
    ) t
    """

    df = read_df(sql, params=(list(code_magasin_list),))

    if df.empty:
        return 0

    return int(df.iloc[0]["nb_franchises"] or 0)


# =============================================================================
# KPI FRANCHISÉS (calculé à partir des KPI magasins)
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def achats_kpi_franchises_from_kpi_magasins(
    mags_cte_sql_: str,
    mags_cte_params_: tuple,
    code_op: str
) -> pd.Series:
    codes_parc = get_codes_parc_sales(mags_cte_sql_, mags_cte_params_)
    codes_achats = get_codes_mag_achats_references(mags_cte_sql_, mags_cte_params_, code_op)

    nb_franchises_parc = count_distinct_franchises_from_codes(tuple(codes_parc))
    nb_franchises_achats = count_distinct_franchises_from_codes(tuple(codes_achats))

    pct = 0.0
    if nb_franchises_parc > 0:
        pct = round(nb_franchises_achats / nb_franchises_parc * 100, 1)

    return pd.Series(
        {
            "nb_franchises_parc": nb_franchises_parc,
            "nb_franchises_achats": nb_franchises_achats,
            "pct_franchises_achats": pct,
        }
    )


# =============================================================================
# EN-TÊTE : KPI COMMERCE au-dessus
# =============================================================================
st.markdown("## 🧾 Contexte commerce (sur la période) — A vs B")
with st.spinner(SPINNER_TXT):
    cA = load_commerce_light(mags_cte_sql_A, mags_cte_params_A, dateA0, dateA1)
    cB = load_commerce_light(mags_cte_sql_B, mags_cte_params_B, dateB0, dateB1)

cc = st.columns(3)
with cc[0]:
    kpi_card_compare("CA total", cA["ca_total"], cB["ca_total"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 0))
with cc[1]:
    kpi_card_compare("Tickets total", cA["tickets_total"], cB["tickets_total"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
with cc[2]:
    kpi_card_compare("Panier moyen", cA["panier_moyen"], cB["panier_moyen"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))

st.divider()

# =============================================================================
# RENDER
# =============================================================================
if code_magasin_selected:
    with st.spinner(SPINNER_TXT):
        hasA = has_data_for_magasin(code_opA, code_magasin_selected)
        hasB = has_data_for_magasin(code_opB, code_magasin_selected)

    if (not hasA) and (not hasB):
        _info_no_achats_generic()
    else:
        st.markdown("## 🏬 Magasin sélectionné")
        with st.spinner(SPINNER_TXT):
            render_magasin_fiche(code_magasin_selected)
        st.divider()

        with st.spinner(SPINNER_TXT):
            kA = achats_kpi_magasin(code_opA, code_magasin_selected)
            kB = achats_kpi_magasin(code_opB, code_magasin_selected)

        st.markdown("## 🧾 Achats – Synthèse (magasin)")
        c1, c2, c3 = st.columns(3)

        with c1:
            kpi_card_compare(
                title="Valeur achats (TTC)",
                value_n=_f0(kA.get("valeur_achats_captee")),
                value_n1=_f0(kB.get("valeur_achats_captee")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_money(x, 0),
            )
        with c2:
            kpi_card_compare(
                title="Volume achats (qte)",
                value_n=_f0(kA.get("volume_achats_capte")),
                value_n1=_f0(kB.get("volume_achats_capte")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_int(x),
            )
        with c3:
            kpi_card_compare(
                title="PUM (Valeur TTC / Quantité)",
                value_n=_f0(kA.get("pum")),
                value_n1=_f0(kB.get("pum")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_money(x, 2),
            )

        st.divider()

        st.markdown("## 💶 Prix unitaire moyen (PUM) par fournisseur — magasin")
        st.caption("PUM = **Somme(Valeur achats TTC) / Somme(Quantités)** (pondéré).")

        top_n = st.slider("Top N fournisseurs (table PUM par valeur N)", min_value=3, max_value=30, value=12, step=1)

        with st.spinner(SPINNER_TXT):
            pumA = pum_par_fournisseur_magasin(code_opA, code_magasin_selected).rename(
                columns={"qte": "qte_A", "valeur": "valeur_A", "pum": "pum_A"}
            )
            pumB = pum_par_fournisseur_magasin(code_opB, code_magasin_selected).rename(
                columns={"qte": "qte_B", "valeur": "valeur_B", "pum": "pum_B"}
            )

        merged = pd.merge(pumA, pumB, on="fournisseur", how="outer").fillna(0)
        merged = merged.sort_values("valeur_A", ascending=False).head(top_n)

        merged["delta_pum_pct"] = merged.apply(
            lambda r: 0 if r["pum_B"] == 0 else (r["pum_A"] - r["pum_B"]) / abs(r["pum_B"]) * 100,
            axis=1,
        )

        display_df = merged.copy()
        display_df["PUM A"] = display_df["pum_A"].apply(lambda x: fmt_money(x, 2))
        display_df["PUM B"] = display_df["pum_B"].apply(lambda x: fmt_money(x, 2))
        display_df["Δ PUM %"] = display_df["delta_pum_pct"].apply(lambda x: f"{x:+.1f}%")
        display_df["Valeur A (TTC)"] = display_df["valeur_A"].apply(lambda x: fmt_money(x, 0))
        display_df["Valeur B (TTC)"] = display_df["valeur_B"].apply(lambda x: fmt_money(x, 0))
        display_df["Qte A"] = display_df["qte_A"].apply(fmt_int)
        display_df["Qte B"] = display_df["qte_B"].apply(fmt_int)

        st.dataframe(
            display_df[["fournisseur", "PUM A", "PUM B", "Δ PUM %", "Valeur A (TTC)", "Valeur B (TTC)", "Qte A", "Qte B"]],
            use_container_width=True,
            hide_index=True,
        )

else:
    with st.spinner(SPINNER_TXT):
        if not has_sales_parc(mags_cte_sql_A, mags_cte_params_A):
            no_data_block("Aucun magasin dans le parc VENTES (tickets) avec les filtres actuels.")

        hasAchA = has_achats_for_sales_parc(mags_cte_sql_A, mags_cte_params_A, code_opA)
        hasAchB = has_achats_for_sales_parc(mags_cte_sql_B, mags_cte_params_B, code_opB)

    if (not hasAchA) and (not hasAchB):
        _info_no_achats_generic()
    else:
        with st.spinner(SPINNER_TXT):
            kA = achats_kpi_parc_sales(mags_cte_sql_A, mags_cte_params_A, code_opA)
            kB = achats_kpi_parc_sales(mags_cte_sql_B, mags_cte_params_B, code_opB)

        st.markdown("## 🧾 Achats – Synthèse")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            kpi_card_compare(
                title="Valeur achats (TTC)",
                value_n=_f0(kA.get("valeur_achats_captee")),
                value_n1=_f0(kB.get("valeur_achats_captee")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_money(x, 0),
            )
        with r1c2:
            kpi_card_compare(
                title="Volume achats (qte)",
                value_n=_f0(kA.get("volume_achats_capte")),
                value_n1=_f0(kB.get("volume_achats_capte")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_int(x),
            )

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            kpi_card_compare(
                title="Parc magasins (ventes)",
                value_n=_f0(kA.get("nb_mag_parc")),
                value_n1=_f0(kB.get("nb_mag_parc")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_int(x),
            )
        with r2c2:
            kpi_card_compare(
                title="Nb magasin en achats référencés",
                value_n=_f0(kA.get("nb_mag_acheteurs")),
                value_n1=_f0(kB.get("nb_mag_acheteurs")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_int(x),
            )
        with r2c3:
            kpi_card_compare(
                title="% magasin en achats référencés",
                value_n=_f0(kA.get("pct_mag_acheteurs")),
                value_n1=_f0(kB.get("pct_mag_acheteurs")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: f"{float(x or 0):.1f} %",
            )

        with st.spinner(SPINNER_TXT):
            fA = achats_kpi_franchises_from_kpi_magasins(mags_cte_sql_A, mags_cte_params_A, code_opA)
            fB = achats_kpi_franchises_from_kpi_magasins(mags_cte_sql_B, mags_cte_params_B, code_opB)

        st.divider()
        st.markdown("## 👤 Franchisés – Synthèse")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            kpi_card_compare(
                title="Parc franchisés (ventes)",
                value_n=_f0(fA.get("nb_franchises_parc")),
                value_n1=_f0(fB.get("nb_franchises_parc")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_int(x),
            )
        with fc2:
            kpi_card_compare(
                title="Nb franchisés en achats référencés",
                value_n=_f0(fA.get("nb_franchises_achats")),
                value_n1=_f0(fB.get("nb_franchises_achats")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: fmt_int(x),
            )
        with fc3:
            kpi_card_compare(
                title="% franchisés en achats référencés",
                value_n=_f0(fA.get("pct_franchises_achats")),
                value_n1=_f0(fB.get("pct_franchises_achats")),
                label_n=lib_opA,
                label_n1=lib_opB,
                formatter=lambda x: f"{float(x or 0):.1f} %",
            )

        st.divider()

        st.markdown("## 🥧 Répartition des magasins selon le fournisseur d’achat")
        st.caption("Répartition en **nombre de magasins** : chaque magasin capté est affecté à son **fournisseur principal** (valeur d’achat max).")

        with st.spinner(SPINNER_TXT):
            dfA = camembert_fusion_magasin_parc_sales(mags_cte_sql_A, mags_cte_params_A, code_opA, top_n_fournisseurs=12)
            dfB = camembert_fusion_magasin_parc_sales(mags_cte_sql_B, mags_cte_params_B, code_opB, top_n_fournisseurs=12)

        all_labels = sorted(set(dfA["fournisseur"].tolist()) | set(dfB["fournisseur"].tolist()))
        non_captes = "Autres fournisseurs (non captés)"
        if non_captes in all_labels:
            all_labels = [x for x in all_labels if x != non_captes] + [non_captes]

        color_map = build_color_map(all_labels)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {lib_opA}")
            render_pie_with_shared_palette(dfA, all_labels, color_map, value_label="Nb magasins")
        with c2:
            st.markdown(f"### {lib_opB}")
            render_pie_with_shared_palette(dfB, all_labels, color_map, value_label="Nb magasins")

        st.divider()

        st.markdown("## 👤 Répartition des franchisés selon le fournisseur d’achat")
        st.caption(
            "Répartition en **nombre de franchisés** : un franchisé est compté pour **chaque fournisseur** "
            "chez qui il a commandé sur l’ensemble de ses magasins du parc."
        )

        with st.spinner(SPINNER_TXT):
            dfFrA = camembert_fusion_franchise_parc_sales(
                mags_cte_sql_A, mags_cte_params_A, code_opA, top_n_fournisseurs=12
            )
            dfFrB = camembert_fusion_franchise_parc_sales(
                mags_cte_sql_B, mags_cte_params_B, code_opB, top_n_fournisseurs=12
            )

        all_labels_fr = sorted(set(dfFrA["fournisseur"].tolist()) | set(dfFrB["fournisseur"].tolist()))
        non_captes_fr = "Autres fournisseurs (non captés)"
        if non_captes_fr in all_labels_fr:
            all_labels_fr = [x for x in all_labels_fr if x != non_captes_fr] + [non_captes_fr]

        color_map_fr = build_color_map(all_labels_fr)

        cfr1, cfr2 = st.columns(2)
        with cfr1:
            st.markdown(f"### {lib_opA}")
            render_pie_with_shared_palette(dfFrA, all_labels_fr, color_map_fr, value_label="Nb franchisés")
        with cfr2:
            st.markdown(f"### {lib_opB}")
            render_pie_with_shared_palette(dfFrB, all_labels_fr, color_map_fr, value_label="Nb franchisés")

        st.divider()

        st.markdown("## 💶 Prix unitaire moyen (PUM) par fournisseur")
        st.caption("PUM = **Somme(Valeur achats TTC) / Somme(Quantités)** (pondéré).")

        top_n = st.slider("Top N fournisseurs (table PUM par valeur A)", min_value=3, max_value=30, value=12, step=1)

        with st.spinner(SPINNER_TXT):
            pumA = pum_par_fournisseur_parc_sales(mags_cte_sql_A, mags_cte_params_A, code_opA).rename(
                columns={"qte": "qte_A", "valeur": "valeur_A", "pum": "pum_A"}
            )
            pumB = pum_par_fournisseur_parc_sales(mags_cte_sql_B, mags_cte_params_B, code_opB).rename(
                columns={"qte": "qte_B", "valeur": "valeur_B", "pum": "pum_B"}
            )

        merged = pd.merge(pumA, pumB, on="fournisseur", how="outer").fillna(0)
        merged = merged.sort_values("valeur_A", ascending=False).head(top_n)

        merged["delta_pum_pct"] = merged.apply(
            lambda r: 0 if r["pum_B"] == 0 else (r["pum_A"] - r["pum_B"]) / abs(r["pum_B"]) * 100,
            axis=1,
        )

        display_df = merged.copy()
        display_df["PUM A"] = display_df["pum_A"].apply(lambda x: fmt_money(x, 2))
        display_df["PUM B"] = display_df["pum_B"].apply(lambda x: fmt_money(x, 2))
        display_df["Δ PUM %"] = display_df["delta_pum_pct"].apply(lambda x: f"{x:+.1f}%")
        display_df["Valeur A (TTC)"] = display_df["valeur_A"].apply(lambda x: fmt_money(x, 0))
        display_df["Valeur B (TTC)"] = display_df["valeur_B"].apply(lambda x: fmt_money(x, 0))
        display_df["Qte A"] = display_df["qte_A"].apply(fmt_int)
        display_df["Qte B"] = display_df["qte_B"].apply(fmt_int)

        st.dataframe(
            display_df[["fournisseur", "PUM A", "PUM B", "Δ PUM %", "Valeur A (TTC)", "Valeur B (TTC)", "Qte A", "Qte B"]],
            use_container_width=True,
            hide_index=True,
        )
