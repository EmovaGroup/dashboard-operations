# pages/3_Achats.py
# =============================================================================
# PAGE : ACHATS
# =============================================================================
# Objectif :
# - Comparer 2 opérations (Année N vs Année N-1) sur les achats
# - 2 modes :
#   1) Mode magasin : si un code_magasin est sélectionné -> fiche magasin + KPI + table PUM
#   2) Mode parc : sinon -> KPI parc + camemberts (répartition magasins par fournisseur) + table PUM
#
# Règles métiers :
# - Valeur achats = NET (champ "total achat")
# - Quantités = champ quantite
# - PUM = Somme(valeur NET) / Somme(quantités)  (pondéré)
#
# Fournisseurs :
# - On ne se base PAS sur le fournisseur brut
# - On se base sur la vue normalisée : public.vw_op_achats_norm (fournisseur_norm)
#
# UX :
# - Couleurs homogènes + légende à droite
# - Message clair quand il n’y a aucune donnée sur la période / filtres
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
code_opB = ctx["opB"]["code"]
lib_opB = ctx["opB"]["lib"]

mags_cte_sql = ctx["mags_cte_sql"]
mags_cte_params = ctx["mags_cte_params"]

code_magasin_selected = ctx["filters"]["code_magasin"]

st.caption(f"Année N : **{lib_opA}**  |  Année N-1 : **{lib_opB}**")

# -----------------------------------------------------------------------------
# Source achats normalisée
# -----------------------------------------------------------------------------
ACHATS_SRC = "public.vw_op_achats_norm"  # contient fournisseur_norm


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


def _i0(x) -> int:
    try:
        if x is None:
            return 0
        return int(x)
    except Exception:
        return 0


# =============================================================================
# Palette couleurs (UX-friendly)
# =============================================================================
BASE_PALETTE = [
    "#4E79A7",  # bleu
    "#59A14F",  # vert
    "#F28E2B",  # orange
    "#E15759",  # rouge doux
    "#B07AA1",  # violet
    "#76B7B2",  # turquoise
    "#EDC948",  # jaune doux
    "#FF9DA7",  # rose clair
    "#9C755F",  # brun
    "#BAB0AC",  # gris
]
COLOR_AUTRES_NON_CAPTES = "#6B7280"  # gris neutre


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
# FICHE MAGASIN
# =============================================================================
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
        cls = "badge-ok" if statut.lower() in ("actif", "ouverture") else "badge-warn"
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
# HELPERS SQL PARC
# =============================================================================
def _parc_analysis_cte(code_op: str) -> tuple[str, tuple]:
    """
    Parc d'analyse :
    - Base = mags (déjà filtré par filters.py : région, type, seg, rcr, etc.)
    - Actif / Ouverture
    - On garde aussi les acheteurs OP, mais uniquement si le magasin est déjà dans mags_dist
      => aucun magasin hors filtre ne revient
    """
    sql = f"""
    {mags_cte_sql},

    mags_dist as (
      select distinct code_magasin
      from mags
      where code_magasin is not null
    ),

    mags_actifs as (
      select md.code_magasin
      from mags_dist md
      join public.ref_magasin rm
        on trim(rm.code_magasin::text) = md.code_magasin
      where rm.statut in ('Actif','Ouverture')
    ),

    mags_acheteurs_op as (
      select distinct trim(a.code_magasin::text) as code_magasin
      from {ACHATS_SRC} a
      join mags_dist md
        on md.code_magasin = trim(a.code_magasin::text)
      where a.code_operation = %s
        and a.code_magasin is not null
    ),

    parc_analysis as (
      select code_magasin from mags_actifs
      union
      select code_magasin from mags_acheteurs_op
    )
    """
    return sql, (*mags_cte_params, code_op)


# =============================================================================
# Check data (message clair si rien)
# =============================================================================
def has_data_for_magasin(code_op: str, code_magasin: str) -> bool:
    sql = f"""
    select 1
    from {ACHATS_SRC} a
    where a.code_operation = %s
      and trim(a.code_magasin::text) = %s
    limit 1;
    """
    df = read_df(sql, params=(code_op, code_magasin))
    return not df.empty


def has_data_for_parc(code_op: str) -> bool:
    parc_sql, parc_params = _parc_analysis_cte(code_op)
    sql = f"""
    {parc_sql}
    select 1
    from {ACHATS_SRC} a
    join parc_analysis p on p.code_magasin = trim(a.code_magasin::text)
    where a.code_operation = %s
    limit 1;
    """
    df = read_df(sql, params=(*parc_params, code_op))
    return not df.empty


def no_data_block(msg: str):
    st.warning(
        f"📭 **Je n’ai pas de données pour cette période / ces filtres.**\n\n"
        f"{msg}\n\n"
        f"👉 Merci de choisir une autre opération (ou d’élargir les filtres)."
    )
    st.stop()


# =============================================================================
# MODE MAGASIN : KPI + table PUM
# =============================================================================
def achats_kpi_magasin(code_op: str, code_magasin: str) -> pd.Series:
    sql = f"""
    select
      coalesce(sum(a."total achat"),0) as valeur_achats_captee,
      coalesce(sum(a.quantite),0) as volume_achats_capte,
      coalesce(sum(a."total achat") / nullif(sum(a.quantite)::numeric,0), 0) as pum
    from {ACHATS_SRC} a
    where a.code_operation = %s
      and trim(a.code_magasin::text) = %s;
    """
    df = read_df(sql, params=(code_op, code_magasin))
    if df.empty:
        return pd.Series({"valeur_achats_captee": 0, "volume_achats_capte": 0, "pum": 0})
    return df.iloc[0]


def pum_par_fournisseur_magasin(code_op: str, code_magasin: str) -> pd.DataFrame:
    sql = f"""
    select
      coalesce(nullif(trim(a.fournisseur_norm::text),''),'Fournisseur inconnu') as fournisseur,
      coalesce(sum(a.quantite),0) as qte,
      coalesce(sum(a."total achat"),0) as valeur,
      coalesce(sum(a."total achat") / nullif(sum(a.quantite)::numeric,0), 0) as pum
    from {ACHATS_SRC} a
    where a.code_operation = %s
      and trim(a.code_magasin::text) = %s
    group by 1
    order by valeur desc;
    """
    return read_df(sql, params=(code_op, code_magasin))


# =============================================================================
# MODE PARC : KPI + camemberts + table PUM
# =============================================================================
def achats_kpi_parc(code_op: str) -> pd.Series:
    parc_sql, parc_params = _parc_analysis_cte(code_op)
    sql = f"""
    {parc_sql},

    achats_capt as (
      select
        trim(a.code_magasin::text) as code_magasin,
        coalesce(sum(a."total achat"),0) as valeur_achats,
        coalesce(sum(a.quantite),0) as volume_achats
      from {ACHATS_SRC} a
      join parc_analysis p on p.code_magasin = trim(a.code_magasin::text)
      where a.code_operation = %s
      group by 1
    ),

    parc as (select count(*) as nb_mag_parc from parc_analysis),
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
    df = read_df(sql, params=(*parc_params, code_op))
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


def camembert_fusion_magasin_parc(code_op: str, top_n_fournisseurs: int = 12) -> pd.DataFrame:
    parc_sql, parc_params = _parc_analysis_cte(code_op)
    sql = f"""
    {parc_sql},

    achats_mag_fourn as (
      select
        trim(a.code_magasin::text) as code_magasin,
        coalesce(nullif(trim(a.fournisseur_norm::text),''),'Fournisseur inconnu') as fournisseur,
        coalesce(sum(a."total achat"),0) as valeur
      from {ACHATS_SRC} a
      join parc_analysis p on p.code_magasin = trim(a.code_magasin::text)
      where a.code_operation = %s
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
      from parc_analysis p
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
    return read_df(sql, params=(*parc_params, code_op))


def pum_par_fournisseur_parc(code_op: str) -> pd.DataFrame:
    parc_sql, parc_params = _parc_analysis_cte(code_op)
    sql = f"""
    {parc_sql}
    select
      coalesce(nullif(trim(a.fournisseur_norm::text),''),'Fournisseur inconnu') as fournisseur,
      coalesce(sum(a.quantite),0) as qte,
      coalesce(sum(a."total achat"),0) as valeur,
      coalesce(sum(a."total achat") / nullif(sum(a.quantite)::numeric,0), 0) as pum
    from {ACHATS_SRC} a
    join parc_analysis p on p.code_magasin = trim(a.code_magasin::text)
    where a.code_operation = %s
    group by 1
    order by valeur desc;
    """
    return read_df(sql, params=(*parc_params, code_op))


def render_pie_with_shared_palette(df: pd.DataFrame, all_labels: list[str], color_map: dict):
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
    )
    fig.update_traces(textinfo="percent+label", textposition="inside")
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(t=10, b=10, l=10, r=240),
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# RENDER
# =============================================================================
if code_magasin_selected:
    # --- check data magasin (sur OP A) ---
    if not has_data_for_magasin(code_opA, code_magasin_selected):
        no_data_block(
            f"Magasin **{code_magasin_selected}** : aucune ligne d’achat sur **{lib_opA}**."
        )

    st.markdown("## 🏬 Magasin sélectionné")
    render_magasin_fiche(code_magasin_selected)
    st.divider()

    kA = achats_kpi_magasin(code_opA, code_magasin_selected)
    kB = achats_kpi_magasin(code_opB, code_magasin_selected)

    st.markdown("## 🧾 Achats – Synthèse (magasin)")
    c1, c2, c3 = st.columns(3)

    with c1:
        kpi_card_compare(
            title="Valeur achats NET",
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
            title="PUM (Valeur NET / Quantité)",
            value_n=_f0(kA.get("pum")),
            value_n1=_f0(kB.get("pum")),
            label_n=lib_opA,
            label_n1=lib_opB,
            formatter=lambda x: fmt_money(x, 2),
        )

    st.divider()

    st.markdown("## 💶 Prix unitaire moyen (PUM) par fournisseur — magasin")
    st.caption("PUM = **Somme(Valeur achats NET) / Somme(Quantités)** (pondéré).")

    top_n = st.slider("Top N fournisseurs (table PUM par valeur N)", min_value=3, max_value=30, value=12, step=1)

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
    display_df["PUM N"] = display_df["pum_A"].apply(lambda x: fmt_money(x, 2))
    display_df["PUM N-1"] = display_df["pum_B"].apply(lambda x: fmt_money(x, 2))
    display_df["Δ PUM %"] = display_df["delta_pum_pct"].apply(lambda x: f"{x:+.1f}%")
    display_df["Valeur N (NET)"] = display_df["valeur_A"].apply(lambda x: fmt_money(x, 0))
    display_df["Valeur N-1 (NET)"] = display_df["valeur_B"].apply(lambda x: fmt_money(x, 0))
    display_df["Qte N"] = display_df["qte_A"].apply(fmt_int)
    display_df["Qte N-1"] = display_df["qte_B"].apply(fmt_int)

    st.dataframe(
        display_df[
            ["fournisseur", "PUM N", "PUM N-1", "Δ PUM %", "Valeur N (NET)", "Valeur N-1 (NET)", "Qte N", "Qte N-1"]
        ],
        use_container_width=True,
        hide_index=True,
    )

else:
    # --- check data parc (sur OP A) ---
    if not has_data_for_parc(code_opA):
        no_data_block(f"Aucune ligne d’achat sur **{lib_opA}** avec les filtres actuels.")

    kA = achats_kpi_parc(code_opA)
    kB = achats_kpi_parc(code_opB)

    st.markdown("## 🧾 Achats – Synthèse")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        kpi_card_compare(
            title="Valeur achats (NET)",
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
            title="Parc magasins",
            value_n=_f0(kA.get("nb_mag_parc")),
            value_n1=_f0(kB.get("nb_mag_parc")),
            label_n=lib_opA,
            label_n1=lib_opB,
            formatter=lambda x: fmt_int(x),
        )
    with r2c2:
        kpi_card_compare(
            title="Nb magasins acheteurs",
            value_n=_f0(kA.get("nb_mag_acheteurs")),
            value_n1=_f0(kB.get("nb_mag_acheteurs")),
            label_n=lib_opA,
            label_n1=lib_opB,
            formatter=lambda x: fmt_int(x),
        )
    with r2c3:
        kpi_card_compare(
            title="% magasins acheteurs (sur parc)",
            value_n=_f0(kA.get("pct_mag_acheteurs")),
            value_n1=_f0(kB.get("pct_mag_acheteurs")),
            label_n=lib_opA,
            label_n1=lib_opB,
            formatter=lambda x: f"{x:.1f} %",
        )

    st.divider()

    st.markdown("## 🥧 Répartition des magasins selon le fournisseur d’achat")
    st.caption(
        "Répartition en **nombre de magasins** : chaque magasin capté est affecté à son **fournisseur principal** (valeur d’achat max)."
    )

    dfA = camembert_fusion_magasin_parc(code_opA, top_n_fournisseurs=12)
    dfB = camembert_fusion_magasin_parc(code_opB, top_n_fournisseurs=12)

    all_labels = sorted(set(dfA["fournisseur"].tolist()) | set(dfB["fournisseur"].tolist()))
    non_captes = "Autres fournisseurs (non captés)"
    if non_captes in all_labels:
        all_labels = [x for x in all_labels if x != non_captes] + [non_captes]

    color_map = build_color_map(all_labels)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {lib_opA}")
        render_pie_with_shared_palette(dfA, all_labels, color_map)
    with c2:
        st.markdown(f"### {lib_opB}")
        render_pie_with_shared_palette(dfB, all_labels, color_map)

    st.divider()

    st.markdown("## 💶 Prix unitaire moyen (PUM) par fournisseur")
    st.caption("PUM = **Somme(Valeur achats NET) / Somme(Quantités)** (pondéré).")

    top_n = st.slider("Top N fournisseurs (table PUM par valeur N)", min_value=3, max_value=30, value=12, step=1)

    pumA = pum_par_fournisseur_parc(code_opA).rename(columns={"qte": "qte_A", "valeur": "valeur_A", "pum": "pum_A"})
    pumB = pum_par_fournisseur_parc(code_opB).rename(columns={"qte": "qte_B", "valeur": "valeur_B", "pum": "pum_B"})

    merged = pd.merge(pumA, pumB, on="fournisseur", how="outer").fillna(0)
    merged = merged.sort_values("valeur_A", ascending=False).head(top_n)

    merged["delta_pum_pct"] = merged.apply(
        lambda r: 0 if r["pum_B"] == 0 else (r["pum_A"] - r["pum_B"]) / abs(r["pum_B"]) * 100,
        axis=1,
    )

    display_df = merged.copy()
    display_df["PUM N"] = display_df["pum_A"].apply(lambda x: fmt_money(x, 2))
    display_df["PUM N-1"] = display_df["pum_B"].apply(lambda x: fmt_money(x, 2))
    display_df["Δ PUM %"] = display_df["delta_pum_pct"].apply(lambda x: f"{x:+.1f}%")
    display_df["Valeur N (NET)"] = display_df["valeur_A"].apply(lambda x: fmt_money(x, 0))
    display_df["Valeur N-1 (NET)"] = display_df["valeur_B"].apply(lambda x: fmt_money(x, 0))
    display_df["Qte N"] = display_df["qte_A"].apply(fmt_int)
    display_df["Qte N-1"] = display_df["qte_B"].apply(fmt_int)

    st.dataframe(
        display_df[
            ["fournisseur", "PUM N", "PUM N-1", "Δ PUM %", "Valeur N (NET)", "Valeur N-1 (NET)", "Qte N", "Qte N-1"]
        ],
        use_container_width=True,
        hide_index=True,
    )
