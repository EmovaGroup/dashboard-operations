# pages/1_Global.py
# -----------------------------------------------------------------------------
# GLOBAL — Dashboard (A vs B)
#
# ✅ Logique corrigée :
# - Parc A = ctx["mags_cte_sql_A"] + params_A
# - Parc B = ctx["mags_cte_sql_B"] + params_B
# - KPI calculés sur vw_gold_tickets_jour_clean_op sur la période de l’opération (vw_operations_dedup)
#
# ✅ Fix IMPORTANT :
# - Mode magasin : KPI calculés UNIQUEMENT sur ce magasin
# - Mode parc : KPI A sur parc A, KPI B sur parc B (avant : bug => B compté sur parc A)
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd

from src.auth import require_auth
from src.ui import top_bar, tabs_nav
from src.db import read_df
from src.components import (
    inject_kpi_css,
    inject_kpi_compare_css,
    kpi_card_compare,
    fmt_money,
    fmt_int,
    inject_store_css,
    store_card_3col_html,
    st_meteo_asset,
)
from src.filters import render_filters

st.set_page_config(page_title="Global", layout="wide")
require_auth()

top_bar("Dashboard – Global")
tabs_nav()
st.divider()

inject_kpi_css()
inject_kpi_compare_css()
inject_store_css()

# =========================
# Filtres
# =========================
ctx = render_filters()

code_opA = ctx["opA"]["code"]
lib_opA = ctx["opA"]["lib"]
code_opB = ctx["opB"]["code"]
lib_opB = ctx["opB"]["lib"]

# ✅ FIX : parcs séparés A/B
mags_cte_sql_A = ctx["mags_cte_sql_A"]
mags_cte_params_A = ctx["mags_cte_params_A"]
mags_cte_sql_B = ctx["mags_cte_sql_B"]
mags_cte_params_B = ctx["mags_cte_params_B"]

filters = ctx.get("filters", {})
code_magasin_selected = filters.get("code_magasin")

st.caption(
    f"Année N : **{lib_opA}** ({ctx['opA']['date_debut']} → {ctx['opA']['date_fin']})  |  "
    f"Année N-1 : **{lib_opB}** ({ctx['opB']['date_debut']} → {ctx['opB']['date_fin']})"
)

# =========================
# Helpers schema (robuste colonnes)
# =========================
@st.cache_data(ttl=3600)
def table_columns(schema: str, table: str) -> set[str]:
    df = read_df(
        """
        select column_name
        from information_schema.columns
        where table_schema = %s and table_name = %s
        """,
        params=(schema, table),
    )
    return set(df["column_name"].tolist()) if not df.empty else set()


def qident(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def pick_first(existing: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in existing:
            return c
    return None


def sel_or_null(col: str | None, alias: str, existing_cols: set[str]) -> str:
    if not col or col not in existing_cols:
        return f"null::text as {alias}"
    return f"rm.{qident(col)} as {alias}"


# =========================
# CTE mags override pour MODE MAGASIN
# =========================
def mags_cte_for_store(code_magasin: str) -> tuple[str, tuple]:
    sql = """
WITH mags as (
  select trim(%s::text) as code_magasin
)
""".strip()
    return sql, (code_magasin,)


# =========================
# Scopes (Tous / Paris / IDF / Province)
# =========================
def _scope_clause_and_params(scope: str):
    if scope == "PARIS":
        return "coalesce(rm.ville,'') ilike %s", ["paris%"]

    if scope == "IDF":
        return """
        (
          coalesce(rm."crp_:_region_nationale_d_affectation",'') ilike %s
          OR coalesce(rm."crp_:_region_nationale_d_affectation",'') ilike %s
        )
        """, ["%ile%france%", "%île%france%"]

    if scope == "PROVINCE":
        return """
        NOT (coalesce(rm.ville,'') ilike %s)
        AND NOT (
          coalesce(rm."crp_:_region_nationale_d_affectation",'') ilike %s
          OR coalesce(rm."crp_:_region_nationale_d_affectation",'') ilike %s
        )
        """, ["paris%", "%ile%france%", "%île%france%"]

    return "1=1", []


# =========================
# KPI query (agrégé période) — vw_operations_dedup
# =========================
def kpi_query(code_op: str, scope: str, mags_sql: str, mags_params: tuple) -> pd.DataFrame:
    scope_sql, scope_params = _scope_clause_and_params(scope)

    sql = f"""
{mags_sql},
op as (
  select date_debut, date_fin
  from public.vw_operations_dedup
  where code_operation = %s
  limit 1
)
select
  coalesce(sum(st.total_ttc_net),0) as ca,
  coalesce(sum(st.nb_tickets),0) as tickets,
  coalesce(sum(st.qte_article),0) as qte_articles,
  round(coalesce(sum(st.total_ttc_net),0) / nullif(coalesce(sum(st.nb_tickets),0),0), 2) as panier_moyen,
  round(coalesce(sum(st.qte_article),0)::numeric / nullif(coalesce(sum(st.nb_tickets),0),0), 2) as indice_vente,
  round(coalesce(sum(st.total_ttc_net),0) / nullif(coalesce(sum(st.qte_article),0),0), 2) as prix_moyen_article
from public.vw_gold_tickets_jour_clean_op st
join mags m
  on m.code_magasin = trim(st.code_magasin::text)
join op
  on st.ticket_date between op.date_debut and op.date_fin
left join public.ref_magasin rm
  on trim(rm.code_magasin::text) = trim(st.code_magasin::text)
where {scope_sql};
"""
    params = (*mags_params, code_op, *scope_params)
    df = read_df(sql, params=params)
    if df.empty:
        return pd.DataFrame([{
            "ca": 0, "tickets": 0, "qte_articles": 0,
            "panier_moyen": 0, "indice_vente": 0, "prix_moyen_article": 0
        }])
    return df


def count_magasin_query(code_op: str, scope: str, mags_sql: str, mags_params: tuple) -> int:
    scope_sql, scope_params = _scope_clause_and_params(scope)

    sql = f"""
{mags_sql},
op as (
  select date_debut, date_fin
  from public.vw_operations_dedup
  where code_operation = %s
  limit 1
)
select count(distinct trim(st.code_magasin::text)) as nb_magasin
from public.vw_gold_tickets_jour_clean_op st
join mags m on m.code_magasin = trim(st.code_magasin::text)
join op on st.ticket_date between op.date_debut and op.date_fin
left join public.ref_magasin rm on trim(rm.code_magasin::text) = trim(st.code_magasin::text)
where {scope_sql};
"""
    params = (*mags_params, code_op, *scope_params)
    df = read_df(sql, params=params)
    return int(df["nb_magasin"].iloc[0] or 0) if not df.empty else 0


# =========================
# DAILY SERIES (Jour 1..N)
# =========================
def daily_kpis_query(code_op: str, mags_sql: str, mags_params: tuple) -> pd.DataFrame:
    sql = f"""
{mags_sql},
op as (
  select date_debut, date_fin
  from public.vw_operations_dedup
  where code_operation = %s
  limit 1
)
select
  (st.ticket_date - op.date_debut + 1) as day_idx,
  coalesce(sum(st.total_ttc_net),0) as ca,
  coalesce(sum(st.nb_tickets),0) as tickets,
  coalesce(sum(st.qte_article),0) as qte_articles,
  round(coalesce(sum(st.total_ttc_net),0) / nullif(coalesce(sum(st.nb_tickets),0),0), 2) as panier_moyen,
  round(coalesce(sum(st.qte_article),0)::numeric / nullif(coalesce(sum(st.nb_tickets),0),0), 2) as indice_vente,
  round(coalesce(sum(st.total_ttc_net),0) / nullif(coalesce(sum(st.qte_article),0),0), 2) as prix_moyen_article
from public.vw_gold_tickets_jour_clean_op st
join mags m on m.code_magasin = trim(st.code_magasin::text)
join op on st.ticket_date between op.date_debut and op.date_fin
group by (st.ticket_date - op.date_debut + 1)
order by day_idx;
"""
    df = read_df(sql, params=(*mags_params, code_op))
    if df.empty:
        return df
    df["day_idx"] = df["day_idx"].astype(int)
    return df


# =========================
# FICHE MAGASIN
# =========================
def magasin_info_query(code_magasin: str) -> pd.DataFrame:
    cols = table_columns("public", "ref_magasin")

    col_addr = pick_first(cols, [
        "adresse_:_adresse_de_livraison_:_adresse_1",
        "adresses_:_adresse_de_livraison_:_adresse_1",
    ])
    col_cp = pick_first(cols, [
        "adresse_:_adresse_de_livraison_:_cp",
        "adresses_:_adresse_de_livraison_:_cp",
    ])
    col_ville = pick_first(cols, [
        "adresse_:_adresse_de_livraison_:_ville",
        "adresses_:_adresse_de_livraison_:_ville",
    ])
    col_pays = pick_first(cols, [
        "adresse_:_adresse_de_livraison_:_pays",
        "adresses_:_adresse_de_livraison_:_pays",
    ])

    sql = f"""
select
  trim(rm.code_magasin::text) as code_magasin,
  rm.nom_magasin,
  {sel_or_null("telephone", "telephone", cols)},
  {sel_or_null("e_mail", "e_mail", cols)},
  {sel_or_null(col_addr, "adresse", cols)},
  {sel_or_null(col_cp, "cp", cols)},
  {sel_or_null(col_ville, "ville", cols)},
  {sel_or_null(col_pays, "pays", cols)},
  {sel_or_null("type", "type", cols)},
  {sel_or_null("statut", "statut", cols)},
  {sel_or_null("rcr", "rcr", cols)},
  {sel_or_null("rdr", "rdr", cols)},
  {sel_or_null("nom_franchise", "nom_franchise", cols)},
  {sel_or_null("prenom_franchise", "prenom_franchise", cols)},
  {sel_or_null("telephone_franchise", "telephone_franchise", cols)}
from public.ref_magasin rm
where trim(rm.code_magasin::text) = %s
limit 1;
"""
    return read_df(sql, params=(code_magasin,))


def meteo_filename_for_op(code_op: str, mags_sql: str, mags_params: tuple) -> str | None:
    sql = f"""
{mags_sql},
op as (
  select date_debut, date_fin
  from public.vw_operations_dedup
  where code_operation = %s
  limit 1
)
select st.asset_file_meteo
from public.vw_gold_tickets_jour_clean_op st
join mags m on m.code_magasin = trim(st.code_magasin::text)
join op on st.ticket_date between op.date_debut and op.date_fin
where coalesce(st.asset_file_meteo,'') <> ''
group by st.asset_file_meteo
order by count(*) desc
limit 1;
"""
    df = read_df(sql, params=(*mags_params, code_op))
    if df.empty:
        return None
    return df.iloc[0]["asset_file_meteo"]


def render_magasin_fiche(code_magasin: str):
    st.markdown("## 🏬 Fiche magasin")

    info = magasin_info_query(code_magasin)
    if info.empty:
        st.warning("Magasin introuvable dans ref_magasin.")
        return
    rm = info.iloc[0]

    statut = (rm.get("statut") or "").strip()
    badge_cls = "badge-neutral"
    if statut:
        s = statut.lower()
        if "ouvert" in s or "actif" in s:
            badge_cls = "badge-ok"
        elif "fermé" in s or "ferme" in s or "inactif" in s:
            badge_cls = "badge-warn"

    title = f"{rm['code_magasin']} — {rm.get('nom_magasin') or ''}"

    phone = (rm.get("telephone") or "").strip()
    email = (rm.get("e_mail") or "").strip()

    adresse = (rm.get("adresse") or "").strip()
    cp = (rm.get("cp") or "").strip()
    ville = (rm.get("ville") or "").strip()
    pays = (rm.get("pays") or "").strip()

    type_m = (rm.get("type") or "-")
    rcr = (rm.get("rcr") or "-")
    rdr = (rm.get("rdr") or "-")

    nom_f = (rm.get("nom_franchise") or "").strip()
    prenom_f = (rm.get("prenom_franchise") or "").strip()
    tel_f = (rm.get("telephone_franchise") or "").strip()
    is_franchise = bool(nom_f or prenom_f or tel_f)

    badges = []
    if statut:
        badges.append((statut, badge_cls))
    if is_franchise:
        badges.append(("Franchise", "badge-ok"))

    col_left = [("Code:", rm["code_magasin"]), ("Nom:", rm.get("nom_magasin") or "—")]
    if phone:
        col_left.append(("Téléphone:", phone))
    if email:
        col_left.append(("Email:", f'<a href="mailto:{email}">{email}</a>'))

    col_mid = [
        ("Adresse:", adresse or "—"),
        ("CP / Ville:", f"{cp} {ville}".strip() or "—"),
        ("Pays:", pays or "—"),
    ]

    col_right = [("Type:", type_m), ("RCR:", rcr), ("RDR:", rdr)]
    if is_franchise:
        franch_name = (prenom_f + " " + nom_f).strip()
        if franch_name:
            col_right.append(("Franchisé:", franch_name))
        if tel_f:
            col_right.append(("Tél. franchisé:", tel_f))

    store_card_3col_html(
        title=title,
        subtitle="",
        badges=badges,
        col_left=col_left,
        col_mid=col_mid,
        col_right=col_right,
        left_title="Identité & contact",
        mid_title="Adresse",
        right_title="Infos magasin",
    )

    store_mags_sql, store_mags_params = mags_cte_for_store(code_magasin)

    st.markdown("### 🌤️ Météo (icônes) — magasin")
    fileA = meteo_filename_for_op(code_opA, store_mags_sql, store_mags_params)
    fileB = meteo_filename_for_op(code_opB, store_mags_sql, store_mags_params)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"**Année N — {lib_opA}**")
        st_meteo_asset(fileA, width=72)
    with m2:
        st.markdown(f"**Année N-1 — {lib_opB}**")
        st_meteo_asset(fileB, width=72)

    st.divider()

    rA = kpi_query(code_opA, "ALL", store_mags_sql, store_mags_params).iloc[0]
    rB = kpi_query(code_opB, "ALL", store_mags_sql, store_mags_params).iloc[0]

    st.markdown("### 📌 Performance (N vs N-1)")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card_compare("CA", float(rA["ca"] or 0), float(rB["ca"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 0))
    with c2:
        kpi_card_compare("Tickets", float(rA["tickets"] or 0), float(rB["tickets"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
    with c3:
        kpi_card_compare("Panier moyen", float(rA["panier_moyen"] or 0), float(rB["panier_moyen"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))

    c4, c5, c6 = st.columns(3)
    with c4:
        kpi_card_compare("Articles vendus", float(rA["qte_articles"] or 0), float(rB["qte_articles"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
    with c5:
        kpi_card_compare("Prix moyen article", float(rA["prix_moyen_article"] or 0), float(rB["prix_moyen_article"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))
    with c6:
        kpi_card_compare("Indice de vente", float(rA["indice_vente"] or 0), float(rB["indice_vente"] or 0), lib_opA, lib_opB, formatter=lambda x: f"{x:.2f}")

    st.divider()

    st.markdown("### 📈 Évolution jour par jour (Jour 1 → Jour N) — superposition N vs N-1")

    dfA = daily_kpis_query(code_opA, store_mags_sql, store_mags_params)
    dfB = daily_kpis_query(code_opB, store_mags_sql, store_mags_params)

    if dfA.empty and dfB.empty:
        st.info("Aucune donnée journalière pour ce magasin sur les périodes sélectionnées.")
        return

    def _prep(df: pd.DataFrame, suffix: str):
        if df.empty:
            return pd.DataFrame(columns=["day_idx"])
        out = df.copy()
        out = out.rename(columns={
            "ca": f"ca{suffix}",
            "tickets": f"tickets{suffix}",
            "qte_articles": f"qte_articles{suffix}",
            "panier_moyen": f"panier_moyen{suffix}",
            "indice_vente": f"indice_vente{suffix}",
            "prix_moyen_article": f"prix_moyen_article{suffix}",
        })
        return out

    A = _prep(dfA, "_A")
    B = _prep(dfB, "_B")

    merged = pd.merge(A, B, on="day_idx", how="outer").sort_values("day_idx").set_index("day_idx")

    def chart_two(colA: str, colB: str, title: str):
        c = pd.DataFrame({
            lib_opA: merged[colA] if colA in merged.columns else None,
            lib_opB: merged[colB] if colB in merged.columns else None,
        })
        st.markdown(f"**{title}**")
        st.line_chart(c)

    chart_two("ca_A", "ca_B", "CA")
    chart_two("tickets_A", "tickets_B", "Tickets")
    chart_two("qte_articles_A", "qte_articles_B", "Articles vendus")
    chart_two("panier_moyen_A", "panier_moyen_B", "Panier moyen")
    chart_two("prix_moyen_article_A", "prix_moyen_article_B", "Prix moyen article")
    chart_two("indice_vente_A", "indice_vente_B", "Indice de vente")

    st.divider()


# =========================
# Vues scopes — parc A/B séparés
# =========================
def render_scope_block(title: str, scope_code: str):
    st.markdown(f"## {title}")

    nb_mag_A = count_magasin_query(code_opA, scope_code, mags_cte_sql_A, mags_cte_params_A)
    nb_mag_B = count_magasin_query(code_opB, scope_code, mags_cte_sql_B, mags_cte_params_B)
    st.caption(f"Magasins — Année N: **{nb_mag_A}** | Année N-1: **{nb_mag_B}**")

    rA = kpi_query(code_opA, scope_code, mags_cte_sql_A, mags_cte_params_A).iloc[0]
    rB = kpi_query(code_opB, scope_code, mags_cte_sql_B, mags_cte_params_B).iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card_compare("CA", float(rA["ca"] or 0), float(rB["ca"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 0))
    with c2:
        kpi_card_compare("Tickets", float(rA["tickets"] or 0), float(rB["tickets"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
    with c3:
        kpi_card_compare("Panier moyen", float(rA["panier_moyen"] or 0), float(rB["panier_moyen"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))

    c4, c5, c6 = st.columns(3)
    with c4:
        kpi_card_compare("Articles vendus", float(rA["qte_articles"] or 0), float(rB["qte_articles"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
    with c5:
        kpi_card_compare("Prix moyen article", float(rA["prix_moyen_article"] or 0), float(rB["prix_moyen_article"] or 0), lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))
    with c6:
        kpi_card_compare("Indice de vente", float(rA["indice_vente"] or 0), float(rB["indice_vente"] or 0), lib_opA, lib_opB, formatter=lambda x: f"{x:.2f}")

    st.divider()


# =========================
# LOGIQUE D'AFFICHAGE
# =========================
if code_magasin_selected:
    render_magasin_fiche(code_magasin_selected)
else:
    render_scope_block("Tous les magasins", "ALL")
    render_scope_block("Paris", "PARIS")
    render_scope_block("Île-de-France", "IDF")
    render_scope_block("Province", "PROVINCE")
