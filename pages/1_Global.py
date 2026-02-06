# pages/1_Global.py
# -----------------------------------------------------------------------------
# GLOBAL — Dashboard (A vs B) — VERSION OPTIMISÉE (comme Commerce)
#
# Objectif demandé :
# - "On s'en fout actif/fermé" -> base = magasins ayant des ventes, point.
# - On applique les filtres -> on obtient un parc de codes magasins A et B
# - On calcule KPI via ANY(%s) (rapide) comme Commerce
#
# Fix IDF :
# - On ne dépend plus de crp_region_nationale... (souvent vide/incohérent)
# - IDF = CP département ∈ {75,77,78,91,92,93,94,95}
# - PARIS = CP département = 75 (plus fiable que ville)
#
# Perf :
# - fetch_selected_mags() 1 seule fois pour A/B
# - ref_magasin chargé 1 seule fois sur union(codes A/B)
# - scopes filtrés en Python
# - KPI/Counts en SQL via ANY(%s)
#
# ✅ FIX COMPARABLE (codes canoniques) :
# - Les codes "mags" sont canoniques, mais les ventes peuvent contenir l'ancien code
# - Donc on canonise aussi st.code_magasin via vw_param_magasin_ancien_code dans TOUTES les requêtes ventes
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd

from src.auth import require_auth
from src.ui import top_bar, tabs_nav
from src.db import read_df
from src.filters import render_filters, fetch_selected_mags
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

filters = ctx.get("filters", {})
code_magasin_selected = (filters.get("code_magasin") or "").strip().upper() or None

st.caption(
    f"Année N : **{lib_opA}** ({dateA0} → {dateA1})  |  "
    f"Année N-1 : **{lib_opB}** ({dateB0} → {dateB1})"
)

# =========================
# Helpers
# =========================
def _safe_float(x) -> float:
    try:
        return float(x or 0)
    except Exception:
        return 0.0


def _norm_code_list(codes: list[str]) -> list[str]:
    if not codes:
        return []
    out = []
    for c in codes:
        if c is None:
            continue
        s = str(c).strip().upper()
        if s:
            out.append(s)
    return sorted(set(out))


def _dept_from_cp(cp: str) -> str | None:
    if not cp:
        return None
    s = str(cp).strip()
    if not s:
        return None
    s = "".join(ch for ch in s if ch.isdigit())
    if len(s) < 2:
        return None
    return s[:2]


IDF_DEPTS = {"75", "77", "78", "91", "92", "93", "94", "95"}


# =========================
# Load codes magasins A/B (comme Commerce)
# =========================
with st.spinner(SPINNER_TXT):
    magsA_codes = _norm_code_list(fetch_selected_mags(mags_cte_sql_A, mags_cte_params_A))
    magsB_codes = _norm_code_list(fetch_selected_mags(mags_cte_sql_B, mags_cte_params_B))

st.markdown("## 🧩 Global — KPI (A vs B)")
st.caption(f"Magasins sélectionnés (parc filtres) : **{len(magsA_codes)}** vs **{len(magsB_codes)}**")
st.divider()

# =========================
# Ref magasin (pour scopes + fiche magasin)
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_ref_magasin_for_codes(codes: list[str]) -> pd.DataFrame:
    if not codes:
        return pd.DataFrame(
            columns=[
                "code_magasin",
                "nom_magasin",
                "ville",
                "cp",
                "pays",
                "type",
                "statut",
                "rcr",
                "rdr",
                "telephone",
                "e_mail",
                "adresse",
                "nom_franchise",
                "prenom_franchise",
                "telephone_franchise",
            ]
        )
    sql = """
select
  upper(trim(rm.code_magasin::text)) as code_magasin,
  rm.nom_magasin,
  coalesce(rm."adresses_:_adresse_de_livraison_:_ville",'') as ville,
  coalesce(rm."adresses_:_adresse_de_livraison_:_cp",'') as cp,
  coalesce(rm."adresses_:_adresse_de_livraison_:_pays",'') as pays,
  rm.type,
  rm.statut,
  rm.rcr,
  rm.rdr,
  rm.telephone,
  rm.e_mail,
  rm."adresses_:_adresse_de_livraison_:_adresse_1" as adresse,
  rm.nom_franchise,
  rm.prenom_franchise,
  rm.telephone_franchise
from public.ref_magasin rm
where upper(trim(rm.code_magasin::text)) = any(%s);
"""
    return read_df(sql, (codes,))


with st.spinner(SPINNER_TXT):
    codes_union = _norm_code_list(magsA_codes + magsB_codes)
    df_ref = load_ref_magasin_for_codes(codes_union)

ref_map = {}
if not df_ref.empty:
    ref_map = {r["code_magasin"]: r for _, r in df_ref.iterrows()}

# =========================
# Magasins présents dans les ventes (sur période)
# ✅ canonise st.code_magasin (ancien_code -> code_magasin)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def sales_present_codes(codes: list[str], date_debut: str, date_fin: str) -> list[str]:
    if not codes:
        return []
    sql = """
select distinct
  coalesce(mp.code_magasin, upper(trim(st.code_magasin::text))) as code_magasin
from public.vw_gold_tickets_jour_clean_op st
left join public.vw_param_magasin_ancien_code mp
  on mp.ancien_code = upper(trim(st.code_magasin::text))
where st.ticket_date >= %s::date
  and st.ticket_date <= %s::date
  and coalesce(mp.code_magasin, upper(trim(st.code_magasin::text))) = any(%s);
"""
    df = read_df(sql, (date_debut, date_fin, codes))
    if df.empty:
        return []
    return _norm_code_list(df["code_magasin"].tolist())


@st.cache_data(ttl=600, show_spinner=False)
def kpi_from_codes(codes: list[str], date_debut: str, date_fin: str) -> dict:
    if not codes:
        return {
            "ca": 0.0,
            "tickets": 0.0,
            "qte_articles": 0.0,
            "panier_moyen": 0.0,
            "indice_vente": 0.0,
            "prix_moyen_article": 0.0,
        }
    sql = """
select
  coalesce(sum(st.total_ttc_net),0)::numeric as ca,
  coalesce(sum(st.nb_tickets),0)::numeric as tickets,
  coalesce(sum(st.qte_article),0)::numeric as qte_articles,
  round(coalesce(sum(st.total_ttc_net),0) / nullif(coalesce(sum(st.nb_tickets),0),0), 2) as panier_moyen,
  round(coalesce(sum(st.qte_article),0)::numeric / nullif(coalesce(sum(st.nb_tickets),0),0), 2) as indice_vente,
  round(coalesce(sum(st.total_ttc_net),0) / nullif(coalesce(sum(st.qte_article),0),0), 2) as prix_moyen_article
from public.vw_gold_tickets_jour_clean_op st
left join public.vw_param_magasin_ancien_code mp
  on mp.ancien_code = upper(trim(st.code_magasin::text))
where st.ticket_date >= %s::date
  and st.ticket_date <= %s::date
  and coalesce(mp.code_magasin, upper(trim(st.code_magasin::text))) = any(%s);
"""
    df = read_df(sql, (date_debut, date_fin, codes))
    if df.empty:
        return {
            "ca": 0.0,
            "tickets": 0.0,
            "qte_articles": 0.0,
            "panier_moyen": 0.0,
            "indice_vente": 0.0,
            "prix_moyen_article": 0.0,
        }
    r = df.iloc[0].to_dict()
    return {k: _safe_float(r.get(k)) for k in ["ca", "tickets", "qte_articles", "panier_moyen", "indice_vente", "prix_moyen_article"]}


# =========================
# Scopes (Python) — basés CP
# =========================
def filter_codes_scope(codes: list[str], scope: str) -> list[str]:
    if not codes:
        return []

    scope = (scope or "ALL").upper()
    if scope == "ALL":
        return codes

    kept = []
    for c in codes:
        r = ref_map.get(c)
        cp = (r.get("cp") if r is not None else "") if isinstance(r, dict) else (r["cp"] if r is not None else "")
        dept = _dept_from_cp(cp)

        if scope == "PARIS":
            if dept == "75":
                kept.append(c)

        elif scope == "IDF":
            if dept in IDF_DEPTS:
                kept.append(c)

        elif scope == "PROVINCE":
            if dept is None:
                continue
            if dept not in IDF_DEPTS:
                kept.append(c)

        else:
            kept.append(c)

    return _norm_code_list(kept)


# =========================
# Fiche magasin (simple)
# =========================
def render_magasin_fiche(code_magasin: str):
    code_magasin = (code_magasin or "").strip().upper()
    r = ref_map.get(code_magasin)
    if r is None:
        st.warning("Magasin introuvable dans ref_magasin.")
        return

    nom = r.get("nom_magasin") or ""
    tel = (r.get("telephone") or "").strip()
    mail = (r.get("e_mail") or "").strip()
    adr = (r.get("adresse") or "").strip()
    cp = (r.get("cp") or "").strip()
    ville = (r.get("ville") or "").strip()
    pays = (r.get("pays") or "").strip()

    statut = (r.get("statut") or "").strip()
    badges = []
    if statut:
        cls = "badge-ok" if ("actif" in statut.lower() or "ouvert" in statut.lower()) else "badge-warn"
        badges.append((statut, cls))

    nom_f = (r.get("nom_franchise") or "").strip()
    prenom_f = (r.get("prenom_franchise") or "").strip()
    tel_f = (r.get("telephone_franchise") or "").strip()

    title = f"{code_magasin} — {nom}"
    subtitle = "  •  ".join([x for x in [f"📞 {tel}" if tel else "", f"✉️ {mail}" if mail else ""] if x])

    col_left = [("Type :", r.get("type") or "—"), ("RCR :", r.get("rcr") or "—"), ("RDR :", r.get("rdr") or "—")]
    col_mid = [("Adresse :", adr or "—"), ("CP / Ville :", f"{cp} {ville}".strip() or "—"), ("Pays :", pays or "—")]
    col_right = []
    if nom_f or prenom_f:
        col_right.append(("Franchisé :", f"{prenom_f} {nom_f}".strip()))
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


# =========================
# Block rendu scope
# =========================
def render_scope_block(title: str, scope_code: str):
    st.markdown(f"## {title}")

    # Codes scope (parc filtres)
    codesA_scope = filter_codes_scope(magsA_codes, scope_code)
    codesB_scope = filter_codes_scope(magsB_codes, scope_code)

    # Magasins présents dans les ventes (✅ canon)
    presentA = sales_present_codes(codesA_scope, dateA0, dateA1)
    presentB = sales_present_codes(codesB_scope, dateB0, dateB1)

    st.caption(
        f"Magasins (présents dans les ventes sur la période) — Année N: **{len(presentA)}** | Année N-1: **{len(presentB)}**"
    )

    # KPI calculés sur les codes scope (✅ canon)
    rA = kpi_from_codes(codesA_scope, dateA0, dateA1)
    rB = kpi_from_codes(codesB_scope, dateB0, dateB1)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card_compare("CA", rA["ca"], rB["ca"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 0))
    with c2:
        kpi_card_compare("Tickets", rA["tickets"], rB["tickets"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
    with c3:
        kpi_card_compare("Panier moyen", rA["panier_moyen"], rB["panier_moyen"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))

    c4, c5, c6 = st.columns(3)
    with c4:
        kpi_card_compare("Articles vendus", rA["qte_articles"], rB["qte_articles"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
    with c5:
        kpi_card_compare("Prix moyen article", rA["prix_moyen_article"], rB["prix_moyen_article"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))
    with c6:
        kpi_card_compare("Indice de vente", rA["indice_vente"], rB["indice_vente"], lib_opA, lib_opB, formatter=lambda x: f"{float(x or 0):.2f}")

    st.divider()


# =========================
# LOGIQUE D'AFFICHAGE
# =========================
if code_magasin_selected:
    st.markdown("## 🏬 Magasin sélectionné")
    render_magasin_fiche(code_magasin_selected)
    st.divider()

    codes_store = [code_magasin_selected]

    rA = kpi_from_codes(codes_store, dateA0, dateA1)
    rB = kpi_from_codes(codes_store, dateB0, dateB1)

    st.markdown("## 📌 Performance (N vs N-1) — magasin")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card_compare("CA", rA["ca"], rB["ca"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 0))
    with c2:
        kpi_card_compare("Tickets", rA["tickets"], rB["tickets"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
    with c3:
        kpi_card_compare("Panier moyen", rA["panier_moyen"], rB["panier_moyen"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))

    c4, c5, c6 = st.columns(3)
    with c4:
        kpi_card_compare("Articles vendus", rA["qte_articles"], rB["qte_articles"], lib_opA, lib_opB, formatter=lambda x: fmt_int(x))
    with c5:
        kpi_card_compare("Prix moyen article", rA["prix_moyen_article"], rB["prix_moyen_article"], lib_opA, lib_opB, formatter=lambda x: fmt_money(x, 2))
    with c6:
        kpi_card_compare("Indice de vente", rA["indice_vente"], rB["indice_vente"], lib_opA, lib_opB, formatter=lambda x: f"{float(x or 0):.2f}")

else:
    render_scope_block("Tous les magasins", "ALL")
    render_scope_block("Paris", "PARIS")
    render_scope_block("Île-de-France", "IDF")
    render_scope_block("Province", "PROVINCE")
