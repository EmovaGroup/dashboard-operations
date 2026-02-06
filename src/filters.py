# src/filters.py
import streamlit as st
import pandas as pd
from src.db import read_df


# =========================
# DATA LOADERS
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def load_operations() -> pd.DataFrame:
    df = read_df("""
    select
      ro.code_operation,
      ro.libelle_op,
      min(ro.date_debut) as date_debut,
      max(ro.date_fin)   as date_fin
    from public.ref_operations ro
    group by ro.code_operation, ro.libelle_op
    order by min(ro.date_debut) desc;
    """)
    if df.empty:
        return df

    df["label"] = (
        df["libelle_op"]
        + " ("
        + df["date_debut"].astype(str)
        + " → "
        + df["date_fin"].astype(str)
        + ")"
    )
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_ref_magasin_distincts() -> dict:
    return {
        "magasins": read_df("""
            select distinct trim(code_magasin::text) as code_magasin, nom_magasin
            from public.ref_magasin
            where code_magasin is not null
            order by code_magasin;
        """),
        "types": read_df("""
            select distinct type
            from public.ref_magasin
            where type is not null and type <> ''
            order by type;
        """)["type"].tolist(),
        "seg": read_df("""
            select distinct "crp_:_segmentation" as seg
            from public.ref_magasin
            where "crp_:_segmentation" is not null and "crp_:_segmentation" <> ''
            order by seg;
        """)["seg"].tolist(),
        "rcr": read_df("""
            select distinct rcr
            from public.ref_magasin
            where rcr is not null and rcr <> ''
            order by rcr;
        """)["rcr"].tolist(),
        "reg_elargie": read_df("""
            select distinct "crp_:_region_elargie" as reg_elargie
            from public.ref_magasin
            where "crp_:_region_elargie" is not null and "crp_:_region_elargie" <> ''
            order by reg_elargie;
        """)["reg_elargie"].tolist(),
        "reg_admin": read_df("""
            select distinct "crp_:_region_nationale_d_affectation" as reg_admin
            from public.ref_magasin
            where "crp_:_region_nationale_d_affectation" is not null
              and "crp_:_region_nationale_d_affectation" <> ''
            order by reg_admin;
        """)["reg_admin"].tolist(),
        "enseignes": read_df("""
            select distinct enseigne
            from public.ref_magasin
            where enseigne is not null and enseigne <> ''
            order by enseigne;
        """)["enseigne"].tolist(),
    }


# =========================
# FILTER STATE
# =========================
def _init_state():
    ss = st.session_state

    ss.setdefault("flt_opA_label", None)
    ss.setdefault("flt_opB_label", None)

    ss.setdefault("flt_parc_mode_A", "Participants")
    ss.setdefault("flt_ermes_mode_A", "Tous")
    ss.setdefault("flt_fid_mode_A", "Tous")

    ss.setdefault("flt_parc_mode_B", "Participants")
    ss.setdefault("flt_ermes_mode_B", "Tous")
    ss.setdefault("flt_fid_mode_B", "Tous")

    ss.setdefault("flt_code_magasin", None)

    # ✅ comparable : "Tous" | "C" | "NC"
    ss.setdefault("flt_comp_mode", "Tous")

    ss.setdefault("flt_types", [])
    ss.setdefault("flt_seg", [])
    ss.setdefault("flt_rcr", [])
    ss.setdefault("flt_reg_elargie", [])
    ss.setdefault("flt_reg_admin", [])
    ss.setdefault("flt_enseignes", [])

    ss.setdefault("filters_applied", False)


def _safe_index(options: list[str], value: str | None, fallback: int = 0) -> int:
    if not options:
        return 0
    if value is None:
        return min(max(fallback, 0), len(options) - 1)
    try:
        return options.index(value)
    except ValueError:
        return min(max(fallback, 0), len(options) - 1)


# =========================
# BUILD BASE CTE "mags0" (A or B)
# =========================
def build_mags_cte_sql_base(
    code_operation_ref: str,
    parc_mode: str,
    ermes_mode: str,
    fid_mode: str,
    code_magasin: str | None,
    types: list[str],
    seg: list[str],
    rcr: list[str],
    reg_elargie: list[str],
    reg_admin: list[str],
    enseignes: list[str],
) -> tuple[str, tuple]:
    """
    Construit le parc "mags0" en 3 étapes :
      0) periode_op : période via ref_operations
      1) mags_parc  : magasins qui ont AU MOINS 1 ticket sur la période (GOLD agrégé)
      2) mags_base  : filtre OP (op_magasin) selon parc_mode
      3) mags0      : filtre ensuite ERMES puis FID
    """

    where_parts = ["rm.code_magasin is not null"]
    params: list = []

    if code_magasin:
        where_parts.append("upper(trim(rm.code_magasin::text)) = upper(trim(%s::text))")
        params.append(code_magasin)

    if types:
        where_parts.append("rm.type = any(%s)")
        params.append(types)
    if seg:
        where_parts.append('rm."crp_:_segmentation" = any(%s)')
        params.append(seg)
    if rcr:
        where_parts.append("rm.rcr = any(%s)")
        params.append(rcr)
    if reg_elargie:
        where_parts.append('rm."crp_:_region_elargie" = any(%s)')
        params.append(reg_elargie)
    if reg_admin:
        where_parts.append('rm."crp_:_region_nationale_d_affectation" = any(%s)')
        params.append(reg_admin)
    if enseignes:
        where_parts.append("rm.enseigne = any(%s)")
        params.append(enseignes)

    where_sql = " AND ".join(where_parts)

    periode_op_cte = """
periode_op as (
  select
    min(date_debut) as date_debut,
    max(date_fin)   as date_fin
  from public.ref_operations
  where code_operation = %s
)
""".strip()

    # GOLD (agrégé) + filtre ref_magasin (comme ton fichier d'origine)
    mags_parc_cte = f"""
mags_parc as (
  select distinct upper(trim(st.code_magasin::text)) as code_magasin
  from public.vw_gold_tickets_jour_clean_op st
  cross join periode_op p
  left join public.ref_magasin rm
    on upper(trim(rm.code_magasin::text)) = upper(trim(st.code_magasin::text))
  where st.code_magasin is not null
    and st.ticket_date >= p.date_debut
    and st.ticket_date <= p.date_fin
    and {where_sql}
)
""".strip()

    mags_op_cte = """
mags_op as (
  select distinct upper(trim(code_magasin::text)) as code_magasin
  from public.op_magasin
  where code_operation = %s
)
""".strip()

    if parc_mode == "Tous":
        mags_base_cte = """
mags_base as (
  select code_magasin from mags_parc
)
""".strip()
        parc_params = []
        include_mags_op = False

    elif parc_mode == "Participants":
        mags_base_cte = """
mags_base as (
  select p.code_magasin
  from mags_parc p
  join mags_op o using(code_magasin)

  union all

  select p.code_magasin
  from mags_parc p
  where not exists (select 1 from mags_op)
)
""".strip()
        parc_params = [code_operation_ref]
        include_mags_op = True

    else:  # "Non participants"
        mags_base_cte = """
mags_base as (
  select p.code_magasin
  from mags_parc p
  where exists (select 1 from mags_op)
    and not exists (select 1 from mags_op o where o.code_magasin = p.code_magasin)
)
""".strip()
        parc_params = [code_operation_ref]
        include_mags_op = True

    ermes_op_cte = """
ermes_op as (
  select distinct upper(trim(e.code_magasin::text)) as code_magasin
  from public.ori_ermes e
  where e.code_operation = %s
    and e.code_magasin is not null
)
""".strip()

    ermes_filter_cte = """
mags_after_ermes as (
  select m.code_magasin
  from mags_base m
  where
    (%s = 'Tous')
    or (%s = 'Participants' and exists (select 1 from ermes_op e where e.code_magasin = m.code_magasin))
    or (%s = 'Non participants' and not exists (select 1 from ermes_op e where e.code_magasin = m.code_magasin))
)
""".strip()

    fid_op_cte = """
fid_op as (
  select distinct upper(trim(k.code_magasin::text)) as code_magasin
  from public.vw_ktb_with_sms_cost k
  where k.code_operation = %s
    and k.code_magasin is not null
)
""".strip()

    fid_filter_cte = """
mags0 as (
  select m.code_magasin
  from mags_after_ermes m
  where
    (%s = 'Tous')
    or (%s = 'Participants' and exists (select 1 from fid_op f where f.code_magasin = m.code_magasin))
    or (%s = 'Non participants' and not exists (select 1 from fid_op f where f.code_magasin = m.code_magasin))
)
""".strip()

    ctes = [periode_op_cte]
    if include_mags_op:
        ctes.append(mags_op_cte)
    ctes += [mags_parc_cte, mags_base_cte, ermes_op_cte, ermes_filter_cte, fid_op_cte, fid_filter_cte]

    sql = "WITH " + ",\n".join(ctes)

    # ✅ IMPORTANT : ordre des params = ordre des %s dans le SQL
    final_params: list = []
    final_params += [code_operation_ref]   # periode_op
    final_params += parc_params            # (optionnel) op_magasin
    final_params += params                 # filtres ref_magasin (dans mags_parc)
    final_params += [code_operation_ref]   # ermes_op
    final_params += [ermes_mode, ermes_mode, ermes_mode]
    final_params += [code_operation_ref]   # fid_op
    final_params += [fid_mode, fid_mode, fid_mode]

    return sql, tuple(final_params)


# =========================
# SQL — compute comparable canonical set (intersection A/B)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def compute_comparable_canons(
    mags_cte_sql_A_base: str, mags_params_A_base: tuple,
    mags_cte_sql_B_base: str, mags_params_B_base: tuple,
) -> list[str]:
    sql = f"""
WITH
magsA AS (
  SELECT DISTINCT code_magasin
  FROM (
    {mags_cte_sql_A_base}
    SELECT DISTINCT code_magasin
    FROM mags0
  ) qA
),
magsB AS (
  SELECT DISTINCT code_magasin
  FROM (
    {mags_cte_sql_B_base}
    SELECT DISTINCT code_magasin
    FROM mags0
  ) qB
),
map AS (
  SELECT ancien_code, code_magasin
  FROM public.vw_param_magasin_ancien_code
),
canonA AS (
  SELECT DISTINCT
    a.code_magasin AS code_src,
    COALESCE(m.code_magasin, a.code_magasin) AS canon
  FROM magsA a
  LEFT JOIN map m ON m.ancien_code = a.code_magasin
),
canonB AS (
  SELECT DISTINCT
    b.code_magasin AS code_src,
    COALESCE(m.code_magasin, b.code_magasin) AS canon
  FROM magsB b
  LEFT JOIN map m ON m.ancien_code = b.code_magasin
),
comparables AS (
  SELECT DISTINCT a.canon
  FROM canonA a
  JOIN canonB b ON b.canon = a.canon
  WHERE left(a.code_src, 2) = left(a.canon, 2)
    AND left(b.code_src, 2) = left(b.canon, 2)
    AND left(a.canon, 2) = left(b.canon, 2)
)
SELECT canon
FROM comparables
ORDER BY canon;
"""
    params = tuple(list(mags_params_A_base) + list(mags_params_B_base))
    df = read_df(sql, params=params)
    if df.empty:
        return []
    return df["canon"].astype(str).tolist()


# =========================
# Apply comparable mode on top of mags0 -> mags
# =========================
def apply_comparable_filter_cte(
    base_sql: str,
    base_params: tuple,
    comp_mode: str,          # "Tous" | "C" | "NC"
    comp_canons: list[str],  # canons (codes magasins communs)
) -> tuple[str, tuple]:
    """
    FIX IMPORTANT :
    - Comparable = C  ➜ mags = EXACTEMENT comp_canons
      ➜ même parc garanti côté A et côté B
    """

    # ✅ Comparable STRICT = intersection figée
    if comp_mode == "C":
        add_sql = """
,
mags as (
  select distinct upper(trim(x::text)) as code_magasin
  from unnest(%s::text[]) as t(x)
)
""".rstrip()

        params = list(base_params) + [comp_canons]
        return base_sql + add_sql, tuple(params)

    # Cas Tous / NC : logique classique depuis mags0
    add_sql = """
,
map as (
  select ancien_code, code_magasin
  from public.vw_param_magasin_ancien_code
),
mags as (
  select distinct
    upper(trim(coalesce(mp.code_magasin, m0.code_magasin)::text)) as code_magasin
  from mags0 m0
  left join map mp
    on mp.ancien_code = m0.code_magasin
  where
    (
      %s = 'Tous'
      OR (
        %s = 'NC'
        AND NOT (
          coalesce(mp.code_magasin, m0.code_magasin) = any(%s)
          AND left(m0.code_magasin, 2) = left(coalesce(mp.code_magasin, m0.code_magasin), 2)
        )
      )
    )
)
""".rstrip()

    params = list(base_params) + [comp_mode, comp_mode, comp_canons]
    return base_sql + add_sql, tuple(params)

# =========================
# RENDER FILTERS (shared)
# =========================
def render_filters() -> dict:
    _init_state()

    ops = load_operations()
    if ops.empty:
        st.error("Aucune opération trouvée dans ref_operations.")
        st.stop()

    refd = load_ref_magasin_distincts()

    # ✅ dernière op (par date_debut desc)
    latest_label = ops["label"].iloc[0]

    # ✅ affichage en ordre alphabétique
    ops_alpha = ops.sort_values("label", ascending=True).reset_index(drop=True)
    labels_ops_alpha = ops_alpha["label"].tolist()

    # ✅ defaults = dernière op ajoutée
    if st.session_state["flt_opA_label"] is None or st.session_state["flt_opA_label"] not in labels_ops_alpha:
        st.session_state["flt_opA_label"] = latest_label
    if st.session_state["flt_opB_label"] is None or st.session_state["flt_opB_label"] not in labels_ops_alpha:
        st.session_state["flt_opB_label"] = latest_label

    PARC_CHOICES = ["Participants", "Tous", "Non participants"]
    COMP_CHOICES = ["Tous", "C", "NC"]

    with st.expander("Filtres", expanded=True):
        with st.form("filters_form", clear_on_submit=False):

            left, right = st.columns(2)

            with left:
                st.selectbox(
                    "Opération A",
                    labels_ops_alpha,
                    index=_safe_index(labels_ops_alpha, st.session_state["flt_opA_label"], 0),
                    key="form_opA",
                )
                st.caption("Parcs — Opération A")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.selectbox("Parc OP", PARC_CHOICES, index=_safe_index(PARC_CHOICES, st.session_state["flt_parc_mode_A"], 0), key="form_parc_mode_A")
                with c2:
                    st.selectbox("ERMES", PARC_CHOICES, index=_safe_index(PARC_CHOICES, st.session_state["flt_ermes_mode_A"], 0), key="form_ermes_mode_A")
                with c3:
                    st.selectbox("FID", PARC_CHOICES, index=_safe_index(PARC_CHOICES, st.session_state["flt_fid_mode_A"], 0), key="form_fid_mode_A")

            with right:
                st.selectbox(
                    "Opération B",
                    labels_ops_alpha,
                    index=_safe_index(labels_ops_alpha, st.session_state["flt_opB_label"], 0),
                    key="form_opB",
                )
                st.caption("Parcs — Opération B")
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.selectbox("Parc OP", PARC_CHOICES, index=_safe_index(PARC_CHOICES, st.session_state["flt_parc_mode_B"], 0), key="form_parc_mode_B")
                with d2:
                    st.selectbox("ERMES", PARC_CHOICES, index=_safe_index(PARC_CHOICES, st.session_state["flt_ermes_mode_B"], 0), key="form_ermes_mode_B")
                with d3:
                    st.selectbox("FID", PARC_CHOICES, index=_safe_index(PARC_CHOICES, st.session_state["flt_fid_mode_B"], 0), key="form_fid_mode_B")

            st.divider()

            magasins_df = refd["magasins"].copy()
            magasins_df["code_magasin"] = magasins_df["code_magasin"].astype(str).str.strip()
            magasins_df["label"] = magasins_df["code_magasin"] + " — " + magasins_df["nom_magasin"].fillna("")
            labels_mag = ["(Tous)"] + magasins_df["label"].tolist()

            default_mag_label = "(Tous)"
            if st.session_state["flt_code_magasin"]:
                m = magasins_df[magasins_df["code_magasin"] == st.session_state["flt_code_magasin"]]
                if not m.empty:
                    default_mag_label = m.iloc[0]["label"]

            mag_col, comp_col = st.columns([2, 1])
            with mag_col:
                st.selectbox(
                    "Magasin (optionnel)",
                    labels_mag,
                    index=_safe_index(labels_mag, default_mag_label, 0),
                    key="form_magasin",
                )
            with comp_col:
                st.selectbox(
                    "Comparable",
                    COMP_CHOICES,
                    index=_safe_index(COMP_CHOICES, st.session_state["flt_comp_mode"], 0),
                    key="form_comp_mode",
                )

            row1_a, row1_b, row1_c = st.columns(3)
            with row1_a:
                reg_elargie = st.multiselect("Région élargie", refd["reg_elargie"], st.session_state["flt_reg_elargie"], key="form_reg_elargie")
            with row1_b:
                reg_admin = st.multiselect("Région admin", refd["reg_admin"], st.session_state["flt_reg_admin"], key="form_reg_admin")
            with row1_c:
                types = st.multiselect("Type magasin", refd["types"], st.session_state["flt_types"], key="form_types")

            row2_a, row2_b, row2_c = st.columns(3)
            with row2_a:
                rcr = st.multiselect("RCR", refd["rcr"], st.session_state["flt_rcr"], key="form_rcr")
            with row2_b:
                seg = st.multiselect("Segmentation", refd["seg"], st.session_state["flt_seg"], key="form_seg")
            with row2_c:
                enseignes = st.multiselect("Enseigne", refd["enseignes"], st.session_state["flt_enseignes"], key="form_enseignes")

            col_apply, col_reset = st.columns([2, 1])
            with col_apply:
                apply = st.form_submit_button("✅ Appliquer les filtres", use_container_width=True)
            with col_reset:
                reset = st.form_submit_button("↩️ Réinitialiser", use_container_width=True)

            if reset:
                st.session_state["flt_opA_label"] = latest_label
                st.session_state["flt_opB_label"] = latest_label

                st.session_state["flt_parc_mode_A"] = "Participants"
                st.session_state["flt_ermes_mode_A"] = "Tous"
                st.session_state["flt_fid_mode_A"] = "Tous"

                st.session_state["flt_parc_mode_B"] = "Participants"
                st.session_state["flt_ermes_mode_B"] = "Tous"
                st.session_state["flt_fid_mode_B"] = "Tous"

                st.session_state["flt_code_magasin"] = None
                st.session_state["flt_comp_mode"] = "Tous"

                st.session_state["flt_types"] = []
                st.session_state["flt_seg"] = []
                st.session_state["flt_rcr"] = []
                st.session_state["flt_reg_elargie"] = []
                st.session_state["flt_reg_admin"] = []
                st.session_state["flt_enseignes"] = []
                st.session_state["filters_applied"] = False

            if apply:
                st.session_state["flt_opA_label"] = st.session_state["form_opA"]
                st.session_state["flt_opB_label"] = st.session_state["form_opB"]

                st.session_state["flt_parc_mode_A"] = st.session_state["form_parc_mode_A"]
                st.session_state["flt_ermes_mode_A"] = st.session_state["form_ermes_mode_A"]
                st.session_state["flt_fid_mode_A"] = st.session_state["form_fid_mode_A"]

                st.session_state["flt_parc_mode_B"] = st.session_state["form_parc_mode_B"]
                st.session_state["flt_ermes_mode_B"] = st.session_state["form_ermes_mode_B"]
                st.session_state["flt_fid_mode_B"] = st.session_state["form_fid_mode_B"]

                if st.session_state["form_magasin"] == "(Tous)":
                    st.session_state["flt_code_magasin"] = None
                else:
                    st.session_state["flt_code_magasin"] = st.session_state["form_magasin"].split(" — ")[0].strip()

                st.session_state["flt_comp_mode"] = st.session_state["form_comp_mode"]

                st.session_state["flt_reg_elargie"] = reg_elargie
                st.session_state["flt_reg_admin"] = reg_admin
                st.session_state["flt_rcr"] = rcr
                st.session_state["flt_seg"] = seg
                st.session_state["flt_types"] = types
                st.session_state["flt_enseignes"] = enseignes

                st.session_state["filters_applied"] = True

    if not st.session_state.get("filters_applied", False):
        st.info("Choisis tes filtres puis clique sur **✅ Appliquer les filtres**.")
        st.stop()

    # ✅ récupérer opA / opB depuis ops (pas ops_alpha) grâce au label
    opA = ops.loc[ops["label"] == st.session_state["flt_opA_label"]].iloc[0]
    opB = ops.loc[ops["label"] == st.session_state["flt_opB_label"]].iloc[0]

    code_opA, lib_opA = opA["code_operation"], opA["libelle_op"]
    code_opB, lib_opB = opB["code_operation"], opB["libelle_op"]

    base_sql_A, base_params_A = build_mags_cte_sql_base(
        code_operation_ref=code_opA,
        parc_mode=st.session_state["flt_parc_mode_A"],
        ermes_mode=st.session_state["flt_ermes_mode_A"],
        fid_mode=st.session_state["flt_fid_mode_A"],
        code_magasin=st.session_state["flt_code_magasin"],
        types=st.session_state["flt_types"],
        seg=st.session_state["flt_seg"],
        rcr=st.session_state["flt_rcr"],
        reg_elargie=st.session_state["flt_reg_elargie"],
        reg_admin=st.session_state["flt_reg_admin"],
        enseignes=st.session_state["flt_enseignes"],
    )

    base_sql_B, base_params_B = build_mags_cte_sql_base(
        code_operation_ref=code_opB,
        parc_mode=st.session_state["flt_parc_mode_B"],
        ermes_mode=st.session_state["flt_ermes_mode_B"],
        fid_mode=st.session_state["flt_fid_mode_B"],
        code_magasin=st.session_state["flt_code_magasin"],
        types=st.session_state["flt_types"],
        seg=st.session_state["flt_seg"],
        rcr=st.session_state["flt_rcr"],
        reg_elargie=st.session_state["flt_reg_elargie"],
        reg_admin=st.session_state["flt_reg_admin"],
        enseignes=st.session_state["flt_enseignes"],
    )

    comp_mode = st.session_state["flt_comp_mode"]

    with st.spinner("Données en cours de chargement… merci de patienter."):
        comp_canons = compute_comparable_canons(base_sql_A, base_params_A, base_sql_B, base_params_B)

    mags_cte_sql_A, mags_cte_params_A = apply_comparable_filter_cte(base_sql_A, base_params_A, comp_mode, comp_canons)
    mags_cte_sql_B, mags_cte_params_B = apply_comparable_filter_cte(base_sql_B, base_params_B, comp_mode, comp_canons)

    return {
        "opA": {"code": code_opA, "lib": lib_opA, "date_debut": opA["date_debut"], "date_fin": opA["date_fin"]},
        "opB": {"code": code_opB, "lib": lib_opB, "date_debut": opB["date_debut"], "date_fin": opB["date_fin"]},

        "comparable": comp_mode,

        "filters": {
            "code_magasin": st.session_state["flt_code_magasin"],
            "comp_mode": comp_mode,
            "types": st.session_state["flt_types"],
            "seg": st.session_state["flt_seg"],
            "rcr": st.session_state["flt_rcr"],
            "reg_elargie": st.session_state["flt_reg_elargie"],
            "reg_admin": st.session_state["flt_reg_admin"],
            "enseignes": st.session_state["flt_enseignes"],
        },

        "mags_cte_sql_A": mags_cte_sql_A,
        "mags_cte_params_A": mags_cte_params_A,
        "mags_cte_sql_B": mags_cte_sql_B,
        "mags_cte_params_B": mags_cte_params_B,

        # rétro compat (A par défaut)
        "mags_cte_sql": mags_cte_sql_A,
        "mags_cte_params": mags_cte_params_A,
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_selected_mags(mags_cte_sql_: str, mags_cte_params_: tuple) -> list[str]:
    """
    Exécute UNE fois le CTE mags et renvoie la liste des codes magasins (canoniques).
    Permet ensuite d'éviter de rejouer le CTE dans toutes les requêtes (timeout).
    """
    sql = f"""
{mags_cte_sql_}
select distinct upper(trim(code_magasin::text)) as code_magasin
from mags
order by 1;
"""
    df = read_df(sql, mags_cte_params_)
    if df.empty:
        return []
    return df["code_magasin"].astype(str).tolist()
