# src/filters.py
import streamlit as st
import pandas as pd
from src.db import read_df


# =========================
# DATA LOADERS
# =========================
@st.cache_data(ttl=600)
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


@st.cache_data(ttl=600)
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
    }


# =========================
# FILTER STATE
# =========================
def _init_state():
    ss = st.session_state
    ss.setdefault("flt_opA_label", None)
    ss.setdefault("flt_opB_label", None)

    ss.setdefault("flt_parc_mode", "Participants")  # Participants | Tous | Non participants

    ss.setdefault("flt_code_magasin", None)  # magasin unique
    ss.setdefault("flt_types", [])
    ss.setdefault("flt_seg", [])
    ss.setdefault("flt_rcr", [])
    ss.setdefault("flt_reg_elargie", [])
    ss.setdefault("flt_reg_admin", [])

    ss.setdefault("filters_applied", False)


# =========================
# BUILD CTE "mags" (RETURNS FULL "WITH ...")
# =========================
def build_mags_cte_sql(
    code_op_for_parc: str,
    parc_mode: str,
    code_magasin: str | None,
    types: list[str],
    seg: list[str],
    rcr: list[str],
    reg_elargie: list[str],
    reg_admin: list[str],
) -> tuple[str, tuple]:
    """
    Retourne:
      - un SQL qui commence par "WITH ..." et définit au minimum:
          mags_parc AS (...),
          mags AS (...)
        et parfois:
          mags_op AS (...),
          mags_parc AS (...),
          mags AS (...)
      - params correspondant aux %s du SQL
    """

    where_parts = ["rm.code_magasin is not null"]
    params: list = []

    # Magasin unique
    if code_magasin:
        where_parts.append("trim(rm.code_magasin::text) = %s")
        params.append(code_magasin)

    # Multi filtres (ANY)
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

    where_sql = " AND ".join(where_parts)

    mags_parc_cte = f"""
mags_parc as (
  select distinct trim(rm.code_magasin::text) as code_magasin
  from public.ref_magasin rm
  where {where_sql}
)
""".strip()

    mags_op_cte = """
mags_op as (
  select distinct trim(code_magasin::text) as code_magasin
  from public.op_magasin
  where code_operation = %s
)
""".strip()

    if parc_mode == "Tous":
        mags_cte = """
mags as (
  select code_magasin from mags_parc
)
""".strip()

        sql = "WITH " + ",\n".join([mags_parc_cte, mags_cte])
        return sql, tuple(params)

    if parc_mode == "Participants":
        mags_cte = """
mags as (
  select p.code_magasin
  from mags_parc p
  join mags_op o using(code_magasin)

  union all

  select p.code_magasin
  from mags_parc p
  where not exists (select 1 from mags_op)
)
""".strip()

        sql = "WITH " + ",\n".join([mags_op_cte, mags_parc_cte, mags_cte])
        return sql, tuple([code_op_for_parc] + params)

    # Non participants
    mags_cte = """
mags as (
  select p.code_magasin
  from mags_parc p
  where exists (select 1 from mags_op)
    and not exists (select 1 from mags_op o where o.code_magasin = p.code_magasin)
)
""".strip()

    sql = "WITH " + ",\n".join([mags_op_cte, mags_parc_cte, mags_cte])
    return sql, tuple([code_op_for_parc] + params)


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

    # Defaults opA/opB (avant "appliquer")
    if st.session_state["flt_opA_label"] is None:
        st.session_state["flt_opA_label"] = ops["label"].iloc[0]
    if st.session_state["flt_opB_label"] is None:
        st.session_state["flt_opB_label"] = ops["label"].iloc[1] if len(ops) > 1 else ops["label"].iloc[0]

    with st.expander("Filtres", expanded=True):
        with st.form("filters_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                st.selectbox(
                    "Opération A",
                    ops["label"].tolist(),
                    index=ops["label"].tolist().index(st.session_state["flt_opA_label"]),
                    key="form_opA",
                )
            with c2:
                st.selectbox(
                    "Opération B",
                    ops["label"].tolist(),
                    index=ops["label"].tolist().index(st.session_state["flt_opB_label"]),
                    key="form_opB",
                )

            st.radio(
                "Parc magasins",
                ["Participants", "Tous", "Non participants"],
                horizontal=True,
                index=["Participants", "Tous", "Non participants"].index(st.session_state["flt_parc_mode"]),
                key="form_parc_mode",
            )

            st.divider()

            magasins_df = refd["magasins"].copy()
            magasins_df["label"] = magasins_df["code_magasin"] + " — " + magasins_df["nom_magasin"].fillna("")
            labels_mag = ["(Tous)"] + magasins_df["label"].tolist()

            default_mag_label = "(Tous)"
            if st.session_state["flt_code_magasin"]:
                m = magasins_df[magasins_df["code_magasin"] == st.session_state["flt_code_magasin"]]
                if not m.empty:
                    default_mag_label = m.iloc[0]["label"]

            st.selectbox(
                "Magasin (optionnel)",
                labels_mag,
                index=labels_mag.index(default_mag_label),
                key="form_magasin",
            )

            a, b, c = st.columns(3)
            with a:
                reg_elargie = st.multiselect(
                    "Région élargie",
                    refd["reg_elargie"],
                    st.session_state["flt_reg_elargie"],
                    key="form_reg_elargie",
                )
                rcr = st.multiselect(
                    "RCR",
                    refd["rcr"],
                    st.session_state["flt_rcr"],
                    key="form_rcr",
                )
            with b:
                reg_admin = st.multiselect(
                    "Région admin",
                    refd["reg_admin"],
                    st.session_state["flt_reg_admin"],
                    key="form_reg_admin",
                )
                seg = st.multiselect(
                    "Segmentation",
                    refd["seg"],
                    st.session_state["flt_seg"],
                    key="form_seg",
                )
            with c:
                types = st.multiselect(
                    "Type magasin",
                    refd["types"],
                    st.session_state["flt_types"],
                    key="form_types",
                )

            col_apply, col_reset = st.columns([2, 1])
            with col_apply:
                apply = st.form_submit_button("✅ Appliquer les filtres", use_container_width=True)
            with col_reset:
                reset = st.form_submit_button("↩️ Réinitialiser", use_container_width=True)

            if reset:
                st.session_state["flt_opA_label"] = ops["label"].iloc[0]
                st.session_state["flt_opB_label"] = ops["label"].iloc[1] if len(ops) > 1 else ops["label"].iloc[0]
                st.session_state["flt_parc_mode"] = "Participants"
                st.session_state["flt_code_magasin"] = None
                st.session_state["flt_types"] = []
                st.session_state["flt_seg"] = []
                st.session_state["flt_rcr"] = []
                st.session_state["flt_reg_elargie"] = []
                st.session_state["flt_reg_admin"] = []
                st.session_state["filters_applied"] = False

            if apply:
                st.session_state["flt_opA_label"] = st.session_state["form_opA"]
                st.session_state["flt_opB_label"] = st.session_state["form_opB"]
                st.session_state["flt_parc_mode"] = st.session_state["form_parc_mode"]

                if st.session_state["form_magasin"] == "(Tous)":
                    st.session_state["flt_code_magasin"] = None
                else:
                    st.session_state["flt_code_magasin"] = st.session_state["form_magasin"].split(" — ")[0].strip()

                st.session_state["flt_reg_elargie"] = reg_elargie
                st.session_state["flt_reg_admin"] = reg_admin
                st.session_state["flt_rcr"] = rcr
                st.session_state["flt_seg"] = seg
                st.session_state["flt_types"] = types

                st.session_state["filters_applied"] = True

    if not st.session_state.get("filters_applied", False):
        st.info("Choisis tes filtres puis clique sur **✅ Appliquer les filtres**.")
        st.stop()

    opA = ops.loc[ops["label"] == st.session_state["flt_opA_label"]].iloc[0]
    opB = ops.loc[ops["label"] == st.session_state["flt_opB_label"]].iloc[0]

    code_opA, lib_opA = opA["code_operation"], opA["libelle_op"]
    code_opB, lib_opB = opB["code_operation"], opB["libelle_op"]

    # ✅ CTE mags basé sur OP A (référence pour Participants/Non participants)
    mags_cte_sql, mags_cte_params = build_mags_cte_sql(
        code_op_for_parc=code_opA,
        parc_mode=st.session_state["flt_parc_mode"],
        code_magasin=st.session_state["flt_code_magasin"],
        types=st.session_state["flt_types"],
        seg=st.session_state["flt_seg"],
        rcr=st.session_state["flt_rcr"],
        reg_elargie=st.session_state["flt_reg_elargie"],
        reg_admin=st.session_state["flt_reg_admin"],
    )

    return {
        "opA": {"code": code_opA, "lib": lib_opA, "date_debut": opA["date_debut"], "date_fin": opA["date_fin"]},
        "opB": {"code": code_opB, "lib": lib_opB, "date_debut": opB["date_debut"], "date_fin": opB["date_fin"]},
        "parc_mode": st.session_state["flt_parc_mode"],
        "filters": {
            "code_magasin": st.session_state["flt_code_magasin"],
            "types": st.session_state["flt_types"],
            "seg": st.session_state["flt_seg"],
            "rcr": st.session_state["flt_rcr"],
            "reg_elargie": st.session_state["flt_reg_elargie"],
            "reg_admin": st.session_state["flt_reg_admin"],
        },
        "mags_cte_sql": mags_cte_sql,         # ✅ commence par WITH
        "mags_cte_params": mags_cte_params,   # ✅ params correspondant
    }
