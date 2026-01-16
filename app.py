import streamlit as st
from src.auth import is_logged_in, login_form

st.set_page_config(page_title="Dashboard OP", layout="wide")

if is_logged_in():
    st.switch_page("pages/1_Global.py")

login_form()
