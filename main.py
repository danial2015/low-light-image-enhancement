import streamlit as st

st.set_page_config(
    page_title="Projek Comvis - Kelompok 4",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("enlighted_gan_view.py", title="EnlightedGAN", icon=":material/home:")
delete_page = st.Page("zero_dce_view.py", title="Zero DCE", icon=":material/delete:")

pg = st.navigation([home, delete_page])
pg.run()