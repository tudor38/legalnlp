import streamlit as st

page_1 = st.Page("pages/page_1.py", title="Page 1", icon=":material/add_circle:")
page_2 = st.Page("pages/page_2.py", title="Page 2", icon=":material/add_circle:")

pg = st.navigation([page_1, page_2])

st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()
