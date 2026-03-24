import streamlit as st

doc_stats = st.Page("pages/document_statistics.py", title="Document Statistics", icon=":material/description:")
topic_exploration = st.Page("pages/topic_exploration.py", title="Topic Exploration", icon=":material/explore:")
position_check = st.Page("pages/position_check.py", title="Position Check", icon=":material/gavel:")

pg = st.navigation([doc_stats, topic_exploration, position_check])

st.set_page_config(page_title="Legal NLP", page_icon=":material/gavel:")
pg.run()
