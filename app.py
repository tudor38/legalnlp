import streamlit as st

st.set_page_config(page_title="Word NLP", layout="wide")

doc_stats = st.Page(
    "pages/document_statistics.py",
    title="Document Statistics",
    icon=":material/bar_chart:",
)
document_terms = st.Page(
    "pages/document_terms.py", title="Document Terms", icon=":material/label:"
)
topic_explorer = st.Page(
    "pages/topic_explorer.py", title="Topic Explorer", icon=":material/explore:"
)
multi_doc_search = st.Page(
    "pages/search.py", title="Multi-Doc Search", icon=":material/search:"
)

pg = st.navigation(
    [
        doc_stats,
        document_terms,
        topic_explorer,
        multi_doc_search,
    ]
)
pg.run()
