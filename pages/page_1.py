import streamlit as st
from src.package.extract_comments import extract_comments
from src.package.render_comments import render_comments


comments = extract_comments("/home/tudor/Downloads/documents/policy.docx")

ALL_ELEMENTS = ["Comment", "Paragraph", "Sentence"]
order = st.sidebar.multiselect(
    "Elements to show", options=ALL_ELEMENTS, default=ALL_ELEMENTS
)
render_comments(*comments, order=order)
