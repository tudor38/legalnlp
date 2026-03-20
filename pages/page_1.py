import streamlit as st
from src.comments.extract import extract_comments
from src.comments.render import render_comments, render_thread_depth
from src.stats.compute import resolution_rate
from src.stats.render import render_open_comment_ages

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["docx", "doc"])

options = [
    "Overview",
    "Comments",
]
selection = st.pills("Show", options, selection_mode="single")

if uploaded_file:
    comments, version = extract_comments(uploaded_file)
    match selection:
        case "Overview":
            rate = resolution_rate(comments)

            st.progress(rate, text=f"Resolved: {rate:.0%}")
            render_thread_depth(comments)
            render_open_comment_ages(comments)
        case "Comments":
            ALL_ELEMENTS = ["Sentence", "Comment", "Paragraph"]
            order = st.sidebar.multiselect(
                "Elements to show", options=ALL_ELEMENTS, default=ALL_ELEMENTS[:-1]
            )
            render_comments(comments, version, order=order)
