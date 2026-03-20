import streamlit as st

st.markdown("# Test the tiles")

# row = st.container(horizontal=True)
# with row:
#     st.metric("Metric A", 10, 1.5, border=True)
#     st.metric("Metric B", 20, -0.5, border=True)
row1 = st.columns(2)
row2 = st.columns(2)

lst = ["this", "that", "other", "ezzakly"]
for i, col in enumerate(row1 + row2):
    tile = col.container(border=True, height=120)
    tile.metric("label", len(lst[i]))
