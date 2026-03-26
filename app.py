from PIL import Image, ImageDraw
import streamlit as st


def _diamond_icon() -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 48

    def p(x, y):
        return (round(x * s), round(y * s))

    # Large diamond
    d.polygon(
        [p(24, 6), p(38, 24), p(24, 42), p(10, 24)],
        outline=(232, 121, 249, 255),
        width=2,
    )
    # Medium offset diamond
    d.polygon(
        [p(28, 10), p(38, 20), p(28, 30), p(18, 20)],
        outline=(129, 140, 248, 160),
        width=1,
    )
    # Small inner diamond
    d.polygon(
        [p(24, 16), p(31, 24), p(24, 32), p(17, 24)],
        fill=(56, 189, 248, 60),
        outline=(129, 140, 248, 220),
        width=1,
    )
    return img


st.set_page_config(page_title="Legal NLP", page_icon=_diamond_icon(), layout="wide")
st.logo("assets/logo.svg")

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
search = st.Page("pages/search.py", title="Search", icon=":material/search:")

pg = st.navigation(
    [
        doc_stats,
        document_terms,
        topic_explorer,
        search,
    ]
)
pg.run()
