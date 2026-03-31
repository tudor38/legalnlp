import pandas as pd
import streamlit as st
from _pytest.monkeypatch import MonkeyPatch
from datetime import date

from src.stats.ui import sidebar_controls
from src.app_state import KEY_FILTER_AUTHORS, KEY_FILTER_DATE_MAX, KEY_FILTER_DATE_MIN


def test_sidebar_controls_non_empty(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(st.sidebar, "markdown", lambda text: None)
    monkeypatch.setattr(st.sidebar, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        st.sidebar,
        "slider",
        lambda *args, **kwargs: (date(2024, 1, 1), date(2024, 1, 1)),
    )
    monkeypatch.setattr(
        st.sidebar,
        "multiselect",
        lambda *args, **kwargs: [],
    )

    st.session_state.clear()
    st.session_state[KEY_FILTER_DATE_MIN] = None
    st.session_state[KEY_FILTER_DATE_MAX] = None
    st.session_state[KEY_FILTER_AUTHORS] = []

    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "author": ["Alice"],
        }
    )

    reference_date, is_closed, date_range, selected_authors = sidebar_controls(
        comments=[],
        redlines=[],
        all_dfs=[df],
        store_is_closed=lambda: None,
        store_closed_date=lambda: None,
        store_date_range=lambda: None,
        store_timeline_authors=lambda: None,
    )

    assert not is_closed
    assert date_range == (date(2024, 1, 1), date(2024, 1, 1))
    assert selected_authors == ["Alice"]
    assert reference_date.date() == date.today()


def test_sidebar_controls_closed_matter(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(st.sidebar, "markdown", lambda text: None)
    monkeypatch.setattr(st.sidebar, "toggle", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        st.sidebar,
        "date_input",
        lambda *args, **kwargs: date(2024, 1, 15),
    )
    monkeypatch.setattr(
        st.sidebar,
        "slider",
        lambda *args, **kwargs: (date(2024, 1, 1), date(2024, 1, 1)),
    )
    monkeypatch.setattr(
        st.sidebar,
        "multiselect",
        lambda *args, **kwargs: [],
    )

    st.session_state.clear()
    st.session_state[KEY_FILTER_DATE_MIN] = None
    st.session_state[KEY_FILTER_DATE_MAX] = None
    st.session_state[KEY_FILTER_AUTHORS] = []

    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "author": ["Alice"],
        }
    )

    reference_date, is_closed, date_range, selected_authors = sidebar_controls(
        comments=[],
        redlines=[],
        all_dfs=[df],
        store_is_closed=lambda: None,
        store_closed_date=lambda: None,
        store_date_range=lambda: None,
        store_timeline_authors=lambda: None,
    )

    assert is_closed
    assert date_range == (date(2024, 1, 1), date(2024, 1, 1))
    assert selected_authors == ["Alice"]
    assert reference_date.date() == date(2024, 1, 16)  # closed date + 1 day offset
