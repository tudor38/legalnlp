"""
Config loader for the legalnlp app.

The TOML config lives at the project root: config/app.toml
It is loaded once at import time and exposed as CFG (an AppConfig instance).

Any missing or mistyped key raises ConfigError at startup — before the first
page renders — so configuration problems are never silent.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "app.toml"


class ConfigError(ValueError):
    """Raised when app.toml is missing a required key or has a wrong type."""


# ---------------------------------------------------------------------------
# Typed config schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DisplayConfig:
    date_format: str
    para_truncate_len: int
    allowed_filetypes: list[str]
    closed_date_offset_days: int


@dataclass(frozen=True)
class ChartConfig:
    card_height: int
    marker_size: int
    marker_opacity: float
    bar_height_per_row: int
    bar_height_base: int
    timeline_height_per_author: int
    timeline_height_base: int


@dataclass(frozen=True)
class TimeBinConfig:
    day_max_days: int
    week_max_days: int


@dataclass(frozen=True)
class Page1Tabs:
    main: list[str]
    comment_views: list[str]
    redline_views: list[str]
    move_views: list[str]


@dataclass(frozen=True)
class AppConfig:
    display: DisplayConfig
    chart: ChartConfig
    time_bin: TimeBinConfig
    page_1_tabs: Page1Tabs


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _section(raw: dict, *path: str) -> dict:
    """Walk nested dicts, raising ConfigError on a missing key."""
    node = raw
    for key in path:
        if key not in node:
            raise ConfigError(
                f"app.toml is missing required section [{'.'.join(path[: path.index(key) + 1])}]"
            )
        node = node[key]
    return node


def _field(section: dict, key: str, section_name: str, expected_type: type):
    if key not in section:
        raise ConfigError(f"app.toml [{section_name}] is missing required key '{key}'")
    val = section[key]
    if not isinstance(val, expected_type):
        raise ConfigError(
            f"app.toml [{section_name}] '{key}' must be {expected_type.__name__}, "
            f"got {type(val).__name__}"
        )
    return val


def _parse(raw: dict) -> AppConfig:
    try:
        display_raw = _section(raw, "display")
        display = DisplayConfig(
            date_format=_field(display_raw, "date_format", "display", str),
            para_truncate_len=_field(display_raw, "para_truncate_len", "display", int),
            allowed_filetypes=_field(display_raw, "allowed_filetypes", "display", list),
            closed_date_offset_days=_field(
                display_raw, "closed_date_offset_days", "display", int
            ),
        )

        chart_raw = _section(raw, "chart")
        chart = ChartConfig(
            card_height=_field(chart_raw, "card_height", "chart", int),
            marker_size=_field(chart_raw, "marker_size", "chart", int),
            marker_opacity=_field(chart_raw, "marker_opacity", "chart", float),
            bar_height_per_row=_field(chart_raw, "bar_height_per_row", "chart", int),
            bar_height_base=_field(chart_raw, "bar_height_base", "chart", int),
            timeline_height_per_author=_field(
                chart_raw, "timeline_height_per_author", "chart", int
            ),
            timeline_height_base=_field(
                chart_raw, "timeline_height_base", "chart", int
            ),
        )

        tb_raw = _section(raw, "time_bin")
        time_bin = TimeBinConfig(
            day_max_days=_field(tb_raw, "day_max_days", "time_bin", int),
            week_max_days=_field(tb_raw, "week_max_days", "time_bin", int),
        )

        tabs_raw = _section(raw, "pages", "page_1", "tabs")
        page_1_tabs = Page1Tabs(
            main=_field(tabs_raw, "main", "pages.page_1.tabs", list),
            comment_views=_field(tabs_raw, "comment_views", "pages.page_1.tabs", list),
            redline_views=_field(tabs_raw, "redline_views", "pages.page_1.tabs", list),
            move_views=_field(tabs_raw, "move_views", "pages.page_1.tabs", list),
        )

        return AppConfig(
            display=display,
            chart=chart,
            time_bin=time_bin,
            page_1_tabs=page_1_tabs,
        )
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Unexpected error parsing app.toml: {exc}") from exc


def _load() -> AppConfig:
    try:
        with open(_CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"Config file not found: {_CONFIG_PATH}. "
            "Make sure config/app.toml exists at the project root."
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"app.toml has a syntax error: {exc}") from exc
    return _parse(raw)


CFG: AppConfig = _load()
