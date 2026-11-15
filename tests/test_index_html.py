"""Structure tests for the click-to-go page. The behaviour is checked by hand in a browser."""

from __future__ import annotations

import re
from importlib.resources import files

import pytest


@pytest.fixture(scope="module")
def page() -> str:
    return (files("worm") / "static" / "index.html").read_text(encoding="utf-8")


def test_index_html_has_no_external_urls(page: str) -> None:
    urls = re.findall(r"""(?:src|href|action)\s*=\s*["']([^"']+)""", page)
    assert [u for u in urls if re.match(r"(?:https?:)?//", u)] == []
    assert "http://" not in page and "https://" not in page


def test_index_html_references_every_api_endpoint(page: str) -> None:
    for endpoint in (
        "/api/state",
        "/api/map.png",
        "/api/map.json",
        "/api/map",
        "/api/goal",
        "/api/control",
        "/api/frame.jpg",
    ):
        assert endpoint in page, endpoint


def test_index_html_has_a_map_canvas_and_the_controls(page: str) -> None:
    for element_id in ("map", "seed", "size", "new-map", "pause", "state-text"):
        assert f'id="{element_id}"' in page, element_id


NEW_IDS = (
    "map",
    "verdict",
    "state-text",
    "state-surface",
    "state-replans",
    "state-time",
    "legend",
    "cam",
    "cam-toggle",
    "pause",
    "drawer",
    "drawer-toggle",
    "drawer-close",
    "seed",
    "size",
    "new-map",
    "map-error",
)


def test_index_html_has_the_click_anchored_layout(page: str) -> None:
    for element_id in NEW_IDS:
        assert f'id="{element_id}"' in page, element_id


def test_index_html_has_speed_presets_in_place_of_a_slider(page: str) -> None:
    speeds = re.findall(r'data-speed="(\d+)"', page)
    assert speeds == ["1", "2", "4", "8", "16"]
    assert 'id="speed"' not in page


def test_index_html_loads_only_its_own_fonts(page: str) -> None:
    faces = re.findall(r"url\(([^)]+)\)", page)
    assert faces, "the page declares no font files"
    assert all(re.fullmatch(r"""["']?fonts/[a-z0-9-]+\.woff2["']?""", f) for f in faces)


def test_index_html_draws_its_icons_instead_of_using_glyphs(page: str) -> None:
    glyphs = re.findall(r"[\u2190-\u2BFF\U0001F300-\U0001FAFF]", page)
    assert glyphs == []
    assert page.count("<svg") >= 5


def test_index_html_names_every_button_and_labels_every_field(page: str) -> None:
    for match in re.finditer(r"<button\b([^>]*)>(.*?)</button>", page, flags=re.S):
        attrs, inner = match.groups()
        text = re.sub(r"<[^>]+>", "", inner).strip()
        assert "aria-label=" in attrs or text, match.group(0)[:80]
    for field in re.finditer(r"<(?:input|select)\b([^>]*)>", page):
        assert "id=" in field.group(1)
    assert page.count("<label") >= page.count("<input") + page.count("<select") - 1


def test_index_html_themes_the_browser_parts(page: str) -> None:
    for needle in ("::selection", "scrollbar", ":focus-visible", "tabular-nums", "accent-color"):
        assert needle in page, needle
    assert "prefers-reduced-motion" in page


def test_the_new_map_request_uses_the_keys_the_server_reads(page: str) -> None:
    """The server reads `slope`, `rocks` and `ditches`. Other names are silently ignored."""
    body = page[page.index("api.newMap({") :][:400]
    for key in ("seed", "size", "mix", "slope", "rocks", "ditches"):
        assert re.search(rf"\b{key}\s*:", body), key
    for wrong in ("max_slope_deg", "n_rocks", "n_ditches"):
        assert wrong not in body, wrong


def test_index_html_has_a_loading_and_error_panel(page: str) -> None:
    for element_id in ("boot", "boot-title", "boot-detail", "boot-retry"):
        assert f'id="{element_id}"' in page, element_id
    assert 'id="boot" role="status"' in page


def test_index_html_can_take_another_data_source(page: str) -> None:
    """The browser build sets `window.wormApi` before this script runs."""
    assert "window.wormApi" in page
    assert '<script type="module">' in page
    assert "await api.ready" in page


def test_index_html_lets_the_data_source_draw_the_follow_cam(page: str) -> None:
    """The server sends JPEG frames. The browser build draws its own 3D view."""
    assert "api.camera.mount" in page and "api.camera.unmount" in page
    assert "no connection to the server" not in page.lower() or "api.boot" in page


def test_index_html_stops_polling_when_the_data_source_failed_to_start(page: str) -> None:
    assert 'api.boot.state === "error"' in page
