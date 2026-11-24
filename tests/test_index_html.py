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


def _pixels(svg: str) -> set[tuple[int, int]]:
    """Return the filled cells of an 8x8 icon drawn with M/H/V/h/v/z paths."""
    cells: set[tuple[int, int]] = set()
    for attrs in re.findall(r"<path([^>]*)/>", svg):
        d = re.search(r'\sd="([^"]+)"', attrs)
        assert d is not None
        polygons: list[list[tuple[float, float]]] = []
        x = y = 0.0
        for cmd, args in re.findall(r"([MHVhvz])([^MHVhvz]*)", d.group(1)):
            nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", args)]
            if cmd == "M":
                x, y = nums
                polygons.append([(x, y)])
            elif cmd in "Hh":
                x = nums[0] if cmd == "H" else x + nums[0]
                polygons[-1].append((x, y))
            elif cmd in "Vv":
                y = nums[0] if cmd == "V" else y + nums[0]
                polygons[-1].append((x, y))
        evenodd = "evenodd" in attrs
        for cx in range(8):
            for cy in range(8):
                px, py = cx + 0.5, cy + 0.5
                hits = 0
                for poly in polygons:
                    inside = False
                    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1], strict=True):
                        if (y1 > py) != (y2 > py) and px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
                            inside = not inside
                    hits += inside
                if (hits % 2 == 1) if evenodd else hits > 0:
                    cells.add((cx, cy))
    return cells


def _icons(page: str) -> dict[str, set[tuple[int, int]]]:
    found = re.findall(r'<svg class="ico( [\w-]+)?"[^>]*>(.*?)</svg>', page, re.S)
    named = {}
    for index, (extra, body) in enumerate(found):
        named[extra.strip() or f"icon {index}"] = _pixels(body)
    return named


def test_every_icon_sits_in_the_middle_of_its_box(page: str) -> None:
    # The ink's centre of mass, so the play triangle counts as optically centred.
    # The one-headed arrow is left out: its head carries the weight by design.
    own_page = re.search(r'id="own-page".*?<svg[^>]*>(.*?)</svg>', page, re.S)
    assert own_page is not None
    off = {}
    for name, cells in _icons(page).items():
        if cells == _pixels(own_page.group(1)):
            continue
        cx = sum(x + 0.5 for x, _ in cells) / len(cells)
        cy = sum(y + 0.5 for _, y in cells) / len(cells)
        if abs(cx - 4) >= 0.5 or abs(cy - 4) >= 0.5:
            off[name] = (round(cx, 2), round(cy, 2))
    assert off == {}


def test_arrow_icons_mirror_onto_themselves_across_their_diagonal(page: str) -> None:
    for button in ("full-toggle", "own-page"):
        svg = re.search(rf'id="{button}".*?<svg[^>]*>(.*?)</svg>', page, re.S)
        assert svg is not None
        cells = _pixels(svg.group(1))
        assert {(7 - y, 7 - x) for x, y in cells} == cells, button
