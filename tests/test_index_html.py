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
    for element_id in ("map", "seed", "size", "new-map", "speed", "pause", "status"):
        assert f'id="{element_id}"' in page, element_id
