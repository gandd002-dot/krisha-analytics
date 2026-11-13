"""UI regression checks with a fake DB; never uses production credentials."""
from pathlib import Path
from unittest.mock import MagicMock
import json
import sys

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_helpers import get_photo_urls, photo_html, safe_url

APP = Path(__file__).resolve().parents[1] / "app_v3.py"


@pytest.fixture
def app(monkeypatch):
    rows = []
    for i in range(36):
        rows.append(dict(
            id=i+1, title=f"Квартира {i+1}", price=100000+i*10000, area=40+i,
            rooms=1+i%3, district=["Бостандыкский р-н", "Ауэзовский р-н"][i%2],
            url=f"https://krisha.kz/a/show/{i+1}",
            photos=json.dumps([f"https://example.com/{i+1}/{n}-800x600.jpg" for n in range(7)]),
            predicted_class="BAD" if i%5==0 else "GOOD", predicted_confidence=.9,
        ))
    monkeypatch.setattr("psycopg2.connect", lambda **kwargs: MagicMock())
    monkeypatch.setattr(pd, "read_sql", lambda *args, **kwargs: pd.DataFrame(rows))
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=15)
    at.secrets.update(DB_HOST="test", DB_PORT="5432", DB_NAME="test", DB_USER="test", DB_PASSWORD="not-a-real-secret")
    at.run()
    assert not at.exception
    yield at
    st.cache_data.clear()


def test_gallery_selection_collapse_and_favorites(app):
    first = app.session_state["selected_listing"]
    app.button(key="photo_next").click().run()
    assert app.session_state["photo_index"] == 1
    app.button(key="next").click().run()
    assert app.session_state["selected_listing"] != first
    assert app.session_state["photo_index"] == 0
    selected = app.session_state["selected_listing"]
    app.button(key="hero_favorite").click().run()
    assert selected in app.session_state["favorites"]
    app.button(key="toggle_preview").click().run()
    assert not app.session_state["preview_open"]
    assert not any(b.key == "photo_next" for b in app.button)
    app.button(key=f"open_{first}").click().run()
    assert app.session_state["preview_open"]
    assert app.session_state["selected_listing"] == first
    app.button(key="favorites_view").click().run()
    assert app.session_state["selected_listing"] == selected
    assert not app.exception
    app.button(key="hero_favorite").click().run()
    assert app.info
    assert not app.exception


def test_filters_sort_and_pagination(app):
    app.button(key="show_more").click().run()
    assert app.session_state["display_count"] == 24
    app.selectbox(key="sorting").set_value("Дешевле").run()
    assert app.session_state["display_count"] == 12
    app.multiselect(key="rooms").set_value([3]).run()
    app.multiselect(key="districts").set_value(["Ауэзовский р-н"]).run()
    assert not app.exception
    # Empty choices retain the original 'all options' semantics.
    app.multiselect(key="rooms").set_value([]).run()
    app.slider(key="confidence").set_value(1.0).run()
    assert app.info
    assert not app.exception


def test_gallery_wrap_and_thumbnails(app):
    app.button(key="photo_previous").click().run()
    assert app.session_state["photo_index"] == 6
    app.button(key="thumb_5").click().run()
    assert app.session_state["photo_index"] == 5
    app.button(key="photo_next").click().run()
    app.button(key="photo_next").click().run()
    assert app.session_state["photo_index"] == 0
    assert not app.exception


def test_photo_inputs_are_safe_and_deduplicated():
    photos = ["https://example.com/123-800x600.jpg", "https://example.com/123-400x300.jpg",
              "https://example.com/124-800x600.jpg?x=1", "javascript:alert(1)", None]
    expected = [photos[0], photos[2]]
    assert get_photo_urls(photos, limit=None) == expected
    assert get_photo_urls(json.dumps(photos), limit=None) == expected
    assert get_photo_urls(photos, limit=1) == expected[:1]
    for value in [None, "broken", "{}", {"url": "https://example.com"}]:
        assert get_photo_urls(value) == []
    assert safe_url("javascript:alert(1)") == ""
    assert 'src=' not in photo_html("javascript:alert(1)")
    assert '&lt;script&gt;' in photo_html("https://example.com/photo.jpg", alt="<script>")


def test_luxury_keeps_original_repair_default(app):
    app.toggle(key="luxury").set_value(True).run()
    assert app.checkbox(key="only_good").value
    app.toggle(key="luxury").set_value(False).run()
    assert not app.checkbox(key="only_good").value
    assert not app.exception
