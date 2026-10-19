"""An upstream outage is a 502, an unset key is a 400. Both used to be 400, so a
caller's retry on 5xx never fired for a fal outage. No network: httpx.post is patched."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from bricks_infra import image as image_mod
from bricks_infra.config import get_settings


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("BRICKS_INFRA_FAL_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_three_transport_failures_raise_upstream_error(with_key):
    with patch("bricks_infra.image.httpx.post", side_effect=httpx.ConnectError("boom")) as post, \
         patch("bricks_infra.image.time.sleep") as sleep:
        with pytest.raises(image_mod.ImageUpstreamError):
            image_mod.generate_image("a hero image")
    assert post.call_count == 3 and sleep.call_count == 2


def test_unset_key_is_a_plain_runtime_error(monkeypatch):
    monkeypatch.setenv("BRICKS_INFRA_FAL_API_KEY", "")
    get_settings.cache_clear()
    with patch("bricks_infra.image.httpx.post") as post:
        with pytest.raises(RuntimeError) as e:
            image_mod.generate_image("x")
    assert not isinstance(e.value, image_mod.ImageUpstreamError) and post.call_count == 0
    get_settings.cache_clear()


def test_app_maps_upstream_to_502_and_unset_key_to_400(with_key):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bricks_infra.app import app
    client = TestClient(app)
    with patch("bricks_infra.image.httpx.post", side_effect=httpx.ConnectError("boom")), \
         patch("bricks_infra.image.time.sleep"):
        r = client.post("/image", json={"prompt": "x"})
    assert r.status_code == 502, r.text
    with patch("bricks_infra.image.get_settings") as gs:
        gs.return_value = get_settings().model_copy(update={"fal_api_key": ""})
        r = client.post("/image", json={"prompt": "x"})
    assert r.status_code == 400, r.text
