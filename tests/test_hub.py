from fastapi.testclient import TestClient

import automata.server as srv
from automata.hub import WORLDS, render_cards, world_by_key


async def _unreachable(client, url):
    return None


def test_render_cards_covers_every_world():
    cards = render_cards()
    assert "<!--WORLDS-->" not in cards
    for w in WORLDS:
        assert f'id="st-{w.key}"' in cards
        assert w.url in cards
    assert "wt-thumb" in cards and "wt-live" in cards  # watchtower live-chart overlay
    for w in WORLDS:  # each card shows either its thumbnail image or its glyph fallback
        assert (w.thumb and w.thumb in cards) or (w.glyph and w.glyph in cards), w.key


def test_hub_status_keys_match_registry(monkeypatch):
    monkeypatch.setattr(srv, "_fetch_json", _unreachable)
    c = TestClient(srv.app)
    d = c.get("/hub/status.json").json()
    assert set(d) == {w.key for w in WORLDS}
    assert d["automata"]["up"] is True  # local world, always up
    assert d["phalanx"]["up"] is False  # remote, unreachable here
    assert d["pitch"]["paused"] is False  # reports_paused = False


def test_ui_stats_ordered_by_num_and_exposes_flags(monkeypatch):
    monkeypatch.setattr(srv, "_fetch_json", _unreachable)
    c = TestClient(srv.app)
    js = c.get("/ui/stats.json").json()
    keys = [w["key"] for w in js["worlds"]]
    assert keys == sorted(keys, key=lambda k: world_by_key(k).num)
    assert keys[0] == "automata"
    for w in js["worlds"]:
        assert "pausable" in w and "resettable" in w
    assert {"total_spend", "today_spend", "spend_series", "ts"} <= set(js)


def test_pause_reset_guards_come_from_registry():
    c = TestClient(srv.app)
    assert c.post("/ui/pause?world=pitch&paused=1").status_code == 400  # non-pausable
    assert c.post("/ui/pause?world=ghost").status_code == 400  # unknown
    assert c.post("/ui/reset?world=ghost").status_code == 400  # unknown


def test_hub_served_with_cards_injected(monkeypatch):
    c = TestClient(srv.app)
    r = c.get("/", headers={"host": "worlds.artel.run"})
    assert r.status_code == 200
    assert "<!--WORLDS-->" not in r.text
    assert 'id="st-verglas"' in r.text and 'id="st-watchtower"' in r.text
