from fastapi.testclient import TestClient

import automata.server as srv
from automata.hub import WORLDS, render_cards, render_featured, world_by_key


async def _unreachable(client, url):
    return None


def test_every_world_renders_in_hero_or_grid():
    grid, hero = render_cards(), render_featured()
    page = grid + hero
    for w in WORLDS:
        assert f'id="st-{w.key}"' in page  # status badge present (hero or grid)
        assert w.url in page
        # featured worlds live in the hero, the rest in the grid — never both
        assert (f'id="st-{w.key}"' in hero) == bool(w.featured), w.key
    assert "wt-thumb" in grid and "wt-live" in grid  # watchtower live-chart overlay
    assert render_featured(), "expected at least one featured world"


def test_grid_excludes_featured_worlds():
    grid = render_cards()
    for w in WORLDS:
        if w.featured:
            assert f'id="st-{w.key}"' not in grid, w.key


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
