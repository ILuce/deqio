"""B13: the Watch page's JavaScript, executed for real under Node with a DOM stub.

The script is run exactly as the browser would (`<script>` content), with
`document`, `fetch` and `setInterval` replaced by recording stubs. Earlier
tests only pinned source strings; these check behavior: auto-refresh keeps the
pages loaded with "Load older", a slow refresh is not duplicated by the timer,
and no handler leaves an unhandled rejection.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from deqio.ui import DASHBOARD, WATCH_DASHBOARD

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required to execute the dashboard scripts")


def _scripts(page: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", page, re.S)


@pytest.mark.parametrize("page", [DASHBOARD, WATCH_DASHBOARD], ids=["dashboard", "watch"])
def test_dashboard_scripts_parse(tmp_path: Path, page: str) -> None:
    for index, script in enumerate(_scripts(page)):
        path = tmp_path / f"script{index}.js"
        path.write_text(script, encoding="utf-8")
        check = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr


HARNESS = r'''
const fs = require("fs");
const vm = require("vm");
const script = fs.readFileSync(process.argv[2], "utf8");

const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      id, textContent: "", innerHTML: "", value: "*", checked: true, disabled: false, title: "", style: {}, dataset: {},
      addEventListener() {}, querySelectorAll() { return []; }, showModal() {}, close() {},
      replaceChildren() {}, appendChild() {}, options: [],
    });
  }
  return elements.get(id);
}
const createElement = (tag) => ({ tag, textContent: "", value: "", selected: false, appendChild() {} });

const state = { total: 1200, failMode: null, deferWatch: false, pending: [], log: [] };
const json = (status, body) => ({ ok: status < 400, status, json: async () => body });
function page(limit, offset) {
  const rows = [];
  for (let i = offset; i < Math.min(state.total, offset + limit); i += 1) {
    rows.push({ event_id: `e${i}`, timestamp: "2026-10-09T00:00:00Z", endpoint: "/v1/noul", status_code: 200, source: "api", model_id: "m", backend: "b" });
  }
  return { events: rows, session: {}, pagination: { total: state.total, limit, offset, returned: rows.length, has_more: offset + rows.length < state.total } };
}
async function fetchStub(url) {
  state.log.push(url);
  if (state.failMode === "throw") throw new Error("network down");
  if (url.startsWith("/v1/watch?")) {
    if (state.failMode === "watch") return json(500, { detail: "boom" });
    const params = new URLSearchParams(url.split("?")[1]);
    const body = page(Number(params.get("limit")), Number(params.get("offset")));
    if (state.deferWatch) return new Promise(resolve => state.pending.push(() => resolve(json(200, body))));
    return json(200, body);
  }
  if (url === "/v1/watch/clear") return state.failMode === "clear" ? json(500, { detail: "no" }) : json(200, { status: "ok" });
  if (url.startsWith("/v1/watch/")) return json(200, { endpoint: "/v1/noul", status_code: 200, timestamp: "2026-10-09T00:00:00Z", request: {}, response: {} });
  if (url === "/health") return json(200, { status: "ok", engine: "e", model_id: "m", backend: "b" });
  return json(404, {});
}
let tick = null;
const context = {
  document: { getElementById: element, createElement },
  fetch: fetchStub,
  setInterval: (callback) => { tick = callback; return 1; },
  alert() {}, confirm() { return true; }, prompt() {},
  navigator: { clipboard: { writeText: async () => {} } },
  console,
};
context.window = context;
vm.createContext(context);
const run = (code) => vm.runInContext(code, context);
const settle = async () => { for (let i = 0; i < 25; i += 1) await new Promise(resolve => setImmediate(resolve)); };
const watchCalls = () => state.log.filter(url => url.startsWith("/v1/watch?")).length;
async function resolves(code) {
  try { await run(code); return true; } catch (error) { return false; }
}

(async () => {
  const result = {};
  run(script);
  await settle();
  result.afterLoad = run("events.length");
  await run("loadOlder()");
  await settle();
  result.afterLoadOlder = run("events.length");
  tick();
  await settle();
  result.afterTimerRefresh = run("events.length");
  result.timerRefreshUrl = state.log[state.log.length - 2];
  state.deferWatch = true;
  const before = watchCalls();
  run("refresh()"); run("refresh()");
  await settle();
  result.inFlightWatchFetches = watchCalls() - before;
  state.pending.splice(0).forEach(resolve => resolve());
  await settle();
  state.deferWatch = false;
  state.failMode = "watch";
  result.loadOlderResolves = await resolves("loadOlder()");
  result.badgeAfterLoadOlderError = element("healthBadge").textContent;
  state.failMode = "throw";
  result.openDetailResolves = await resolves("openDetail('e1')");
  result.badgeAfterDetailError = element("healthBadge").textContent;
  state.failMode = "clear";
  result.clearSessionResolves = await resolves("clearSession()");
  result.badgeAfterClearError = element("healthBadge").textContent;
  process.stdout.write(JSON.stringify(result));
})().catch(error => { process.stdout.write(JSON.stringify({ harnessError: String(error && error.stack || error) })); });
'''


def _run_watch_scenario(tmp_path: Path) -> dict:
    [script] = _scripts(WATCH_DASHBOARD)
    (tmp_path / "watch.js").write_text(script, encoding="utf-8")
    (tmp_path / "harness.js").write_text(HARNESS, encoding="utf-8")
    completed = subprocess.run(
        ["node", str(tmp_path / "harness.js"), str(tmp_path / "watch.js")], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert "harnessError" not in result, result
    return result


def test_watch_auto_refresh_keeps_pages_loaded_with_load_older(tmp_path: Path) -> None:
    result = _run_watch_scenario(tmp_path)
    assert result["afterLoad"] == 500
    assert result["afterLoadOlder"] == 1000
    assert result["afterTimerRefresh"] == 1000, result
    assert "limit=1000" in result["timerRefreshUrl"]


def test_watch_timer_does_not_pile_up_refreshes_while_one_is_in_flight(tmp_path: Path) -> None:
    result = _run_watch_scenario(tmp_path)
    assert result["inFlightWatchFetches"] == 1, result


def test_watch_handlers_report_errors_instead_of_rejecting(tmp_path: Path) -> None:
    result = _run_watch_scenario(tmp_path)
    assert result["loadOlderResolves"] is True
    assert "500" in result["badgeAfterLoadOlderError"]
    assert result["openDetailResolves"] is True
    assert "network down" in result["badgeAfterDetailError"]
    assert result["clearSessionResolves"] is True
    assert "500" in result["badgeAfterClearError"]
