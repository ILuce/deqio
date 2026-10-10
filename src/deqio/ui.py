DASHBOARD = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deqio</title>
<style>
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1180px;
    margin: 32px auto;
    padding: 0 20px 48px;
    background: Canvas;
    color: CanvasText;
}
h1 { margin: 0 0 4px; font-size: 28px; }
h2 { margin: 0 0 16px; font-size: 19px; }
.sub { opacity: .65; margin-bottom: 24px; }
.toolbar, .tabs, .row, .option-row, .actions { display: flex; gap: 10px; align-items: center; }
.toolbar { justify-content: space-between; flex-wrap: wrap; margin-bottom: 18px; }
.runtime-grid { display: grid; grid-template-columns: minmax(260px, 1fr) auto; gap: 10px; align-items: end; }
.runtime-note { opacity: .65; font-size: 13px; margin-top: 8px; }
.tabs { margin-bottom: 18px; }
button, select, input, textarea {
    font: inherit;
    border: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
    border-radius: 8px;
    background: Canvas;
    color: CanvasText;
}
button { padding: 8px 12px; cursor: pointer; }
.nav-link {
    display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 8px;
    border: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
    color: CanvasText; text-decoration: none;
}
button.primary { background: CanvasText; color: Canvas; border-color: CanvasText; }
button.tab.active { font-weight: 700; border-color: CanvasText; }
button.danger { border-color: #a33; }
input, select, textarea { width: 100%; padding: 9px 10px; }
textarea { min-height: 120px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
label { display: block; font-size: 13px; font-weight: 650; margin-bottom: 6px; }
.panel {
    border: 1px solid color-mix(in srgb, CanvasText 15%, transparent);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 18px;
    background: color-mix(in srgb, Canvas 96%, CanvasText 4%);
}
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.field { margin-bottom: 14px; }
.inline-field { min-width: 170px; }
.option-row { margin-bottom: 8px; align-items: stretch; }
.option-row .id { flex: 0 0 190px; }
.option-row .desc { flex: 1; }
.option-row button { flex: 0 0 auto; }
.shared-card {
    border: 1px solid color-mix(in srgb, CanvasText 15%, transparent);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}
.shared-head { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
pre {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.cards { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }
.card { border: 1px solid color-mix(in srgb, CanvasText 15%, transparent); border-radius: 10px; padding: 14px; }
.card small { display: block; opacity: .6; margin-bottom: 5px; }
.card strong { font-size: 20px; }
.result-item { border-top: 1px solid color-mix(in srgb, CanvasText 12%, transparent); padding: 12px 0; }
.result-item:first-child { border-top: 0; padding-top: 0; }
.prob-row { display: grid; grid-template-columns: 160px 1fr 90px; gap: 10px; align-items: center; margin: 7px 0; }
.bar { height: 8px; border-radius: 5px; background: color-mix(in srgb, CanvasText 10%, transparent); overflow: hidden; }
.bar > span { display: block; height: 100%; background: CanvasText; }
.meta { opacity: .7; font-size: 12px; margin-top: 8px; }
.status { font-size: 13px; min-height: 20px; }
.status.error { color: #b23b3b; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 9px; border-bottom: 1px solid color-mix(in srgb, CanvasText 10%, transparent); vertical-align: top; }
th { opacity: .7; }
.hidden { display: none !important; }
.badge { padding: 5px 9px; border: 1px solid color-mix(in srgb, CanvasText 15%, transparent); border-radius: 999px; font-size: 12px; }
.capabilities { display:flex; flex-wrap:wrap; gap:6px; margin:-8px 0 16px; }
.capability { padding:4px 7px; border-radius:999px; font-size:11px; border:1px solid color-mix(in srgb, CanvasText 15%, transparent); }
.capability.off { opacity:.35; text-decoration:line-through; }
.api-guide {
    display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:10px; margin:0 0 16px;
}
.api-guide-card {
    border:1px solid color-mix(in srgb, CanvasText 12%, transparent); border-radius:10px; padding:12px;
    background:color-mix(in srgb, Canvas 98%, CanvasText 2%);
}
.api-guide-card strong { display:block; margin-bottom:5px; font-size:13px; }
.api-guide-card span { display:block; opacity:.68; font-size:12px; line-height:1.45; }
.compare-list { display:grid; gap:12px; margin-top:10px; }
.compare-card {
    border:1px solid color-mix(in srgb, CanvasText 14%, transparent); border-radius:12px; padding:14px;
    background:color-mix(in srgb, Canvas 98%, CanvasText 2%);
}
.compare-card-head { display:flex; gap:10px; align-items:flex-start; justify-content:space-between; margin-bottom:12px; }
.compare-card-title { min-width:0; }
.compare-card-title strong { display:block; font-size:15px; overflow-wrap:anywhere; }
.compare-card-title small { display:block; opacity:.62; margin-top:3px; }
.compare-card-tags { display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
.compare-metrics { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:8px; }
.compare-metric { border:1px solid color-mix(in srgb, CanvasText 10%, transparent); border-radius:9px; padding:10px; min-width:0; }
.compare-metric > strong { display:block; font-size:12px; margin-bottom:8px; }
.compare-values { display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:12px; }
.compare-values span { min-width:0; }
.compare-values small { display:block; opacity:.55; font-size:10px; text-transform:uppercase; letter-spacing:.04em; }
.compare-delta { margin-top:7px; padding-top:7px; border-top:1px solid color-mix(in srgb, CanvasText 8%, transparent); font:600 12px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap:anywhere; }
.benchmark-controls { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin-bottom: 14px; }
.benchmark-run-grid { display: grid; grid-template-columns: minmax(260px, 1fr) auto; gap: 10px; align-items: end; }
.benchmark-meta { font-size: 13px; opacity: .7; margin: 8px 0 14px; }
.table-scroll { overflow: auto; }
.result-pass { font-weight: 700; }
.result-fail { font-weight: 700; color: #b23b3b; }
.compact-pre { max-height: 360px; overflow: auto; }
@media (max-width: 800px) {
    .grid { grid-template-columns: 1fr; }
    .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .option-row { flex-wrap: wrap; }
    .option-row .id { flex: 1 1 140px; }
    .option-row .desc { flex: 1 1 280px; }
    .prob-row { grid-template-columns: 100px 1fr 70px; }
    .benchmark-controls { grid-template-columns: 1fr 1fr; }
    .benchmark-run-grid { grid-template-columns: 1fr; }
    .api-guide { grid-template-columns: 1fr; }
    .compare-metrics { grid-template-columns: 1fr; }
    .compare-card-head { flex-direction:column; }
    .compare-card-tags { justify-content:flex-start; }
}
@media (min-width: 801px) and (max-width: 1100px) {
    .compare-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
</head>
<body>
<div class="toolbar">
  <div>
    <h1>Deqio</h1>
    <div class="sub" id="runtimeSub">Loading runtime information...</div>
  </div>
  <div class="actions">
    <span class="badge" id="healthBadge">health: ...</span>
    <label for="watchAutoClear" style="margin:0;display:flex;align-items:center;gap:6px;font-weight:500">Watch clear
      <select id="watchAutoClear" style="width:auto;padding:7px 9px">
        <option value="0">off</option><option value="15">15 min</option><option value="30">30 min</option>
        <option value="60">1 h</option><option value="120">2 h</option><option value="240">4 h</option>
      </select>
    </label>
    <a class="nav-link" href="/ui/watch">Watch requests</a>
    <button id="clearCache" class="danger" type="button">Clear runtime cache</button>
  </div>
</div>

<section class="panel">
  <h2>Active model</h2>
  <div class="runtime-grid">
    <div class="field" style="margin:0">
      <label for="modelSelect">Installed model / backend</label>
      <select id="modelSelect"><option>Loading installed models...</option></select>
    </div>
    <button id="activateModel" class="primary" type="button">Activate</button>
  </div>
  <div class="runtime-note">Only locally installed profiles are listed. Switching pauses inference requests while the new runtime loads; the public API address stays unchanged.</div>
  <div id="modelStatus" class="status" style="margin-top:8px"></div>
</section>

<section class="panel">
  <h2>Decision playground</h2>
  <div class="api-guide">
    <div class="api-guide-card"><strong>Portable Deqio API</strong><span><code>/v1/noul</code>, <code>/v1/choice</code> and <code>/v1/shared</code> keep the stable cross-model Deqio contract.</span></div>
    <div class="api-guide-card"><strong>Native typed API</strong><span><code>/v1/score</code>, <code>/v1/multi</code> and <code>/v1/act</code> are thin convenience wrappers over the active model's native SystemOne implementation.</span></div>
    <div class="api-guide-card"><strong>SOAM = State Once, Ask Many</strong><span><code>/v1/soam</code> sends one state with several typed questions in one native request. <code>/v1/systemone</code> remains the raw canonical contract.</span></div>
  </div>
  <div class="tabs">
    <button class="tab endpoint-tab active" data-endpoint="noul" type="button">Noul</button>
    <button class="tab endpoint-tab" data-endpoint="choice" type="button">Choice</button>
    <button class="tab endpoint-tab" data-endpoint="shared" type="button">Shared</button>
    <button class="tab endpoint-tab" id="scoreTab" data-endpoint="score" type="button">Score</button>
    <button class="tab endpoint-tab" id="multiTab" data-endpoint="multi" type="button">Multi</button>
    <button class="tab endpoint-tab" id="actTab" data-endpoint="act" type="button">Act</button>
    <button class="tab endpoint-tab" id="systemOneTab" data-endpoint="soam" type="button">SOAM · /v1/soam</button>
  </div>
  <div id="systemOneCapabilities" class="capabilities"></div>

  <div class="grid">
    <div>
      <div class="field">
        <label for="stateFormat">State format</label>
        <select id="stateFormat">
          <option value="text">Text</option>
          <option value="json">JSON</option>
        </select>
      </div>
      <div class="field">
        <label for="stateInput">Context / state</label>
        <textarea id="stateInput">A patch modified three source files responsible for data validation.
The patch applied successfully.
No tests have been run after the changes.
The project has an existing unit test suite.</textarea>
      </div>

      <div id="singleFields">
        <div class="field">
          <label for="questionInput">Question</label>
          <input id="questionInput" value="Should tests be run before considering the task complete?">
        </div>
        <div class="field inline-field">
          <label for="modeInput">Mode</label>
          <select id="modeInput">
            <option value="serial">serial</option>
            <option value="direct">direct</option>
          </select>
        </div>
      </div>

      <div id="choiceFields" class="hidden">
        <div class="field">
          <label>Options</label>
          <div id="choiceOptions"></div>
          <button id="addChoiceOption" type="button">+ Add option</button>
        </div>
      </div>

      <div id="sharedFields" class="hidden">
        <div class="field">
          <label>Decisions</label>
          <div id="sharedDecisions"></div>
          <button id="addSharedDecision" type="button">+ Add decision</button>
        </div>
      </div>

      <div id="nativeSingleFields" class="hidden">
        <div class="field">
          <label for="nativeQuestionInput">Native question JSON</label>
          <textarea id="nativeQuestionInput" style="min-height:260px"></textarea>
          <div class="runtime-note">The endpoint injects its own <code>type</code>. Put native fields such as <code>criteria</code>, <code>threshold</code>, <code>max</code>, <code>costs</code> or <code>evidence</code> here. Unsupported capabilities fail explicitly.</div>
        </div>
        <div class="field">
          <label for="nativeFactsMode">Facts modifier</label>
          <select id="nativeFactsMode"><option value="off">off</option><option value="auto">auto</option></select>
          <div class="runtime-note"><code>facts</code> is a request modifier, not a separate decision endpoint. It is enabled only when the active native runtime supports it.</div>
        </div>
      </div>

      <div id="systemOneFields" class="hidden">
        <div class="field">
          <label for="factsMode">Facts modifier</label>
          <select id="factsMode">
            <option value="off">off</option>
            <option value="auto">auto</option>
          </select>
          <div class="runtime-note"><code>facts: "auto"</code> asks a supporting native runtime (currently Basal 1.5 profiles) to derive supported deterministic facts before the decision. It is not a standalone endpoint.</div>
        </div>
        <div class="field">
          <label for="systemOneQuestions">SOAM questions (JSON object)</label>
          <textarea id="systemOneQuestions" style="min-height:360px"></textarea>
          <div class="runtime-note"><strong>SOAM</strong> means one shared state + many typed questions in one request. <code>option_keys</code> and <code>evidence</code> are per-question modifiers, not separate endpoints. The raw equivalent remains <code>POST /v1/systemone</code>.</div>
        </div>
      </div>

      <div class="actions">
        <button id="sendRequest" class="primary" type="button">Send</button>
        <span id="requestStatus" class="status"></span>
      </div>
    </div>

    <div>
      <div class="field">
        <label>Generated request JSON</label>
        <div class="panel" style="margin:0; min-height:250px"><pre id="requestPreview"></pre></div>
      </div>
    </div>
  </div>
</section>

<section class="panel">
  <h2>Result</h2>
  <div id="resultSummary" class="sub">No request sent yet.</div>
  <div id="resultView"></div>
  <details style="margin-top:14px">
    <summary>Raw response JSON</summary>
    <pre id="responseJson" style="margin-top:10px"></pre>
  </details>
</section>

<section>
  <div class="cards">
    <div class="card"><small>Requests</small><strong id="requests">-</strong></div>
    <div class="card"><small>Decisions</small><strong id="decisions">-</strong></div>
    <div class="card"><small>P50 latency</small><strong id="p50">-</strong></div>
    <div class="card"><small>P95 latency</small><strong id="p95">-</strong></div>
    <div class="card"><small>Cache clears</small><strong id="cacheClears">-</strong></div>
  </div>

  <div class="panel">
    <h2>Recent requests</h2>
    <div style="overflow:auto">
      <table>
        <thead><tr><th>Time</th><th>Engine</th><th>Backend</th><th>Mode</th><th>Question</th><th>Decision</th><th>Latency</th><th>Cache</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>
</section>

<section class="panel" id="benchmarkPanel">
  <div class="toolbar" style="margin-bottom:12px">
    <div>
      <h2 style="margin-bottom:4px">Benchmark results</h2>
      <div class="runtime-note">Reads completed benchmark runs from <code>.deqio/benchmarks/</code>. No benchmark is executed from the UI.</div>
    </div>
    <button id="refreshBenchmarks" type="button">Refresh</button>
  </div>

  <div id="benchmarkEmpty" class="runtime-note">Checking for benchmark runs...</div>
  <div id="benchmarkContent" class="hidden">
    <div class="benchmark-run-grid">
      <div class="field" style="margin:0">
        <label for="benchmarkRunSelect">Benchmark run</label>
        <select id="benchmarkRunSelect"></select>
      </div>
      <span id="benchmarkRunBadge" class="badge">-</span>
    </div>
    <div id="benchmarkMeta" class="benchmark-meta"></div>

    <div class="tabs">
      <button class="benchmark-tab tab active" data-benchmark-view="summary" type="button">Summary</button>
      <button class="benchmark-tab tab" data-benchmark-view="results" type="button">Results</button>
      <button class="benchmark-tab tab" data-benchmark-view="compare" type="button">Compare</button>
    </div>

    <div id="benchmarkSummaryView">
      <div class="benchmark-controls">
        <div><label for="benchmarkSummaryModel">Model</label><select id="benchmarkSummaryModel"></select></div>
        <div><label for="benchmarkSummaryType">Type</label><select id="benchmarkSummaryType">
          <option value="overall">Overall</option><option value="noul">Noul</option><option value="choice">Choice</option><option value="shared">Shared</option><option value="*">All rows</option>
        </select></div>
        <div><label for="benchmarkSummarySort">Sort by</label><select id="benchmarkSummarySort">
          <option value="case_accuracy">Accuracy</option><option value="decision_accuracy">Decision accuracy</option><option value="mean_ms">Mean latency</option><option value="median_ms">Median latency</option><option value="p95_ms">P95 latency</option><option value="throughput_decisions_per_s">Throughput</option><option value="runtime_errors">Runtime errors</option><option value="load_ms">Model load time</option><option value="cases">Cases</option>
        </select></div>
        <div><label for="benchmarkSummaryDirection">Order</label><select id="benchmarkSummaryDirection">
          <option value="desc">Highest first</option><option value="asc">Lowest / fastest first</option>
        </select></div>
      </div>
      <div id="benchmarkSummaryStatus" class="status"></div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Model</th><th>Engine</th><th>Type</th><th>Cases</th><th>Passed</th><th>Runtime errors</th><th>Accuracy</th><th>Decision acc.</th><th>Mean</th><th>Median</th><th>P95</th><th>Throughput</th><th>Load</th></tr></thead>
          <tbody id="benchmarkSummaryRows"></tbody>
        </table>
      </div>
      <details style="margin-top:14px"><summary>Raw summary.json</summary><pre id="benchmarkRawSummary" class="compact-pre" style="margin-top:10px"></pre></details>
    </div>

    <div id="benchmarkResultsView" class="hidden">
      <div class="benchmark-controls">
        <div><label for="benchmarkResultModel">Model</label><select id="benchmarkResultModel"></select></div>
        <div><label for="benchmarkResultType">Type</label><select id="benchmarkResultType">
          <option value="*">All types</option><option value="noul">Noul</option><option value="choice">Choice</option><option value="shared">Shared</option>
        </select></div>
        <div><label for="benchmarkResultStatus">Status</label><select id="benchmarkResultStatus">
          <option value="*">PASS + FAIL</option><option value="pass">PASS</option><option value="fail">FAIL</option>
        </select></div>
        <div><label for="benchmarkResultSort">Sort</label><select id="benchmarkResultSort">
          <option value="case_id:asc">Case ID</option><option value="latency_ms:asc">Latency: fastest</option><option value="latency_ms:desc">Latency: slowest</option><option value="top_probability:desc">Confidence: highest</option><option value="top_probability:asc">Confidence: lowest</option>
        </select></div>
      </div>
      <div id="benchmarkResultsStatus" class="status"></div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Case</th><th>Model</th><th>Engine</th><th>Type</th><th>Status</th><th>Expected</th><th>Actual</th><th>Confidence</th><th>Latency</th><th>Error</th></tr></thead>
          <tbody id="benchmarkResultRows"></tbody>
        </table>
      </div>
      <details style="margin-top:14px"><summary>Raw results.jsonl (parsed)</summary><pre id="benchmarkRawResults" class="compact-pre" style="margin-top:10px"></pre></details>
    </div>

    <div id="benchmarkCompareView" class="hidden">
      <div class="benchmark-controls">
        <div><label for="benchmarkCompareLeft">Benchmark A</label><select id="benchmarkCompareLeft"></select></div>
        <div><label for="benchmarkCompareRight">Benchmark B</label><select id="benchmarkCompareRight"></select></div>
        <div><label for="benchmarkCompareBackend">Backend</label><select id="benchmarkCompareBackend"><option value="*">All backends</option></select></div>
        <div><label for="benchmarkCompareSort">Sort by</label><select id="benchmarkCompareSort">
          <option value="accuracy">Accuracy delta</option><option value="decision_accuracy">Decision accuracy delta</option><option value="median">Median latency delta</option><option value="p95">P95 latency delta</option><option value="throughput">Throughput delta</option><option value="rank">Rank shift</option>
        </select></div>
      </div>
      <div id="benchmarkCompareStatus" class="status"></div>
      <div id="benchmarkCompareCards" class="compare-list"></div>
      <div id="benchmarkCompareOnly" class="benchmark-meta"></div>
      <details style="margin-top:14px"><summary>Raw comparison JSON</summary><pre id="benchmarkRawComparison" class="compact-pre" style="margin-top:10px"></pre></details>
    </div>
  </div>
</section>

<script>
let endpoint = 'noul';
let sharedCounter = 0;
let activeModelKey = '';
let installedModels = [];
let benchmarkRuns = [];
let benchmarkSummary = null;
let benchmarkResults = [];
let benchmarkComparison = null;
let benchmarkRunId = '';
let systemOneExampleModelKey = '';
const endpointInitialized = new Set();

const EXAMPLES = {
  noul: {
    stateFormat: 'text',
    state: `A patch modified three source files responsible for data validation.
The patch applied successfully.
No tests have been run after the changes.
The project has an existing unit test suite.`,
    question: 'Should tests be run before considering the task complete?',
  },
  choice: {
    stateFormat: 'text',
    state: `A customer was charged twice for the same subscription renewal.
The service itself is working and the customer can access the account.
The issue concerns the duplicate payment only.`,
    question: 'Which team should handle this request?',
    options: [
      { id: 'billing', description: 'Handle payments, refunds, invoices, and duplicate charges.' },
      { id: 'access', description: 'Handle login and account access problems.' },
      { id: 'technical', description: 'Handle product defects and service failures.' },
    ],
  },
  shared: {
    stateFormat: 'text',
    state: `A patch changed an internal authentication module.
Unit tests passed.
Integration tests have not been run yet.
The change does not modify the public API.
A rollback commit is available.`,
    decisions: [
      {
        question: 'What should be the next validation action?',
        options: [
          { id: 'run_integration_tests', description: 'Run the integration test suite before proceeding.' },
          { id: 'finish_task', description: 'Finish the task without additional validation.' },
          { id: 'revert_change', description: 'Immediately revert the change.' },
        ],
      },
      {
        question: 'How should the change be treated before integration tests run?',
        options: [
          { id: 'continue_validation', description: 'Keep the change and continue validation.' },
          { id: 'ready_for_release', description: 'Treat the change as fully verified and ready for release.' },
          { id: 'revert_immediately', description: 'Revert the change without further investigation.' },
        ],
      },
    ],
  },
};

const NATIVE_EXAMPLES = {
  score: {
    state: 'A production incident affects checkout for some customers. Orders can still be placed after a retry, but payment latency is elevated.',
    question: { instructions: 'Rate the operational severity.', criteria: ['low', 'medium', 'high', 'critical'] },
  },
  multi: {
    state: 'The customer reports a cracked screen, a damaged shipping box, and asks for a refund instead of repair.',
    question: {
      instructions: 'Which labels apply to this case?',
      criteria: { damage: 'Physical product damage.', delivery: 'Delivery or courier problem.', refund: 'Customer requests a refund.', repair: 'Customer requests repair.' },
      threshold: 0.5, max: 3,
    },
  },
  act: {
    state: 'A refund claim is likely valid, but approving an invalid claim is much more expensive than asking a human to review it.',
    question: {
      instructions: 'Choose the next action using the supplied costs.',
      criteria: { true: 'The refund is justified.', false: 'The refund is not justified.' },
      costs: { approve: { true: 0, false: 1000 }, reject: { true: 200, false: 0 }, human: { true: 20, false: 20 } },
    },
  },
};

function activeProfile() {
  return installedModels.find(model => modelKey(model) === activeModelKey) || null;
}

function activeCapabilities() {
  return activeProfile()?.capabilities || {};
}

function buildSystemOneExample(caps = {}) {
  const questions = {};
  if (caps.choice) {
    questions.route = {
      type: 'choice',
      instructions: 'Which workflow should handle this request?',
      criteria: { refund: 'Refund the purchase.', repair: 'Repair or replace the item.', review: 'Send for human review.' },
    };
    if (caps.option_keys) questions.route.option_keys = 'show';
  }
  if (caps.noul) {
    questions.damaged = { type: 'noul', instructions: 'Is the item damaged?' };
    if (caps.evidence) questions.damaged.evidence = true;
  }
  if (caps.score) {
    questions.severity = { type: 'score', instructions: 'Rate the issue severity.', criteria: ['low', 'medium', 'high'] };
  }
  if (caps.multi) {
    questions.tags = {
      type: 'multi', instructions: 'Which labels apply?',
      criteria: { damage: 'Physical damage.', delivery: 'Delivery problem.', refund: 'Refund requested.' },
      threshold: 0.5, max: 3,
    };
  }
  if (caps.act) {
    questions.next_action = {
      type: 'act', instructions: 'Choose the next action.',
      criteria: { true: 'The refund is justified.', false: 'The refund is not justified.' },
      costs: { approve: { true: 0, false: 1000 }, reject: { true: 200, false: 0 }, human: { true: 20, false: 20 } },
    };
  }
  if (!Object.keys(questions).length) {
    questions.decision = { type: 'choice', instructions: 'Choose the best option.', criteria: { yes: 'Yes.', no: 'No.' } };
  }
  return questions;
}

function refreshSystemOneCapabilities() {
  const caps = activeCapabilities();
  const soam = Boolean(caps.soam || caps.multi_question);
  const definitions = [
    [caps.systemone, 'raw /v1/systemone'],
    [soam, 'SOAM /v1/soam'],
    [caps.score, 'Score /v1/score'],
    [caps.multi, 'Multi /v1/multi'],
    [caps.act, 'Act /v1/act'],
    [caps.evidence, 'Evidence modifier'],
    [caps.facts, 'Facts modifier'],
    [caps.option_keys, 'Option keys modifier'],
  ];
  const holder = byId('systemOneCapabilities');
  holder.innerHTML = definitions.map(([enabled, label]) => `<span class="capability ${enabled ? '' : 'off'}">${label}</span>`).join('');

  const tabSupport = { score: Boolean(caps.systemone && caps.score), multi: Boolean(caps.systemone && caps.multi), act: Boolean(caps.systemone && caps.act), soam: Boolean(caps.systemone && soam) };
  for (const [name, supported] of Object.entries(tabSupport)) {
    const tab = byId(name === 'soam' ? 'systemOneTab' : `${name}Tab`);
    tab.disabled = !supported;
    tab.title = supported ? `Supported by ${activeProfile()?.label || 'the active profile'}` : `The active profile does not support ${name}`;
  }
  byId('factsMode').disabled = !caps.facts;
  byId('nativeFactsMode').disabled = !caps.facts;
  if (!caps.facts) { byId('factsMode').value = 'off'; byId('nativeFactsMode').value = 'off'; }

  if (['score', 'multi', 'act', 'soam'].includes(endpoint) && !tabSupport[endpoint]) {
    setEndpoint('noul');
  }

  if (activeModelKey !== systemOneExampleModelKey) {
    for (const nativeEndpoint of ['score', 'multi', 'act', 'soam']) endpointInitialized.delete(nativeEndpoint);
    systemOneExampleModelKey = activeModelKey;
    if (['score', 'multi', 'act', 'soam'].includes(endpoint)) {
      endpointInitialized.add(endpoint);
      loadEndpointExample(endpoint);
      updatePreview();
    }
  }
}

const byId = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '')
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#039;');

function optionRow(id = '', description = '') {
  const row = document.createElement('div');
  row.className = 'option-row';
  row.innerHTML = `
    <input class="id" placeholder="id" value="${escapeHtml(id)}">
    <input class="desc" placeholder="description" value="${escapeHtml(description)}">
    <button type="button" class="remove-option">Remove</button>`;
  row.querySelector('.remove-option').addEventListener('click', () => {
    row.remove();
    updatePreview();
  });
  return row;
}

function addOption(container, id = '', description = '') {
  container.appendChild(optionRow(id, description));
  updatePreview();
}

function readOptions(container) {
  return [...container.querySelectorAll('.option-row')].map(row => ({
    id: row.querySelector('.id').value.trim(),
    description: row.querySelector('.desc').value.trim(),
  }));
}

function addSharedDecision(question = 'Is action required?', initialOptions = null) {
  sharedCounter += 1;
  const card = document.createElement('div');
  card.className = 'shared-card';
  card.dataset.decision = String(sharedCounter);
  card.innerHTML = `
    <div class="shared-head"><strong>Decision ${sharedCounter}</strong><button type="button" class="remove-decision">Remove</button></div>
    <div class="field"><label>Question</label><input class="shared-question" value="${escapeHtml(question)}"></div>
    <div class="field"><label>Options</label><div class="shared-options"></div><button type="button" class="add-shared-option">+ Add option</button></div>`;
  const options = card.querySelector('.shared-options');
  const seedOptions = initialOptions || [
    { id: 'yes', description: 'Yes. The evidence supports the criterion.' },
    { id: 'no', description: 'No. The evidence does not support the criterion.' },
  ];
  seedOptions.forEach(option => addOption(options, option.id, option.description));
  card.querySelector('.add-shared-option').addEventListener('click', () => addOption(options));
  card.querySelector('.remove-decision').addEventListener('click', () => {
    card.remove();
    updatePreview();
  });
  byId('sharedDecisions').appendChild(card);
  updatePreview();
}

function readState() {
  const raw = byId('stateInput').value.trim();
  if (!raw) throw new Error('State cannot be empty.');
  if (byId('stateFormat').value === 'json') {
    const parsed = JSON.parse(raw);
    if (parsed === null || (typeof parsed !== 'object')) {
      throw new Error('JSON state must be an object or array.');
    }
    return parsed;
  }
  return raw;
}

function buildPayload() {
  const state = readState();
  if (endpoint === 'noul') {
    return { state, question: byId('questionInput').value.trim(), mode: byId('modeInput').value };
  }
  if (endpoint === 'choice') {
    return {
      state,
      question: byId('questionInput').value.trim(),
      mode: byId('modeInput').value,
      options: readOptions(byId('choiceOptions')),
    };
  }
  if (['score', 'multi', 'act'].includes(endpoint)) {
    const question = JSON.parse(byId('nativeQuestionInput').value.trim() || '{}');
    if (!question || Array.isArray(question) || typeof question !== 'object' || !Object.keys(question).length) {
      throw new Error('Native question must be a non-empty JSON object.');
    }
    const payload = { state, question };
    if (byId('nativeFactsMode').value === 'auto') payload.facts = 'auto';
    return payload;
  }
  if (endpoint === 'soam') {
    const raw = byId('systemOneQuestions').value.trim();
    const questions = JSON.parse(raw || '{}');
    if (!questions || Array.isArray(questions) || typeof questions !== 'object' || !Object.keys(questions).length) {
      throw new Error('SOAM questions must be a non-empty JSON object.');
    }
    const payload = { state, questions };
    if (byId('factsMode').value === 'auto') payload.facts = 'auto';
    return payload;
  }
  return {
    state,
    decisions: [...byId('sharedDecisions').querySelectorAll('.shared-card')].map(card => ({
      id: `ui-shared-${card.dataset.decision}`,
      question: card.querySelector('.shared-question').value.trim(),
      options: readOptions(card.querySelector('.shared-options')),
    })),
  };
}

function updatePreview() {
  try {
    byId('requestPreview').textContent = JSON.stringify(buildPayload(), null, 2);
    byId('requestStatus').textContent = '';
    byId('requestStatus').className = 'status';
  } catch (error) {
    byId('requestPreview').textContent = `Invalid input: ${error.message}`;
  }
}

function loadEndpointExample(next) {
  if (['score', 'multi', 'act'].includes(next)) {
    const example = NATIVE_EXAMPLES[next];
    byId('stateFormat').value = 'text';
    byId('stateInput').value = example.state;
    const question = JSON.parse(JSON.stringify(example.question));
    const caps = activeCapabilities();
    if (caps.evidence && next !== 'multi') question.evidence = true;
    byId('nativeQuestionInput').value = JSON.stringify(question, null, 2);
    byId('nativeFactsMode').value = caps.facts ? 'auto' : 'off';
    return;
  }
  if (next === 'soam') {
    byId('stateFormat').value = 'text';
    byId('stateInput').value = 'The customer reports that a laptop arrived with a cracked screen and asks for a refund. The courier left the parcel at the door.';
    const caps = activeCapabilities();
    byId('systemOneQuestions').value = JSON.stringify(buildSystemOneExample(caps), null, 2);
    byId('factsMode').value = caps.facts ? 'auto' : 'off';
    systemOneExampleModelKey = activeModelKey;
    return;
  }
  const example = EXAMPLES[next];
  byId('stateFormat').value = example.stateFormat;
  byId('stateInput').value = example.state;

  if (next === 'noul') {
    byId('questionInput').value = example.question;
  } else if (next === 'choice') {
    byId('questionInput').value = example.question;
    byId('choiceOptions').replaceChildren();
    example.options.forEach(option => addOption(byId('choiceOptions'), option.id, option.description));
  } else if (next === 'shared') {
    byId('sharedDecisions').replaceChildren();
    sharedCounter = 0;
    example.decisions.forEach(decision => addSharedDecision(decision.question, decision.options));
  }
}

function setEndpoint(next) {
  endpoint = next;
  const nativeSingle = ['score', 'multi', 'act'].includes(next);
  document.querySelectorAll('.endpoint-tab').forEach(button => button.classList.toggle('active', button.dataset.endpoint === next));
  byId('singleFields').classList.toggle('hidden', next === 'shared' || next === 'soam' || nativeSingle);
  byId('choiceFields').classList.toggle('hidden', next !== 'choice');
  byId('sharedFields').classList.toggle('hidden', next !== 'shared');
  byId('nativeSingleFields').classList.toggle('hidden', !nativeSingle);
  byId('systemOneFields').classList.toggle('hidden', next !== 'soam');
  if (!endpointInitialized.has(next)) {
    endpointInitialized.add(next);
    loadEndpointExample(next);
  }
  updatePreview();
}

function renderOneResult(result) {
  const probabilities = Object.entries(result.probabilities || {});
  const bars = probabilities.map(([id, probability]) => `
    <div class="prob-row">
      <strong>${escapeHtml(id)}</strong>
      <div class="bar"><span style="width:${Math.max(0, Math.min(100, probability * 100))}%"></span></div>
      <span>${(probability * 100).toFixed(2)}%</span>
    </div>`).join('');
  const timing = result.timing || {};
  return `
    <div class="result-item">
      <div><strong>Decision: ${escapeHtml(result.decision)}</strong></div>
      ${bars}
      <div class="meta">tokens=${escapeHtml(result.input_tokens ?? '-')} · total=${escapeHtml(timing.total_ms ?? '-')} ms · cache=${escapeHtml(timing.cache_hit ?? '-')}</div>
      <details><summary>Option logits</summary><pre>${escapeHtml(JSON.stringify(result.option_logits || {}, null, 2))}</pre></details>
    </div>`;
}

function renderResponse(data) {
  byId('responseJson').textContent = JSON.stringify(data, null, 2);
  if (data.answers && typeof data.answers === 'object') {
    const answers = Object.entries(data.answers);
    byId('resultSummary').textContent = `${answers.length} native answer${answers.length === 1 ? '' : 's'} · ${data.deqio?.timing?.total_ms ?? data.usage?.latency_ms ?? '-'} ms`;
    byId('resultView').innerHTML = answers.map(([id, answer]) => {
      const probabilities = Object.entries(answer.probabilities || {});
      const bars = probabilities.map(([key, probability]) => `
        <div class="prob-row"><strong>${escapeHtml(key)}</strong><div class="bar"><span style="width:${Math.max(0, Math.min(100, Number(probability) * 100))}%"></span></div><span>${(Number(probability) * 100).toFixed(2)}%</span></div>`).join('');
      const primary = answer.choice ?? answer.action ?? answer.score ?? answer.noul ?? answer.selected ?? '';
      return `<div class="result-item"><div><strong>${escapeHtml(id)} · ${escapeHtml(answer.type || 'answer')}</strong>${primary !== '' ? ` · ${escapeHtml(Array.isArray(primary) ? primary.join(', ') : primary)}` : ''}</div>${bars}<details><summary>Native answer</summary><pre>${escapeHtml(JSON.stringify(answer, null, 2))}</pre></details></div>`;
    }).join('');
  } else if (Array.isArray(data.results)) {
    byId('resultSummary').textContent = `${data.results.length} shared decisions · ${data.shared_timing?.total_ms ?? '-'} ms total`;
    byId('resultView').innerHTML = data.results.map(renderOneResult).join('');
  } else {
    byId('resultSummary').textContent = `${data.decision ?? 'result'} · ${data.timing?.total_ms ?? '-'} ms`;
    byId('resultView').innerHTML = renderOneResult(data);
  }
}


function modelKey(model) {
  return `${model.model_id}::${model.backend}`;
}

async function refreshModels() {
  const select = byId('modelSelect');
  const status = byId('modelStatus');
  try {
    const data = await fetch('/v1/models/installed').then(async response => {
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || JSON.stringify(body));
      return body;
    });
    installedModels = data.installed || [];
    activeModelKey = `${data.active.model_id}::${data.active.backend}`;
    select.replaceChildren();
    if (!installedModels.length) {
      const option = document.createElement('option');
      option.textContent = 'No installed models detected';
      option.value = '';
      select.appendChild(option);
      select.disabled = true;
      byId('activateModel').disabled = true;
      refreshSystemOneCapabilities();
      return;
    }
    select.disabled = false;
    installedModels.forEach(model => {
      const option = document.createElement('option');
      option.value = modelKey(model);
      option.textContent = `${model.label} · ${model.backend} · ${model.engine}${model.active ? ' · active' : ''}`;
      option.selected = option.value === activeModelKey;
      select.appendChild(option);
    });
    byId('activateModel').disabled = select.value === activeModelKey;
    refreshSystemOneCapabilities();
  } catch (error) {
    status.textContent = `Could not inspect installed models: ${error.message}`;
    status.className = 'status error';
  }
}

async function activateSelectedModel() {
  const select = byId('modelSelect');
  const button = byId('activateModel');
  const status = byId('modelStatus');
  const selected = installedModels.find(model => modelKey(model) === select.value);
  if (!selected || select.value === activeModelKey) return;

  button.disabled = true;
  select.disabled = true;
  status.textContent = `Switching to ${selected.label} / ${selected.backend}...`;
  status.className = 'status';
  try {
    const response = await fetch('/v1/models/activate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model_id: selected.model_id, backend: selected.backend }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    status.textContent = `Active: ${data.active.model_id} / ${data.active.backend} · loaded in ${data.load_ms.toFixed(1)} ms`;
    await Promise.all([refreshHealth(), refreshStats(), refreshModels()]);
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
    select.disabled = false;
    button.disabled = select.value === activeModelKey;
  }
}

async function sendRequest() {
  const status = byId('requestStatus');
  try {
    const payload = buildPayload();
    if (!['shared', 'soam'].includes(endpoint) && !payload.question) throw new Error('Question cannot be empty.');
    if (endpoint === 'choice' && payload.options.length < 2) throw new Error('Choice needs at least two options.');
    if (endpoint === 'shared' && payload.decisions.length < 1) throw new Error('Shared needs at least one decision.');
    status.textContent = 'Sending...';
    status.className = 'status';
    const response = await fetch(`/v1/${endpoint}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    renderResponse(data);
    status.textContent = 'Done.';
    await refreshStats();
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
  }
}

async function clearRuntimeCache() {
  const button = byId('clearCache');
  button.disabled = true;
  try {
    const response = await fetch('/v1/cache/clear', { method: 'POST' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    byId('requestStatus').textContent = data.cache_clear_supported === false
      ? `Cache clearing is not exposed by ${data.engine || data.backend}; model remains loaded.`
      : `Cache cleared (${data.backend}). Model remains loaded.`;
    byId('requestStatus').className = 'status';
    await refreshStats();
  } catch (error) {
    byId('requestStatus').textContent = error.message;
    byId('requestStatus').className = 'status error';
  } finally {
    button.disabled = false;
  }
}

function benchmarkProfileKey(row) {
  return `${row.model_id}::${row.backend}`;
}

function formatBenchmarkValue(value) {
  if (value == null) return '-';
  if (Array.isArray(value) || typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function formatBenchmarkMs(value) {
  return value == null ? '-' : `${Number(value).toFixed(1)} ms`;
}

function formatBenchmarkPercent(value) {
  return value == null ? '-' : `${(Number(value) * 100).toFixed(1)}%`;
}

function setSelectOptions(select, rows, valueFor, labelFor, allLabel) {
  const previous = select.value;
  select.replaceChildren();
  if (allLabel) {
    const option = document.createElement('option');
    option.value = '*';
    option.textContent = allLabel;
    select.appendChild(option);
  }
  rows.forEach(row => {
    const option = document.createElement('option');
    option.value = valueFor(row);
    option.textContent = labelFor(row);
    select.appendChild(option);
  });
  if ([...select.options].some(option => option.value === previous)) select.value = previous;
}

function flattenBenchmarkSummary(summary) {
  const rows = [];
  for (const model of summary?.models || []) {
    if (model.load_error) {
      rows.push({ model_id: model.model_id, backend: model.backend, engine: model.engine, load_ms: model.load_ms, type: 'overall', load_error: model.load_error });
      continue;
    }
    for (const type of ['overall', 'noul', 'choice', 'shared']) {
      const stats = model.summary?.[type];
      if (!stats) continue;
      rows.push({
        model_id: model.model_id,
        backend: model.backend,
        engine: model.engine,
        load_ms: model.load_ms,
        type,
        ...stats,
      });
    }
  }
  return rows;
}

function renderBenchmarkSummary() {
  if (!benchmarkSummary) {
    byId('benchmarkSummaryRows').innerHTML = '';
    byId('benchmarkSummaryStatus').textContent = 'summary.json is not available for this run.';
    return;
  }
  const model = byId('benchmarkSummaryModel').value;
  const type = byId('benchmarkSummaryType').value;
  const metric = byId('benchmarkSummarySort').value;
  const direction = byId('benchmarkSummaryDirection').value;
  let rows = flattenBenchmarkSummary(benchmarkSummary).filter(row =>
    (model === '*' || benchmarkProfileKey(row) === model) && (type === '*' || row.type === type)
  );
  const multiplier = direction === 'asc' ? 1 : -1;
  rows.sort((a, b) => {
    const av = a[metric];
    const bv = b[metric];
    if (av == null && bv == null) return benchmarkProfileKey(a).localeCompare(benchmarkProfileKey(b));
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * multiplier;
    return String(av).localeCompare(String(bv)) * multiplier;
  });
  byId('benchmarkSummaryStatus').className = 'status';
  byId('benchmarkSummaryStatus').textContent = `${rows.length} summary row${rows.length === 1 ? '' : 's'}.`;
  byId('benchmarkSummaryRows').innerHTML = rows.map(row => row.load_error ? `
    <tr>
      <td>${escapeHtml(`${row.model_id}:${row.backend}`)}</td>
      <td>${escapeHtml(row.engine ?? '-')}</td>
      <td>all</td>
      <td colspan="9">load error: ${escapeHtml(row.load_error)}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.load_ms))}</td>
    </tr>` : `
    <tr>
      <td>${escapeHtml(`${row.model_id}:${row.backend}`)}</td>
      <td>${escapeHtml(row.engine ?? '-')}</td>
      <td>${escapeHtml(row.type === 'overall' ? 'all' : row.type)}</td>
      <td>${escapeHtml(row.cases ?? '-')}</td>
      <td>${escapeHtml(row.passed_cases ?? '-')}</td>
      <td>${escapeHtml(row.runtime_errors ?? '-')}</td>
      <td>${escapeHtml(formatBenchmarkPercent(row.case_accuracy))}</td>
      <td>${escapeHtml(formatBenchmarkPercent(row.decision_accuracy))}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.mean_ms))}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.median_ms))}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.p95_ms))}</td>
      <td>${escapeHtml(row.throughput_decisions_per_s == null ? '-' : `${Number(row.throughput_decisions_per_s).toFixed(2)}/s`)}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.load_ms))}</td>
    </tr>`).join('');
}

function renderBenchmarkResults() {
  const model = byId('benchmarkResultModel').value;
  const type = byId('benchmarkResultType').value;
  const status = byId('benchmarkResultStatus').value;
  const [metric, direction] = byId('benchmarkResultSort').value.split(':');
  let rows = benchmarkResults.filter(row =>
    (model === '*' || benchmarkProfileKey(row) === model) &&
    (type === '*' || row.kind === type) &&
    (status === '*' || (status === 'pass' ? row.passed : !row.passed))
  );
  const multiplier = direction === 'asc' ? 1 : -1;
  rows.sort((a, b) => {
    const av = a[metric];
    const bv = b[metric];
    if (av == null && bv == null) return String(a.case_id).localeCompare(String(b.case_id));
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * multiplier;
    return String(av).localeCompare(String(bv)) * multiplier;
  });
  const failed = rows.filter(row => !row.passed).length;
  byId('benchmarkResultsStatus').className = 'status';
  byId('benchmarkResultsStatus').textContent = `${rows.length} result${rows.length === 1 ? '' : 's'} · ${failed} FAIL.`;
  byId('benchmarkResultRows').innerHTML = rows.map(row => `
    <tr>
      <td>${escapeHtml(row.case_id ?? '-')}</td>
      <td>${escapeHtml(`${row.model_id}:${row.backend}`)}</td>
      <td>${escapeHtml(row.engine ?? '-')}</td>
      <td>${escapeHtml(row.kind ?? '-')}</td>
      <td class="${row.passed ? 'result-pass' : 'result-fail'}">${row.passed ? 'PASS' : 'FAIL'}</td>
      <td>${escapeHtml(formatBenchmarkValue(row.expected))}</td>
      <td>${escapeHtml(formatBenchmarkValue(row.actual))}</td>
      <td>${escapeHtml(row.top_probability == null ? '-' : formatBenchmarkPercent(row.top_probability))}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.latency_ms))}</td>
      <td>${escapeHtml(row.error ?? '-')}</td>
    </tr>`).join('');
}

function populateBenchmarkFilters() {
  const summaryModels = [];
  const seenSummary = new Set();
  for (const row of benchmarkSummary?.models || []) {
    const key = benchmarkProfileKey(row);
    if (!seenSummary.has(key)) { seenSummary.add(key); summaryModels.push(row); }
  }
  setSelectOptions(byId('benchmarkSummaryModel'), summaryModels, benchmarkProfileKey, row => `${row.model_id}:${row.backend}`, 'All models');

  const resultModels = [];
  const seenResults = new Set();
  for (const row of benchmarkResults) {
    const key = benchmarkProfileKey(row);
    if (!seenResults.has(key)) { seenResults.add(key); resultModels.push(row); }
  }
  setSelectOptions(byId('benchmarkResultModel'), resultModels, benchmarkProfileKey, row => `${row.model_id}:${row.backend}`, 'All models');
}

async function loadBenchmarkRun(runId) {
  const metadata = benchmarkRuns.find(run => run.id === runId);
  if (!metadata) return;
  benchmarkRunId = runId;
  byId('benchmarkRunBadge').textContent = metadata.results ? `${metadata.results} results` : 'benchmark run';
  byId('benchmarkMeta').textContent = `${metadata.suite_name || 'unknown suite'} · ${new Date(metadata.created_at).toLocaleString()} · ${metadata.models ?? '-'} model(s)`;
  byId('benchmarkSummaryStatus').textContent = 'Loading summary...';
  byId('benchmarkResultsStatus').textContent = 'Loading results...';

  const loadOptional = async (url, available) => {
    if (!available) return null;
    const response = await fetch(url);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || JSON.stringify(body));
    return body;
  };

  try {
    const [summary, results] = await Promise.all([
      loadOptional(`/v1/benchmarks/${encodeURIComponent(runId)}/summary`, metadata.has_summary),
      loadOptional(`/v1/benchmarks/${encodeURIComponent(runId)}/results`, metadata.has_results),
    ]);
    benchmarkSummary = summary;
    benchmarkResults = results?.results || [];
    byId('benchmarkRawSummary').textContent = benchmarkSummary ? JSON.stringify(benchmarkSummary, null, 2) : 'summary.json not available';
    byId('benchmarkRawResults').textContent = JSON.stringify(benchmarkResults, null, 2);
    populateBenchmarkFilters();
    renderBenchmarkSummary();
    renderBenchmarkResults();
  } catch (error) {
    benchmarkSummary = null;
    benchmarkResults = [];
    byId('benchmarkSummaryStatus').textContent = error.message;
    byId('benchmarkSummaryStatus').className = 'status error';
    byId('benchmarkResultsStatus').textContent = error.message;
    byId('benchmarkResultsStatus').className = 'status error';
  }
}

function comparisonMetric(row, name) {
  const metrics = row.metrics || {};
  if (name === 'accuracy') return metrics.case_accuracy?.absolute ?? null;
  if (name === 'decision_accuracy') return metrics.decision_accuracy?.absolute ?? null;
  if (name === 'median') return metrics.median_ms?.absolute ?? null;
  if (name === 'p95') return metrics.p95_ms?.absolute ?? null;
  if (name === 'throughput') return metrics.throughput_decisions_per_s?.absolute ?? null;
  if (name === 'rank') return row.rank?.shift ?? null;
  return null;
}

function formatComparisonDelta(value, kind) {
  if (value == null) return '-';
  if (kind === 'accuracy' || kind === 'decision_accuracy') return `${value >= 0 ? '+' : ''}${(Number(value) * 100).toFixed(1)} pp`;
  if (kind === 'rank') return `${value >= 0 ? '+' : ''}${Number(value).toFixed(0)}`;
  return `${value >= 0 ? '+' : ''}${Number(value).toFixed(2)}`;
}

function formatComparisonMetricDelta(metric, kind) {
  if (!metric || metric.absolute == null) return '-';
  let absolute;
  if (kind === 'accuracy' || kind === 'decision_accuracy') absolute = formatComparisonDelta(metric.absolute, kind);
  else if (kind === 'median' || kind === 'p95') absolute = `${Number(metric.absolute) >= 0 ? '+' : ''}${Number(metric.absolute).toFixed(1)} ms`;
  else if (kind === 'throughput') absolute = `${Number(metric.absolute) >= 0 ? '+' : ''}${Number(metric.absolute).toFixed(2)}/s`;
  else absolute = formatComparisonDelta(metric.absolute, kind);
  return metric.percent == null ? absolute : `${absolute} (${Number(metric.percent) >= 0 ? '+' : ''}${(Number(metric.percent) * 100).toFixed(1)}%)`;
}

function populateBenchmarkComparisonSelectors() {
  const runs = benchmarkRuns.filter(run => run.has_summary);
  const left = byId('benchmarkCompareLeft');
  const right = byId('benchmarkCompareRight');
  const previousLeft = left.value;
  const previousRight = right.value;
  for (const select of [left, right]) {
    select.replaceChildren();
    runs.forEach(run => {
      const option = document.createElement('option');
      option.value = run.id;
      option.textContent = `${new Date(run.created_at).toLocaleString()} · ${run.suite_name || run.id}`;
      select.appendChild(option);
    });
  }
  if (runs.length) {
    left.value = runs.some(run => run.id === previousLeft) ? previousLeft : runs[0].id;
    const fallbackRight = runs.find(run => run.id !== left.value)?.id || left.value;
    right.value = runs.some(run => run.id === previousRight && run.id !== left.value) ? previousRight : fallbackRight;
  }
}

function populateComparisonBackendFilter() {
  const select = byId('benchmarkCompareBackend');
  const previous = select.value || '*';
  const backends = [...new Set((benchmarkComparison?.common_profiles || []).map(row => String(row.backend || '-')))].sort();
  select.replaceChildren();
  const all = document.createElement('option'); all.value='*'; all.textContent='All backends'; select.appendChild(all);
  backends.forEach(backend => { const option=document.createElement('option'); option.value=backend; option.textContent=backend; select.appendChild(option); });
  select.value = backends.includes(previous) ? previous : '*';
}

function renderBenchmarkComparison() {
  const holder = byId('benchmarkCompareCards');
  if (!benchmarkComparison) {
    holder.innerHTML = '';
    byId('benchmarkCompareStatus').textContent = 'Select two completed benchmark runs.';
    return;
  }
  const backend = byId('benchmarkCompareBackend').value;
  const sort = byId('benchmarkCompareSort').value;
  let rows = (benchmarkComparison.common_profiles || []).filter(row => backend === '*' || row.backend === backend);
  rows = [...rows].sort((a, b) => {
    const av = comparisonMetric(a, sort);
    const bv = comparisonMetric(b, sort);
    if (av == null && bv == null) return String(a.model_id).localeCompare(String(b.model_id));
    if (av == null) return 1;
    if (bv == null) return -1;
    const direction = (sort === 'median' || sort === 'p95') ? 1 : -1;
    const diff = (Number(av) - Number(bv)) * direction;
    return diff || String(a.model_id).localeCompare(String(b.model_id));
  });

  const metric = (row, name) => row.metrics?.[name] || {};
  const valuePair = (title, left, right, delta) => `
    <div class="compare-metric">
      <strong>${escapeHtml(title)}</strong>
      <div class="compare-values">
        <span><small>A</small>${escapeHtml(left)}</span>
        <span><small>B</small>${escapeHtml(right)}</span>
      </div>
      <div class="compare-delta">Δ ${escapeHtml(delta)}</div>
    </div>`;

  holder.innerHTML = rows.map(row => {
    const acc = metric(row, 'case_accuracy');
    const dec = metric(row, 'decision_accuracy');
    const median = metric(row, 'median_ms');
    const p95 = metric(row, 'p95_ms');
    const throughput = metric(row, 'throughput_decisions_per_s');
    const rank = row.rank || {};
    const precision = row.identity?.precision || row.identity?.quantization?.quantization || row.identity?.quantization?.kind || '';
    const rankLeft = rank.left == null ? '-' : `#${rank.left}`;
    const rankRight = rank.right == null ? '-' : `#${rank.right}`;
    const breakdownEntries = Object.entries(row.breakdown || {});
    const breakdown = breakdownEntries.length ? `<details style="margin-top:10px"><summary>Decision-type breakdown</summary><div class="table-scroll" style="margin-top:8px"><table><thead><tr><th>Type</th><th>Δ accuracy</th><th>Δ decision</th><th>Δ median</th><th>Δ P95</th></tr></thead><tbody>${breakdownEntries.map(([kind, metrics]) => `<tr><td>${escapeHtml(kind)}</td><td>${escapeHtml(formatComparisonMetricDelta(metrics.case_accuracy, 'accuracy'))}</td><td>${escapeHtml(formatComparisonMetricDelta(metrics.decision_accuracy, 'decision_accuracy'))}</td><td>${escapeHtml(formatComparisonMetricDelta(metrics.median_ms, 'median'))}</td><td>${escapeHtml(formatComparisonMetricDelta(metrics.p95_ms, 'p95'))}</td></tr>`).join('')}</tbody></table></div></details>` : '';
    return `<section class="compare-card">
      <div class="compare-card-head">
        <div class="compare-card-title">
          <strong>${escapeHtml(row.model_id ?? row.label ?? 'profile')}</strong>
          <small>${escapeHtml(row.engine ?? '')}${row.canonical_id ? ` · canonical identity verified` : ''}</small>
        </div>
        <div class="compare-card-tags"><span class="badge">${escapeHtml(row.backend ?? '-')}</span>${precision ? `<span class="badge">${escapeHtml(precision)}</span>` : ''}</div>
      </div>
      <div class="compare-metrics">
        ${valuePair('Accuracy', formatBenchmarkPercent(acc.left), formatBenchmarkPercent(acc.right), formatComparisonMetricDelta(acc, 'accuracy'))}
        ${valuePair('Decision accuracy', formatBenchmarkPercent(dec.left), formatBenchmarkPercent(dec.right), formatComparisonMetricDelta(dec, 'decision_accuracy'))}
        ${valuePair('Median latency', formatBenchmarkMs(median.left), formatBenchmarkMs(median.right), formatComparisonMetricDelta(median, 'median'))}
        ${valuePair('P95 latency', formatBenchmarkMs(p95.left), formatBenchmarkMs(p95.right), formatComparisonMetricDelta(p95, 'p95'))}
        ${valuePair('Throughput', throughput.left == null ? '-' : `${Number(throughput.left).toFixed(2)}/s`, throughput.right == null ? '-' : `${Number(throughput.right).toFixed(2)}/s`, formatComparisonMetricDelta(throughput, 'throughput'))}
        ${valuePair('Rank', rankLeft, rankRight, formatComparisonDelta(rank.shift, 'rank'))}
      </div>
      ${breakdown}
    </section>`;
  }).join('');

  const warnings = benchmarkComparison.warnings || [];
  byId('benchmarkCompareStatus').className = 'status';
  byId('benchmarkCompareStatus').textContent = `${rows.length} common profile${rows.length === 1 ? '' : 's'}. Δ = B − A.${warnings.length ? ` ${warnings.join(' ')}` : ''}`;
  const onlyA = (benchmarkComparison.only_in_left || []).map(row => row.model_id || row.label).join(', ') || 'none';
  const onlyB = (benchmarkComparison.only_in_right || []).map(row => row.model_id || row.label).join(', ') || 'none';
  const legacyA = (benchmarkComparison.legacy_unverified_left || []).length;
  const legacyB = (benchmarkComparison.legacy_unverified_right || []).length;
  const failedA = (benchmarkComparison.load_failed_left || []).length;
  const failedB = (benchmarkComparison.load_failed_right || []).length;
  byId('benchmarkCompareOnly').textContent = `Only in A: ${onlyA} · Only in B: ${onlyB}${legacyA || legacyB ? ` · Legacy profiles without canonical identity: A=${legacyA}, B=${legacyB}` : ''}${failedA || failedB ? ` · Runtime failed to load: A=${failedA}, B=${failedB}` : ''}`;
}


async function loadBenchmarkComparison() {
  const left = byId('benchmarkCompareLeft').value;
  const right = byId('benchmarkCompareRight').value;
  if (!left || !right) {
    benchmarkComparison = null;
    renderBenchmarkComparison();
    return;
  }
  if (left === right) {
    benchmarkComparison = null;
    byId('benchmarkCompareCards').innerHTML = '';
    byId('benchmarkCompareStatus').textContent = 'Choose two different benchmark runs.';
    return;
  }
  byId('benchmarkCompareStatus').textContent = 'Loading comparison...';
  try {
    const response = await fetch(`/v1/benchmarks/compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || JSON.stringify(body));
    benchmarkComparison = body;
    byId('benchmarkRawComparison').textContent = JSON.stringify(body, null, 2);
    populateComparisonBackendFilter();
    renderBenchmarkComparison();
  } catch (error) {
    benchmarkComparison = null;
    byId('benchmarkCompareCards').innerHTML = '';
    byId('benchmarkCompareStatus').textContent = error.message;
    byId('benchmarkCompareStatus').className = 'status error';
  }
}

async function refreshBenchmarks() {
  const empty = byId('benchmarkEmpty');
  const content = byId('benchmarkContent');
  const select = byId('benchmarkRunSelect');
  empty.textContent = 'Checking for benchmark runs...';
  empty.className = 'runtime-note';
  try {
    const response = await fetch('/v1/benchmarks');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    benchmarkRuns = data.runs || [];
    if (!benchmarkRuns.length) {
      content.classList.add('hidden');
      empty.classList.remove('hidden');
      empty.textContent = 'No benchmark runs found. Run `deqio benchmark` and refresh this section.';
      return;
    }

    empty.classList.add('hidden');
    content.classList.remove('hidden');
    const previous = benchmarkRunId || select.value;
    select.replaceChildren();
    benchmarkRuns.forEach(run => {
      const option = document.createElement('option');
      option.value = run.id;
      const created = new Date(run.created_at).toLocaleString();
      option.textContent = `${created} · ${run.suite_name || run.id} · ${run.models ?? '-'} model(s)`;
      select.appendChild(option);
    });
    select.value = benchmarkRuns.some(run => run.id === previous) ? previous : (data.latest || benchmarkRuns[0].id);
    await loadBenchmarkRun(select.value);
    populateBenchmarkComparisonSelectors();
    if (benchmarkRuns.filter(run => run.has_summary).length >= 2) await loadBenchmarkComparison();
  } catch (error) {
    content.classList.add('hidden');
    empty.classList.remove('hidden');
    empty.textContent = `Could not read benchmark history: ${error.message}`;
    empty.className = 'status error';
  }
}

function setBenchmarkView(view) {
  document.querySelectorAll('.benchmark-tab').forEach(button => button.classList.toggle('active', button.dataset.benchmarkView === view));
  byId('benchmarkSummaryView').classList.toggle('hidden', view !== 'summary');
  byId('benchmarkResultsView').classList.toggle('hidden', view !== 'results');
  byId('benchmarkCompareView').classList.toggle('hidden', view !== 'compare');
}

async function loadWatchSettings() {
  try {
    const data = await fetch('/v1/watch/settings').then(r => r.json());
    byId('watchAutoClear').value = String(data.auto_clear_minutes ?? 0);
  } catch (_) {}
}

async function setWatchAutoClear() {
  const minutes = Number(byId('watchAutoClear').value);
  const response = await fetch('/v1/watch/settings', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({auto_clear_minutes:minutes}),
  });
  if (!response.ok) {
    const data = await response.json();
    alert(data.detail || JSON.stringify(data));
  }
}

async function refreshHealth() {
  try {
    const health = await fetch('/health').then(r => r.json());
    byId('healthBadge').textContent = `health: ${health.status}`;
    const suspension = health.runtime_suspension;
    byId('runtimeSub').textContent = suspension
      ? `${health.engine} · ${health.model_id} · ${health.backend} · inference suspended for ${suspension.reason || 'benchmark'} (owner pid ${suspension.owner_pid || '?'})`
      : `${health.engine} · ${health.model_id} · ${health.backend} · ${health.model}`;
  } catch (_) {
    byId('healthBadge').textContent = 'health: unavailable';
  }
}

async function refreshStats() {
  try {
    const [stats, recent] = await Promise.all([
      fetch('/v1/stats').then(r => r.json()),
      fetch('/v1/recent').then(r => r.json()),
    ]);
    byId('requests').textContent = stats.requests;
    byId('decisions').textContent = stats.decisions;
    byId('cacheClears').textContent = stats.cache_clears;
    byId('p50').textContent = stats.latency_ms.p50 == null ? '-' : `${stats.latency_ms.p50.toFixed(1)} ms`;
    byId('p95').textContent = stats.latency_ms.p95 == null ? '-' : `${stats.latency_ms.p95.toFixed(1)} ms`;
    byId('rows').innerHTML = recent.map(x => `
      <tr>
        <td>${escapeHtml(new Date(x.timestamp).toLocaleTimeString())}</td>
        <td>${escapeHtml(x.engine ?? '-')}</td>
        <td>${escapeHtml(x.backend ?? '-')}</td>
        <td>${escapeHtml(x.mode)}</td>
        <td>${escapeHtml(x.question)}</td>
        <td>${escapeHtml(x.decision ?? '-')}</td>
        <td>${escapeHtml(x.latency_ms == null ? '-' : `${x.latency_ms} ms`)}</td>
        <td>${escapeHtml(x.cache_hit == null ? '-' : x.cache_hit)}</td>
      </tr>`).join('');
  } catch (_) {}
}

document.querySelectorAll('.endpoint-tab').forEach(button => button.addEventListener('click', () => setEndpoint(button.dataset.endpoint)));
byId('addChoiceOption').addEventListener('click', () => addOption(byId('choiceOptions')));
byId('addSharedDecision').addEventListener('click', () => addSharedDecision());
byId('sendRequest').addEventListener('click', sendRequest);
byId('clearCache').addEventListener('click', clearRuntimeCache);
byId('watchAutoClear').addEventListener('change', setWatchAutoClear);
byId('activateModel').addEventListener('click', activateSelectedModel);
byId('modelSelect').addEventListener('change', () => { byId('activateModel').disabled = byId('modelSelect').value === activeModelKey; });
byId('refreshBenchmarks').addEventListener('click', refreshBenchmarks);
byId('benchmarkRunSelect').addEventListener('change', event => loadBenchmarkRun(event.target.value));
document.querySelectorAll('.benchmark-tab').forEach(button => button.addEventListener('click', () => setBenchmarkView(button.dataset.benchmarkView)));
['benchmarkSummaryModel', 'benchmarkSummaryType', 'benchmarkSummarySort', 'benchmarkSummaryDirection'].forEach(id => byId(id).addEventListener('change', renderBenchmarkSummary));
['benchmarkResultModel', 'benchmarkResultType', 'benchmarkResultStatus', 'benchmarkResultSort'].forEach(id => byId(id).addEventListener('change', renderBenchmarkResults));
['benchmarkCompareLeft', 'benchmarkCompareRight'].forEach(id => byId(id).addEventListener('change', loadBenchmarkComparison));
byId('benchmarkCompareBackend').addEventListener('change', renderBenchmarkComparison);
byId('benchmarkCompareSort').addEventListener('change', renderBenchmarkComparison);
document.addEventListener('input', updatePreview);
document.addEventListener('change', updatePreview);

setEndpoint('noul');
setBenchmarkView('summary');
refreshHealth();
refreshStats();
loadWatchSettings();
refreshModels();
refreshBenchmarks();
setInterval(refreshStats, 2000);
</script>
</body>
</html>
"""

WATCH_DASHBOARD = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deqio Watch</title>
<style>
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 1360px; margin: 28px auto; padding: 0 20px 48px;
  background: Canvas; color: CanvasText;
}
h1 { margin: 0 0 4px; font-size: 28px; }
.sub { opacity: .66; margin: 0; }
.toolbar, .actions, .filters { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.toolbar { justify-content:space-between; margin-bottom:18px; }
.actions { justify-content:flex-end; }
button, select, input, a.button {
  font: inherit; border:1px solid color-mix(in srgb, CanvasText 18%, transparent);
  border-radius:8px; background:Canvas; color:CanvasText; padding:8px 11px;
}
button { cursor:pointer; }
a.button { text-decoration:none; display:inline-flex; align-items:center; }
button.danger { border-color:#a33; }
.panel { border:1px solid color-mix(in srgb, CanvasText 14%, transparent); border-radius:12px; padding:16px; margin-bottom:16px; background:color-mix(in srgb, Canvas 97%, CanvasText 3%); }
.cards { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin-bottom:16px; }
.card { border:1px solid color-mix(in srgb, CanvasText 14%, transparent); border-radius:10px; padding:13px; }
.card small { display:block; opacity:.6; margin-bottom:5px; }
.card strong { font-size:19px; }
.runtime { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; }
.runtime div { min-width:0; }
.runtime small { display:block; opacity:.6; margin-bottom:3px; }
.runtime code { overflow-wrap:anywhere; }
.filters { margin-bottom:12px; }
.filters label { font-size:12px; opacity:.7; }
.filters select { min-width:160px; }
.table-scroll { overflow:auto; max-height:58vh; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:9px; border-bottom:1px solid color-mix(in srgb, CanvasText 10%, transparent); white-space:nowrap; }
th { position:sticky; top:0; background:Canvas; opacity:.8; z-index:1; }
tbody tr { cursor:pointer; }
tbody tr:hover { background:color-mix(in srgb, CanvasText 6%, transparent); }
.status-ok { font-weight:700; }
.status-error { font-weight:700; color:#b23b3b; }
.empty { opacity:.65; padding:20px 4px; }
dialog { width:min(1100px,94vw); max-height:90vh; border:1px solid color-mix(in srgb, CanvasText 18%, transparent); border-radius:14px; background:Canvas; color:CanvasText; padding:0; }
dialog::backdrop { background:rgba(0,0,0,.45); }
.dialog-head { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:16px 18px; border-bottom:1px solid color-mix(in srgb, CanvasText 12%, transparent); position:sticky; top:0; background:Canvas; }
.dialog-body { padding:18px; overflow:auto; }
.detail-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin-bottom:14px; }
.detail-grid div { min-width:0; border:1px solid color-mix(in srgb, CanvasText 12%, transparent); border-radius:9px; padding:10px; }
.detail-grid small { display:block; opacity:.6; margin-bottom:4px; }
.runtime-id { display:block; width:100%; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; border:0; background:transparent; color:inherit; padding:0; text-align:left; font:inherit; font-weight:700; cursor:copy; }
.runtime-id:focus-visible { outline:2px solid CanvasText; outline-offset:2px; border-radius:3px; }
.json-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
pre { margin:0; padding:12px; border-radius:9px; background:color-mix(in srgb, Canvas 94%, CanvasText 6%); overflow:auto; max-height:52vh; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
.session-note { font-size:12px; opacity:.65; margin-top:8px; }
@media (max-width:900px) { .cards { grid-template-columns:repeat(2,minmax(0,1fr)); } .runtime,.detail-grid,.json-grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="toolbar">
  <div>
    <h1>Deqio Watch</h1>
    <p class="sub">Disk-backed request/response inspection for every decision executed in this server session.</p>
  </div>
  <div class="actions">
    <span id="healthBadge">health: ...</span>
    <a class="button" href="/ui">Back to UI</a>
    <button id="clearSession" class="danger" type="button">Clear session</button>
  </div>
</div>

<section class="panel">
  <div class="runtime">
    <div><small>Active model</small><code id="model">-</code></div>
    <div><small>Backend</small><code id="backend">-</code></div>
    <div><small>Session</small><code id="sessionId">-</code></div>
    <div><small>Started</small><code id="sessionStarted">-</code></div>
    <div><small>History files</small><code id="historyFiles">0</code></div>
    <div><small>Inference</small><code id="inferenceState">-</code></div>
  </div>
  <div class="session-note">Full Watch payloads are stored in temporary <code>.deqio/watch/</code> JSONL files, rotating at 10,000 records per file. They are deleted on server restart, manual Clear, or the configured automatic cleanup interval. Model switches stay in the same server session and each row keeps its own model identity.</div>
</section>

<div class="cards">
  <div class="card"><small>Requests</small><strong id="requests">0</strong></div>
  <div class="card"><small>Decisions</small><strong id="decisions">0</strong></div>
  <div class="card"><small>Errors</small><strong id="errors">0</strong></div>
  <div class="card"><small>P50 latency</small><strong id="p50">-</strong></div>
  <div class="card"><small>P95 latency</small><strong id="p95">-</strong></div>
</div>

<section class="panel">
  <div class="filters">
    <label for="endpointFilter">Endpoint</label>
    <select id="endpointFilter">
      <option value="*">All model requests</option>
      <option value="/v1/noul">/v1/noul</option>
      <option value="/v1/choice">/v1/choice</option>
      <option value="/v1/decision">/v1/decision</option>
      <option value="/v1/shared">/v1/shared</option>
      <option value="/v1/score">/v1/score</option>
      <option value="/v1/multi">/v1/multi</option>
      <option value="/v1/act">/v1/act</option>
      <option value="/v1/soam">/v1/soam</option>
      <option value="/v1/systemone">/v1/systemone</option>
    </select>
    <label for="statusFilter">Status</label>
    <select id="statusFilter">
      <option value="*">All</option>
      <option value="ok">Success</option>
      <option value="error">Errors</option>
    </select>
    <label for="sourceFilter">Source</label>
    <select id="sourceFilter"><option value="*">API + benchmark</option><option value="api">API</option><option value="benchmark">Benchmark</option></select>
    <label for="modelFilter">Model</label>
    <select id="modelFilter"><option value="*">All models</option></select>
    <label for="watchAutoClear">Auto clear</label>
    <select id="watchAutoClear"><option value="0">off</option><option value="15">15 min</option><option value="30">30 min</option><option value="60">1 h</option><option value="120">2 h</option><option value="240">4 h</option></select>
    <label><input id="autoRefresh" type="checkbox" checked style="width:auto"> auto refresh</label>
    <button id="refresh" type="button">Refresh</button>
  </div>
  <div class="table-scroll">
    <table>
      <thead><tr><th>Time</th><th>Source</th><th>Model</th><th>Endpoint</th><th>Status</th><th>Request ID</th><th>Mode</th><th>Decision</th><th>Top p</th><th>Latency</th><th>Tokens</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div id="empty" class="empty">No requests in this server session yet.</div>
  </div>
  <div class="actions" style="margin-top:12px"><span id="paginationNote" class="sub"></span><button id="loadOlder" type="button">Load older</button></div>
</section>

<dialog id="detailDialog">
  <div class="dialog-head">
    <div><strong id="detailTitle">Request</strong><div class="sub" id="detailSubtitle"></div></div>
    <button id="closeDetail" type="button">Close</button>
  </div>
  <div class="dialog-body">
    <div class="detail-grid">
      <div><small>Source</small><strong id="detailSource">-</strong></div>
      <div><small>Model</small><strong id="detailModel">-</strong></div>
      <div><small>Runtime instance</small><button id="detailRuntime" class="runtime-id" type="button" title="Click to copy runtime instance ID">-</button></div>
      <div><small>Latency</small><strong id="detailLatency">-</strong></div>
      <div><small>Input tokens</small><strong id="detailTokens">-</strong></div>
    </div>
    <div class="json-grid">
      <div><h3>Request</h3><pre id="detailRequest"></pre></div>
      <div><h3>Response</h3><pre id="detailResponse"></pre></div>
    </div>
  </div>
</dialog>

<script>
const byId = id => document.getElementById(id);
let events = [];
let totalEvents = 0;
const pageSize = 500;
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
const fmtMs = value => value == null ? '-' : `${Number(value).toFixed(1)} ms`;
const fmtP = value => value == null ? '-' : `${(Number(value) * 100).toFixed(1)}%`;

function refreshModelFilter() {
  const select = byId('modelFilter');
  const previous = select.value || '*';
  const values = [...new Set(events.map(row => `${row.model_id || '-'}:${row.backend || '-'}`))].sort();
  select.replaceChildren();
  const all = document.createElement('option'); all.value='*'; all.textContent='All models'; select.appendChild(all);
  values.forEach(value => { const option=document.createElement('option'); option.value=value; option.textContent=value; select.appendChild(option); });
  select.value = values.includes(previous) ? previous : '*';
}

function renderRows() {
  const endpoint = byId('endpointFilter').value;
  const status = byId('statusFilter').value;
  const source = byId('sourceFilter').value;
  const model = byId('modelFilter').value;
  const filtered = events.filter(row =>
    (endpoint === '*' || row.endpoint === endpoint) &&
    (source === '*' || row.source === source) &&
    (model === '*' || `${row.model_id || '-'}:${row.backend || '-'}` === model) &&
    (status === '*' || (status === 'ok' ? row.status_code < 400 : row.status_code >= 400))
  );
  byId('empty').style.display = filtered.length ? 'none' : 'block';
  byId('rows').innerHTML = filtered.map(row => `
    <tr data-id="${escapeHtml(row.event_id)}">
      <td>${escapeHtml(new Date(row.timestamp).toLocaleTimeString())}</td>
      <td>${escapeHtml(row.source || 'api')}</td>
      <td>${escapeHtml(`${row.model_id || '-'}:${row.backend || '-'}`)}</td>
      <td><code>${escapeHtml(row.endpoint)}</code></td>
      <td class="${row.status_code < 400 ? 'status-ok' : 'status-error'}">${escapeHtml(row.status_code)}</td>
      <td>${escapeHtml(row.request_id || '-')}</td>
      <td>${escapeHtml(row.mode || '-')}</td>
      <td>${escapeHtml(row.decision || '-')}</td>
      <td>${escapeHtml(fmtP(row.top_probability))}</td>
      <td>${escapeHtml(fmtMs(row.latency_ms))}</td>
      <td>${escapeHtml(row.input_tokens ?? '-')}</td>
    </tr>`).join('');
  byId('rows').querySelectorAll('tr').forEach(row => row.addEventListener('click', () => openDetail(row.dataset.id)));
  byId('paginationNote').textContent = `Loaded ${events.length} of ${totalEvents} request(s)`;
  byId('loadOlder').disabled = events.length >= totalEvents;
}

let refreshInFlight = false;

function showError(prefix, error) {
  byId('healthBadge').textContent = `${prefix}: ${error && error.message ? error.message : error}`;
}

async function fetchWatch(offset = 0, limit = pageSize) {
  const response = await fetch(`/v1/watch?limit=${limit}&offset=${offset}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function refresh() {
  // The 2 s timer must not pile up requests behind a slow refresh.
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    // Re-read at least as many rows as are shown, so pages loaded with
    // "Load older" survive the automatic refresh (the server caps the limit).
    const [watch, health] = await Promise.all([
      fetchWatch(0, Math.max(pageSize, events.length)),
      fetch('/health').then(r => r.json()),
    ]);
    events = watch.events || [];
    totalEvents = watch.pagination?.total ?? events.length;
    const session = watch.session || {};
    byId('model').textContent = `${health.engine || '-'} · ${health.model_id || '-'}`;
    byId('backend').textContent = health.backend || '-';
    byId('sessionId').textContent = session.id || '-';
    byId('sessionStarted').textContent = session.started_at ? new Date(session.started_at).toLocaleString() : '-';
    byId('historyFiles').textContent = `${session.storage?.files ?? 0} × ≤${session.storage?.max_lines_per_file ?? 10000}`;
    byId('requests').textContent = session.requests ?? 0;
    byId('decisions').textContent = session.decisions ?? 0;
    byId('errors').textContent = session.errors ?? 0;
    byId('p50').textContent = fmtMs(session.latency_ms?.p50);
    byId('p95').textContent = fmtMs(session.latency_ms?.p95);
    byId('watchAutoClear').value = String(session.auto_clear_minutes ?? 0);
    byId('healthBadge').textContent = `health: ${health.status}`;
    const suspension = health.runtime_suspension || session.runtime_suspension;
    byId('inferenceState').textContent = suspension
      ? `suspended · ${suspension.reason || 'benchmark'} · pid ${suspension.owner_pid || '?'}`
      : 'active';
    refreshModelFilter();
    renderRows();
  } catch (error) {
    showError('watch unavailable', error);
  } finally {
    refreshInFlight = false;
  }
}

async function loadOlder() {
  try {
    const watch = await fetchWatch(events.length);
    events = events.concat(watch.events || []);
    totalEvents = watch.pagination?.total ?? totalEvents;
    refreshModelFilter();
    renderRows();
  } catch (error) {
    showError('load older failed', error);
  }
}

async function openDetail(eventId) {
  try {
    const response = await fetch(`/v1/watch/${encodeURIComponent(eventId)}`);
    const item = await response.json();
    if (!response.ok) { alert(item.detail || JSON.stringify(item)); return; }
    byId('detailTitle').textContent = `${item.endpoint} · ${item.status_code}`;
    byId('detailSubtitle').textContent = `${new Date(item.timestamp).toLocaleString()} · ${item.request_id || '-'}`;
    byId('detailSource').textContent = item.source || 'api';
    byId('detailModel').textContent = `${item.engine} · ${item.model_id} · ${item.backend}`;
    const runtimeInstance = item.runtime_instance_id || '-';
    byId('detailRuntime').textContent = runtimeInstance;
    byId('detailRuntime').dataset.copyValue = item.runtime_instance_id || '';
    byId('detailRuntime').title = item.runtime_instance_id
      ? `${item.runtime_instance_id}\nClick to copy`
      : 'Runtime instance ID unavailable';
    byId('detailLatency').textContent = fmtMs(item.latency_ms);
    byId('detailTokens').textContent = item.input_tokens ?? '-';
    byId('detailRequest').textContent = JSON.stringify(item.request, null, 2);
    byId('detailResponse').textContent = JSON.stringify(item.response, null, 2);
    byId('detailDialog').showModal();
  } catch (error) {
    showError('detail unavailable', error);
  }
}

async function clearSession() {
  if (!confirm('Delete all temporary Watch history files for this server session?')) return;
  try {
    const response = await fetch('/v1/watch/clear', {method:'POST'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    showError('clear failed', error);
    return;
  }
  await refresh();
}

byId('endpointFilter').addEventListener('change', renderRows);
byId('statusFilter').addEventListener('change', renderRows);
byId('sourceFilter').addEventListener('change', renderRows);
byId('modelFilter').addEventListener('change', renderRows);
byId('refresh').addEventListener('click', refresh);
byId('loadOlder').addEventListener('click', loadOlder);
byId('watchAutoClear').addEventListener('change', async () => {
  try {
    const response = await fetch('/v1/watch/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({auto_clear_minutes:Number(byId('watchAutoClear').value)})});
    if (!response.ok) { const data=await response.json(); alert(data.detail || JSON.stringify(data)); }
  } catch (error) {
    showError('settings update failed', error);
  }
  await refresh();
});
byId('clearSession').addEventListener('click', clearSession);
byId('closeDetail').addEventListener('click', () => byId('detailDialog').close());
byId('detailRuntime').addEventListener('click', async () => {
  const value = byId('detailRuntime').dataset.copyValue || '';
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
    byId('detailRuntime').title = `${value}\nCopied`;
  } catch (error) {
    window.prompt('Copy runtime instance ID:', value);
  }
});
setInterval(() => { if (byId('autoRefresh').checked) refresh(); }, 2000);
refresh();
</script>
</body>
</html>
"""
