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
  <h2>Playground</h2>
  <div class="tabs">
    <button class="tab active" data-endpoint="noul" type="button">Noul</button>
    <button class="tab" data-endpoint="choice" type="button">Choice</button>
    <button class="tab" data-endpoint="shared" type="button">Shared</button>
  </div>

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
    </div>

    <div id="benchmarkSummaryView">
      <div class="benchmark-controls">
        <div><label for="benchmarkSummaryModel">Model</label><select id="benchmarkSummaryModel"></select></div>
        <div><label for="benchmarkSummaryType">Type</label><select id="benchmarkSummaryType">
          <option value="overall">Overall</option><option value="noul">Noul</option><option value="choice">Choice</option><option value="shared">Shared</option><option value="*">All rows</option>
        </select></div>
        <div><label for="benchmarkSummarySort">Sort by</label><select id="benchmarkSummarySort">
          <option value="case_accuracy">Accuracy</option><option value="decision_accuracy">Decision accuracy</option><option value="mean_ms">Mean latency</option><option value="median_ms">Median latency</option><option value="p95_ms">P95 latency</option><option value="load_ms">Model load time</option><option value="cases">Cases</option>
        </select></div>
        <div><label for="benchmarkSummaryDirection">Order</label><select id="benchmarkSummaryDirection">
          <option value="desc">Highest first</option><option value="asc">Lowest / fastest first</option>
        </select></div>
      </div>
      <div id="benchmarkSummaryStatus" class="status"></div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Model</th><th>Engine</th><th>Type</th><th>Cases</th><th>Passed</th><th>Accuracy</th><th>Decision acc.</th><th>Mean</th><th>Median</th><th>P95</th><th>Load</th></tr></thead>
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
let benchmarkRunId = '';
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
  document.querySelectorAll('.tab').forEach(button => button.classList.toggle('active', button.dataset.endpoint === next));
  byId('singleFields').classList.toggle('hidden', next === 'shared');
  byId('choiceFields').classList.toggle('hidden', next !== 'choice');
  byId('sharedFields').classList.toggle('hidden', next !== 'shared');
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
      <div class="meta">tokens=${escapeHtml(result.input_tokens)} · total=${escapeHtml(timing.total_ms ?? '-')} ms · cache=${escapeHtml(timing.cache_hit ?? '-')}</div>
      <details><summary>Option logits</summary><pre>${escapeHtml(JSON.stringify(result.option_logits || {}, null, 2))}</pre></details>
    </div>`;
}

function renderResponse(data) {
  byId('responseJson').textContent = JSON.stringify(data, null, 2);
  if (Array.isArray(data.results)) {
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
    if (endpoint !== 'shared' && !payload.question) throw new Error('Question cannot be empty.');
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
  byId('benchmarkSummaryRows').innerHTML = rows.map(row => `
    <tr>
      <td>${escapeHtml(`${row.model_id}:${row.backend}`)}</td>
      <td>${escapeHtml(row.engine ?? '-')}</td>
      <td>${escapeHtml(row.type === 'overall' ? 'all' : row.type)}</td>
      <td>${escapeHtml(row.cases ?? '-')}</td>
      <td>${escapeHtml(row.passed_cases ?? '-')}</td>
      <td>${escapeHtml(formatBenchmarkPercent(row.case_accuracy))}</td>
      <td>${escapeHtml(formatBenchmarkPercent(row.decision_accuracy))}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.mean_ms))}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.median_ms))}</td>
      <td>${escapeHtml(formatBenchmarkMs(row.p95_ms))}</td>
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
      empty.textContent = 'No benchmark runs found. Run `uv run deqio benchmark` and refresh this section.';
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
}

async function refreshHealth() {
  try {
    const health = await fetch('/health').then(r => r.json());
    byId('healthBadge').textContent = `health: ${health.status}`;
    byId('runtimeSub').textContent = `${health.engine} · ${health.model_id} · ${health.backend} · ${health.model}`;
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

document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => setEndpoint(button.dataset.endpoint)));
byId('addChoiceOption').addEventListener('click', () => addOption(byId('choiceOptions')));
byId('addSharedDecision').addEventListener('click', () => addSharedDecision());
byId('sendRequest').addEventListener('click', sendRequest);
byId('clearCache').addEventListener('click', clearRuntimeCache);
byId('activateModel').addEventListener('click', activateSelectedModel);
byId('modelSelect').addEventListener('change', () => { byId('activateModel').disabled = byId('modelSelect').value === activeModelKey; });
byId('refreshBenchmarks').addEventListener('click', refreshBenchmarks);
byId('benchmarkRunSelect').addEventListener('change', event => loadBenchmarkRun(event.target.value));
document.querySelectorAll('.benchmark-tab').forEach(button => button.addEventListener('click', () => setBenchmarkView(button.dataset.benchmarkView)));
['benchmarkSummaryModel', 'benchmarkSummaryType', 'benchmarkSummarySort', 'benchmarkSummaryDirection'].forEach(id => byId(id).addEventListener('change', renderBenchmarkSummary));
['benchmarkResultModel', 'benchmarkResultType', 'benchmarkResultStatus', 'benchmarkResultSort'].forEach(id => byId(id).addEventListener('change', renderBenchmarkResults));
document.addEventListener('input', updatePreview);
document.addEventListener('change', updatePreview);

setEndpoint('noul');
setBenchmarkView('summary');
refreshHealth();
refreshStats();
refreshModels();
refreshBenchmarks();
setInterval(refreshStats, 2000);
</script>
</body>
</html>
"""
