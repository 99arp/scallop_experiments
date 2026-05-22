import "./styles.css";
import { renderGantt } from "./gantt";
import type { EcSnapshot, ScallopQueryResult, Series, Snapshot } from "./types";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app mount point");
}

let pollMs = 1000;
let timer: number | undefined;
let ecTimer: number | undefined;

const DEFAULT_SCALLOP_QUERY =
  `rel ruleSatisfied(name, start, end) =\n  holdsFor("rule", name, start, end)\n\nquery ruleSatisfied`;

function buildShell(root: HTMLDivElement): void {
  root.innerHTML = `
    <header class="topbar">
      <div class="brand">STL <span class="dot">·</span> oPIEC <span class="brand-sub">live monitor</span></div>
      <div class="status status-bad" id="status">connecting…</div>
    </header>
    <main>
      <section class="panel">
        <div class="panel-head">
          <h2>Event Calculus intervals</h2>
          <div class="meta" id="ec-meta"></div>
        </div>
        <div class="legend">
          <span class="chip chip-rule">rule holdsFor</span>
          <span class="chip chip-group">group holdsFor</span>
          <span class="chip chip-now">now</span>
        </div>
        <div class="gantt-wrap" id="ec-gantt"><div class="empty">waiting for EC data…</div></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <h2>Scallop query</h2>
          <div class="meta" id="scallop-meta"></div>
        </div>
        <div class="scallop-controls">
          <label class="scallop-field">
            <span class="qc-label">Scallop program</span>
            <textarea id="scallop-query" class="qc-input scallop-textarea" rows="4" autocomplete="off" spellcheck="false"></textarea>
          </label>
          <button id="scallop-run" class="qc-btn scallop-run">Run</button>
        </div>
        <div class="kb-results" id="scallop-results">
          <div class="kb-empty">Write a Scallop query against the current holdsFor intervals.</div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head"><h2>STL semantic graph</h2></div>
        <div class="graph-wrap"><img id="graph" alt="STL semantic graph" /></div>
      </section>
    </main>`;

  wireScallopQueryControls();
}

function wireScallopQueryControls(): void {
  const textarea = document.querySelector<HTMLTextAreaElement>("#scallop-query");
  const run = document.querySelector<HTMLButtonElement>("#scallop-run");
  const results = document.querySelector("#scallop-results");
  const meta = document.querySelector("#scallop-meta");

  if (textarea) {
    textarea.value = DEFAULT_SCALLOP_QUERY;
  }

  const execute = async (): Promise<void> => {
    const program = textarea?.value.trim() ?? "";
    if (!program) return;
    if (run) run.disabled = true;
    if (meta) meta.textContent = "running...";
    if (results) results.innerHTML = `<div class="kb-empty">Evaluating…</div>`;
    let result: ScallopQueryResult;
    try {
      const res = await fetch("/api/scallop-query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: program }),
      });
      result = (await res.json()) as ScallopQueryResult;
    } catch (error) {
      result = { ok: false, error: error instanceof Error ? error.message : String(error) };
    } finally {
      if (run) run.disabled = false;
    }
    if (results) results.innerHTML = renderScallopResult(result);
    if (meta) {
      meta.textContent = result.ok
        ? `${result.answer_count ?? result.answers?.length ?? 0} answer${(result.answer_count ?? 0) === 1 ? "" : "s"}`
        : "error";
    }
  };

  run?.addEventListener("click", () => void execute());
  textarea?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      void execute();
    }
  });
}

function esc(s: string): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderScallopResult(result: ScallopQueryResult): string {
  if (!result.ok) {
    return `<div class="kb-error">${esc(result.error ?? "query failed")}</div>`;
  }
  const answers = result.answers ?? [];
  if (answers.length === 0) {
    return `<div class="kb-empty">No answers.</div>`;
  }
  const rows = answers
    .map((a) => {
      const tupleStr = `(${a.tuple.map((v) => (typeof v === "string" ? `"${esc(v)}"` : esc(String(v)))).join(", ")})`;
      return `<tr>
        <td class="kb-prob">${a.probability.toFixed(9)}</td>
        <td class="kb-term">${tupleStr}</td>
      </tr>`;
    })
    .join("");
  const count = result.answer_count ?? answers.length;
  return `
    <div class="kb-summary">${count} answer${count === 1 ? "" : "s"}</div>
    <div class="kb-table-wrap">
      <table class="kb-table">
        <thead><tr><th>Probability</th><th>Tuple</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function ecIntervalsToSeries(snap: EcSnapshot): Series[] {
  const byFluent = new Map<string, Series>();
  for (const iv of snap.intervals) {
    const key = `${iv.source_kind}:${iv.name}`;
    const existing = byFluent.get(key);
    if (existing) {
      existing.intervals.push([iv.start, iv.end]);
      existing.probability = Math.min(existing.probability ?? 1, iv.p_min);
    } else {
      byFluent.set(key, {
        series_id: key,
        source_kind: iv.source_kind,
        source_name: iv.name,
        source_signal: null,
        fluent: null,
        value: null,
        probability: iv.p_min,
        declaration_chain: null,
        intervals: [[iv.start, iv.end]],
      });
    }
  }
  return Array.from(byFluent.values());
}

function renderEcSnap(snap: EcSnapshot): void {
  const meta = document.querySelector("#ec-meta");
  if (meta) {
    meta.textContent = `sample ${snap.sample_index ?? "—"} · ${snap.intervals.length} intervals · updated ${snap.updated_at}`;
  }
  const gantt = document.querySelector("#ec-gantt");
  if (!gantt) return;
  if (snap.intervals.length === 0) {
    gantt.innerHTML = `<div class="empty">No holdsFor intervals yet.</div>`;
    return;
  }
  const series = ecIntervalsToSeries(snap);
  const syntheticSnap: Snapshot = {
    updated_at: snap.updated_at,
    status: "ready",
    graph_svg: "",
    sample_index: snap.sample_index,
    timestamp_ns: null,
    poll_ms: pollMs,
    axis: snap.axis,
    series,
  };
  gantt.innerHTML = renderGantt(syntheticSnap);
}

function setStatus(text: string, ok: boolean): void {
  const el = document.querySelector("#status");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("status-ok", ok);
  el.classList.toggle("status-bad", !ok);
}

async function tick(): Promise<void> {
  try {
    const res = await fetch(`./stl_live.json?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const snap = (await res.json()) as Snapshot;
    if (typeof snap.poll_ms === "number" && snap.poll_ms > 0) pollMs = snap.poll_ms;
    const graph = document.querySelector<HTMLImageElement>("#graph");
    if (graph && snap.graph_svg) graph.src = `./${snap.graph_svg}?t=${Date.now()}`;
    setStatus(snap.status, true);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setStatus(`no live data — is the monitor running? (${message})`, false);
  } finally {
    timer = window.setTimeout(() => void tick(), pollMs);
  }
}

async function tickEc(): Promise<void> {
  try {
    const res = await fetch(`./ec_live_intervals.json?t=${Date.now()}`, { cache: "no-store" });
    if (res.ok) renderEcSnap((await res.json()) as EcSnapshot);
  } catch {
    // EC data is optional — silently skip if not available
  } finally {
    ecTimer = window.setTimeout(() => void tickEc(), pollMs);
  }
}

buildShell(app);
void tick();
void tickEc();

window.addEventListener("beforeunload", () => {
  if (timer !== undefined) window.clearTimeout(timer);
  if (ecTimer !== undefined) window.clearTimeout(ecTimer);
});
