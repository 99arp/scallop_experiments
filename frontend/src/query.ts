import type { Series } from "./types";

export interface FactQuery {
  search: string;
  kind: "all" | "rule" | "group";
  minProb: number;
  atStep: number | null;
}

export const DEFAULT_QUERY: FactQuery = {
  search: "",
  kind: "all",
  minProb: 0,
  atStep: null,
};

export function isQueryActive(q: FactQuery): boolean {
  return (
    q.search.trim() !== "" ||
    q.kind !== "all" ||
    q.minProb > 0 ||
    q.atStep !== null
  );
}

function activeAtStep(s: Series, step: number): boolean {
  return s.intervals.some(([a, b]) => step >= a && step < b);
}

export function filterSeries(series: Series[], q: FactQuery): Series[] {
  return series.filter((s) => {
    if (q.kind !== "all" && s.source_kind !== q.kind) return false;
    if (q.minProb > 0 && (s.probability ?? 0) < q.minProb) return false;
    if (q.atStep !== null && !activeAtStep(s, q.atStep)) return false;
    if (q.search.trim()) {
      const term = q.search.trim().toLowerCase();
      const hay = [s.source_name, s.fluent, s.series_id]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!hay.includes(term)) return false;
    }
    return true;
  });
}

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function renderQueryResults(
  filtered: Series[],
  total: number,
  q: FactQuery
): string {
  if (!isQueryActive(q)) {
    return `<div class="qr-empty">Enter a filter above to query facts.</div>`;
  }

  if (filtered.length === 0) {
    return `<div class="qr-empty">No facts match.</div>`;
  }

  const rows = filtered
    .map((s) => {
      const name = esc(s.source_name ?? s.series_id);
      const kind = s.source_kind ?? "—";
      const prob = s.probability !== null ? s.probability.toFixed(5) : "—";
      const ivText = s.intervals.map(([a, b]) => `[${a}, ${b})`).join("  ");
      const hit =
        q.atStep !== null ? activeAtStep(s, q.atStep) : null;
      const hitClass = hit === true ? " qr-hit" : hit === false ? " qr-miss" : "";
      const kindClass = kind === "group" ? "kind-group" : "kind-rule";
      return `<tr class="${hitClass.trim()}">
        <td><span class="kind-badge ${kindClass}">${kind}</span></td>
        <td class="qr-name">${name}</td>
        <td class="qr-prob">${prob}</td>
        <td class="qr-ivs">${esc(ivText)}</td>
      </tr>`;
    })
    .join("");

  return `
    <div class="qr-summary">${filtered.length} of ${total} facts</div>
    <div class="qr-table-wrap">
      <table class="qr-table">
        <thead><tr>
          <th>Kind</th><th>Name</th><th>Probability</th><th>Intervals</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}
