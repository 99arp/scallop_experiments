import type { KnowledgeBaseQueryResult } from "./types";

export const DEFAULT_KB_QUERY = "opiec_interval_active(_, _, _, _)";

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function runKnowledgeBaseQuery(
  query: string
): Promise<KnowledgeBaseQueryResult> {
  const response = await fetch("/api/problog-query", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const payload = (await response.json()) as KnowledgeBaseQueryResult;
  if (!response.ok) {
    return {
      ok: false,
      error: payload.error ?? `HTTP ${response.status}`,
    };
  }
  return payload;
}

export function renderKnowledgeBaseResult(
  result: KnowledgeBaseQueryResult | null
): string {
  if (result === null) {
    return `<div class="kb-empty">Run a ProbLog query against the exported knowledge base.</div>`;
  }

  if (!result.ok) {
    const error = esc(result.error ?? "query failed");
    const type = result.error_type ? `<span class="kb-error-type">${esc(result.error_type)}</span>` : "";
    return `<div class="kb-error">${type}${error}</div>`;
  }

  const answers = result.answers ?? [];
  if (answers.length === 0) {
    return `<div class="kb-empty">No answers.</div>`;
  }

  const rows = answers
    .map((answer) => {
      return `<tr>
        <td class="kb-term">${esc(answer.term)}</td>
        <td class="kb-prob">${answer.probability.toFixed(10)}</td>
      </tr>`;
    })
    .join("");
  const count = result.answer_count ?? answers.length;
  const suffix = result.truncated ? `, showing ${answers.length}` : "";
  return `
    <div class="kb-summary">${count} answer${count === 1 ? "" : "s"}${suffix}</div>
    <div class="kb-table-wrap">
      <table class="kb-table">
        <thead><tr><th>Term</th><th>Probability</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}
