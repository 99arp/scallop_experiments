import type { Snapshot } from "./types";

const LABEL_W = 240;
const ROW_H = 34;
const BAR_H = 18;
const TOP = 34;
const PLOT_W = 1000;
const RIGHT_PAD = 24;
const BOTTOM = 28;

const XML_ESCAPES: Record<string, string> = {
  "<": "&lt;",
  ">": "&gt;",
  "&": "&amp;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeXml(value: string): string {
  return value.replace(/[<>&"']/g, (char) => XML_ESCAPES[char] ?? char);
}

export function renderGantt(snap: Snapshot): string {
  const start = snap.axis.start;
  const end = snap.axis.end;
  const span = Math.max(1, end - start);
  const rows = snap.series;

  const svgW = LABEL_W + PLOT_W + RIGHT_PAD;
  const svgH = TOP + rows.length * ROW_H + BOTTOM;
  const plotBottom = svgH - BOTTOM + 6;

  const mapX = (t: number): number => LABEL_W + ((t - start) / span) * PLOT_W;

  const parts: string[] = [];
  parts.push(
    `<svg viewBox="0 0 ${svgW} ${svgH}" class="gantt" preserveAspectRatio="xMinYMin meet" role="img">`
  );
  parts.push(
    `<rect x="${LABEL_W}" y="${TOP - 6}" width="${PLOT_W}" height="${plotBottom - (TOP - 6)}" class="plot-bg"/>`
  );

  // Axis grid + tick labels (sample index).
  const tickCount = 8;
  for (let i = 0; i <= tickCount; i++) {
    const t = start + (span * i) / tickCount;
    const x = mapX(t);
    parts.push(`<line x1="${x}" y1="${TOP - 6}" x2="${x}" y2="${plotBottom}" class="grid"/>`);
    parts.push(
      `<text x="${x}" y="${TOP - 12}" class="tick" text-anchor="middle">${Math.round(t)}</text>`
    );
  }

  // Interval rows.
  rows.forEach((series, index) => {
    const y = TOP + index * ROW_H;
    const barY = y + (ROW_H - BAR_H) / 2;
    const label = escapeXml(series.source_name ?? series.series_id);
    const probText = series.probability !== null ? `p=${series.probability.toFixed(3)}` : "";
    const barClass = series.source_kind === "group" ? "bar-group" : "bar-rule";

    parts.push(
      `<rect x="0" y="${y}" width="${svgW}" height="${ROW_H}" class="${index % 2 ? "row-alt" : "row-base"}"/>`
    );
    parts.push(`<text x="14" y="${y + ROW_H / 2 + 4}" class="row-label">${label}</text>`);
    if (probText) {
      parts.push(
        `<text x="${LABEL_W - 12}" y="${y + ROW_H / 2 + 4}" class="row-prob" text-anchor="end">${probText}</text>`
      );
    }

    for (const [a, b] of series.intervals) {
      const x = mapX(a);
      const width = Math.max(2, mapX(b) - x);
      const title = `${label}  [${a}, ${b})${probText ? "  " + probText : ""}`;
      parts.push(
        `<rect x="${x}" y="${barY}" width="${width}" height="${BAR_H}" rx="4" class="${barClass}">` +
          `<title>${escapeXml(title)}</title></rect>`
      );
    }
  });

  // "Now" marker at the current sample index.
  if (snap.sample_index !== null) {
    const x = mapX(snap.sample_index);
    parts.push(`<line x1="${x}" y1="${TOP - 6}" x2="${x}" y2="${plotBottom}" class="now"/>`);
    parts.push(
      `<text x="${x}" y="${svgH - 8}" class="now-label" text-anchor="middle">now · ${snap.sample_index}</text>`
    );
  }

  parts.push("</svg>");
  return parts.join("");
}
