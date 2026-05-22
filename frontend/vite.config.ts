import { defineConfig } from "vite";
import { resolve } from "node:path";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";

const repoRoot = resolve(__dirname, "..");
const venvPython = resolve(repoRoot, ".venv", "bin", "python");
const queryPython = process.env.PROBLOG_QUERY_PYTHON
  ?? (existsSync(venvPython) ? venvPython : "python3");
const ecIntervalsJsonPath = resolve(repoRoot, "stl", "stl_viz", "ec_live_intervals.json");

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  if (chunks.length === 0) {
    return {};
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf-8"));
}

function writeJson(res: ServerResponse, statusCode: number, payload: unknown): void {
  res.statusCode = statusCode;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(payload));
}

async function runScallopQuery(query: string): Promise<unknown> {
  const child = spawn(
    queryPython,
    ["-m", "stl.scallop_query"],
    { cwd: repoRoot, env: process.env, stdio: ["pipe", "pipe", "pipe"] },
  );
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  const timeout = setTimeout(() => child.kill("SIGKILL"), 10_000);
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  child.stdin.end(JSON.stringify({ query, json_path: ecIntervalsJsonPath }));
  const exitCode = await new Promise<number | null>((res) => child.on("close", res));
  clearTimeout(timeout);
  const outputText = Buffer.concat(stdout).toString("utf-8").trim();
  let parsed: unknown;
  try {
    parsed = outputText ? JSON.parse(outputText) : {};
  } catch {
    parsed = { ok: false, error: outputText || "invalid response" };
  }
  if (exitCode !== 0) {
    const ep = parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
    return { ok: false, ...ep, stderr: Buffer.concat(stderr).toString("utf-8").trim() };
  }
  return parsed;
}

// The Python monitor writes stl_live.json, stl_live.svg, and ec_live_intervals.json
// into ../stl/stl_viz. Pointing publicDir there serves all three files live.
export default defineConfig({
  base: "./",
  publicDir: resolve(repoRoot, "stl", "stl_viz"),
  build: {
    outDir: resolve(__dirname, "dist"),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    open: true,
  },
  plugins: [
    {
      name: "scallop-query-api",
      configureServer(server) {
        server.middlewares.use("/api/scallop-query", async (req, res) => {
          if (req.method !== "POST") {
            writeJson(res, 405, { ok: false, error: "POST required" });
            return;
          }
          try {
            const body = await readJsonBody(req);
            const q = body && typeof body === "object" ? String((body as Record<string, unknown>).query ?? "") : "";
            writeJson(res, 200, await runScallopQuery(q));
          } catch (error) {
            writeJson(res, 500, { ok: false, error: error instanceof Error ? error.message : String(error) });
          }
        });
      },
    },
  ],
});
