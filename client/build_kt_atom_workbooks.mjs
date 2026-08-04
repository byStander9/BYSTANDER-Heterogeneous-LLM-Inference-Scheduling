import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  SpreadsheetFile,
  Workbook,
} from "file:///C:/Users/byStander/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";


const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const resultsRoot = path.join(repoRoot, "results", "kt_atom_20260803");
const outputDir = path.resolve(repoRoot, "..", "outputs",
  "019fc69e-513d-72c3-9a3b-abbbae050651");
const previewDir = path.join(outputDir, "previews");

const COLORS = {
  navy: "#16324F",
  blue: "#2563EB",
  teal: "#0F766E",
  green: "#15803D",
  amber: "#D97706",
  red: "#B91C1C",
  lightBlue: "#EAF2FF",
  lightTeal: "#E6F5F3",
  lightGray: "#F3F4F6",
  border: "#D1D5DB",
  white: "#FFFFFF",
  text: "#111827",
};


function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const source = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    if (quoted) {
      if (char === '"' && source[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const headers = rows.shift() ?? [];
  return rows.filter((values) => values.some((value) => value !== ""))
    .map((values) => Object.fromEntries(headers.map((header, index) =>
      [header, coerce(values[index] ?? "")])));
}


function coerce(value) {
  if (value === "") return null;
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) {
    return Number(value);
  }
  const cleaned = value.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g, "");
  const iso = cleaned.match(/^(\d{4}-\d{2}-\d{2})T(.+?)([+-]\d{2}:\d{2})$/);
  if (iso) return `${iso[1]} ${iso[2]} UTC${iso[3]}`;
  return cleaned.startsWith("=") ? `'${cleaned}` : cleaned;
}


async function readCsv(experiment, filename) {
  return parseCsv(await fs.readFile(path.join(resultsRoot, experiment, filename), "utf8"));
}


async function readSummary(experiment) {
  return JSON.parse(await fs.readFile(
    path.join(resultsRoot, experiment, "experiment_summary.json"), "utf8"));
}


function colName(index) {
  let value = index + 1;
  let output = "";
  while (value > 0) {
    value -= 1;
    output = String.fromCharCode(65 + (value % 26)) + output;
    value = Math.floor(value / 26);
  }
  return output;
}


function writeMatrix(sheet, startRow, startCol, matrix) {
  if (!matrix.length || !matrix[0].length) return;
  sheet.getRangeByIndexes(startRow, startCol, matrix.length, matrix[0].length)
    .values = matrix;
}


function styleTitle(sheet, range, title, subtitle) {
  sheet.getRange(range).merge();
  sheet.getRange(range).values = [[title]];
  sheet.getRange(range).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 18 },
    verticalAlignment: "center",
  };
  const firstCol = range.split(":")[0].replace(/\d/g, "");
  const lastCol = range.split(":")[1].replace(/\d/g, "");
  const subtitleRange = `${firstCol}2:${lastCol}2`;
  sheet.getRange(subtitleRange).merge();
  sheet.getRange(subtitleRange).values = [[subtitle]];
  sheet.getRange(subtitleRange).format = {
    fill: COLORS.lightBlue,
    font: { color: COLORS.navy, italic: true, size: 10 },
    wrapText: true,
  };
  sheet.getRange("1:1").format.rowHeightPx = 34;
  sheet.getRange("2:2").format.rowHeightPx = 34;
  sheet.showGridLines = false;
}


function styleHeader(sheet, row, columnCount, fill = COLORS.teal) {
  const range = sheet.getRangeByIndexes(row - 1, 0, 1, columnCount);
  range.format = {
    fill,
    font: { bold: true, color: COLORS.white, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
    horizontalAlignment: "center",
    borders: { preset: "all", style: "thin", color: COLORS.border },
  };
  range.format.rowHeightPx = 42;
}


function styleDataSheet(sheet, rowCount, colCount, widths = {}) {
  styleHeader(sheet, 1, colCount);
  sheet.freezePanes.freezeRows(1);
  sheet.getRangeByIndexes(1, 0, Math.max(rowCount - 1, 1), colCount).format = {
    font: { size: 9, color: COLORS.text },
    borders: { preset: "inside", style: "thin", color: "#E5E7EB" },
    verticalAlignment: "top",
  };
  for (let index = 0; index < colCount; index += 1) {
    const width = widths[index] ?? 95;
    sheet.getRangeByIndexes(0, index, Math.max(rowCount, 2), 1)
      .format.columnWidthPx = width;
  }
  sheet.showGridLines = false;
}


function addTable(sheet, name, rowCount, colCount) {
  const last = colName(colCount - 1);
  const table = sheet.tables.add(`A1:${last}${rowCount}`, true, name);
  table.showFilterButton = true;
  table.showBandedRows = true;
  return table;
}


function setNumberFormats(sheet, formats, rowCount) {
  for (const [columnIndex, format] of Object.entries(formats)) {
    sheet.getRangeByIndexes(1, Number(columnIndex), Math.max(rowCount - 1, 1), 1)
      .setNumberFormat(format);
  }
}


async function addPreview(workbook, sheetName, range, prefix) {
  const blob = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const safe = sheetName.replace(/[^a-zA-Z0-9]+/g, "_").toLowerCase();
  await fs.writeFile(path.join(previewDir, `${prefix}_${safe}.png`),
    new Uint8Array(await blob.arrayBuffer()));
}


function addLineChart(sheet, title, categories, seriesSpecs, position) {
  const chart = sheet.charts.add("line", {
    chartType: "line",
    title,
    hasLegend: true,
  });
  for (const spec of seriesSpecs) {
    const series = chart.series.add(spec.name);
    series.categoryFormula = categories;
    series.formula = spec.formula;
    series.fill = spec.color;
  }
  chart.title = title;
  chart.titleTextStyle.fontSize = 12;
  chart.hasLegend = true;
  chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
  chart.yAxis = { numberFormatCode: seriesSpecs[0].numberFormat ?? "0.0" };
  chart.setPosition(position[0], position[1]);
  return chart;
}


async function collectExperiments(names) {
  const output = [];
  for (const name of names) {
    output.push({
      name,
      summary: await readSummary(name),
      requests: await readCsv(name, "client_requests.csv"),
      timeseries: await readCsv(name, "vllm_timeseries.csv"),
      internal: await readCsv(name, "vllm_histogram_summary.csv"),
    });
  }
  return output;
}


function requestMatrix(experiments, training = false) {
  const headers = training ? [
    "experiment_id", "sample_id", "request_id", "dataset_index", "target_qps",
    "endpoint", "prompt_first150", "prompt_last150", "prompt_chars",
    "prompt_tokens_tokenize_api", "prompt_tokens_usage", "vllm_running",
    "vllm_waiting", "inflight_count", "inflight_p99_tokens",
    "inflight_p90_tokens", "inflight_p75_tokens", "inflight_p50_tokens",
    "inflight_p25_tokens", "inflight_prompt_tokens_json", "kv_cache_usage_perc",
    "metrics_age_ms", "client_ttft_ms", "client_e2e_ms", "completion_tokens",
    "success", "quality_eligible", "slm_input_text", "submitted_at",
    "metrics_observed_at", "arrival_lag_ms", "state_source", "error",
  ] : [
    "experiment_id", "request_id", "dataset_index", "target_qps", "endpoint_index",
    "endpoint", "scheduled_elapsed_s", "actual_submit_elapsed_s", "arrival_lag_ms",
    "submitted_at", "completed_at", "success", "http_status", "client_ttft_ms",
    "client_e2e_ms", "prompt_tokens", "completion_tokens", "prompt_chars",
    "vllm_running", "vllm_waiting", "inflight_count", "inflight_p99_tokens",
    "kv_cache_usage_perc", "metrics_age_ms", "prompt_first150", "prompt_last150",
    "error",
  ];
  const rows = [];
  const ranges = new Map();
  for (const experiment of experiments) {
    const start = rows.length + 2;
    for (const request of experiment.requests) {
      if (training) {
        rows.push([
          experiment.name, `${experiment.name}:${request.request_id}`,
          request.request_id, request.dataset_index, request.target_qps, request.endpoint,
          request.prompt_first150, request.prompt_last150, request.prompt_chars,
          request.prompt_tokens_tokenize_api, request.prompt_tokens_usage,
          request.vllm_running, request.vllm_waiting, request.inflight_count,
          request.inflight_p99_tokens, request.inflight_p90_tokens,
          request.inflight_p75_tokens, request.inflight_p50_tokens,
          request.inflight_p25_tokens, request.inflight_prompt_tokens_json,
          request.kv_cache_usage_perc, request.metrics_age_ms,
          request.client_ttft_ms, request.client_e2e_ms, request.completion_tokens,
          request.success, null, null, request.submitted_at,
          request.metrics_observed_at, request.arrival_lag_ms, request.state_source,
          request.error,
        ]);
      } else {
        rows.push([
          experiment.name, request.request_id, request.dataset_index, request.target_qps,
          request.endpoint_index, request.endpoint, request.scheduled_elapsed_s,
          request.actual_submit_elapsed_s, request.arrival_lag_ms, request.submitted_at,
          request.completed_at, request.success, request.http_status,
          request.client_ttft_ms, request.client_e2e_ms,
          request.prompt_tokens_tokenize_api, request.completion_tokens,
          request.prompt_chars, request.vllm_running, request.vllm_waiting,
          request.inflight_count, request.inflight_p99_tokens,
          request.kv_cache_usage_perc, request.metrics_age_ms,
          request.prompt_first150, request.prompt_last150, request.error,
        ]);
      }
    }
    ranges.set(experiment.name, { start, end: rows.length + 1 });
  }
  return { headers, rows, ranges };
}


function timeseriesMatrix(experiments) {
  const headers = [
    "experiment_id", "target_qps", "timestamp", "elapsed_s", "endpoint_index",
    "endpoint", "num_running", "num_waiting", "kv_cache_usage_perc",
    "num_preemptions_total", "prompt_tokens_total", "generation_tokens_total",
    "process_start_time_seconds", "scrape_error",
  ];
  const rows = [];
  const ranges = new Map();
  for (const experiment of experiments) {
    const start = rows.length + 2;
    for (const point of experiment.timeseries) {
      rows.push([
        experiment.name, experiment.summary.target_qps, point.timestamp, point.elapsed_s,
        point.endpoint_index, point.endpoint, point["vllm:num_requests_running"],
        point["vllm:num_requests_waiting"], point["vllm:kv_cache_usage_perc"],
        point["vllm:num_preemptions_total"], point["vllm:prompt_tokens_total"],
        point["vllm:generation_tokens_total"], point.process_start_time_seconds,
        point.scrape_error,
      ]);
    }
    ranges.set(experiment.name, { start, end: rows.length + 1 });
  }
  return { headers, rows, ranges };
}


function internalMatrix(experiments) {
  const headers = [
    "experiment_id", "target_qps", "scope", "endpoint", "metric", "count_delta",
    "sum_delta_s", "average_s", "p99_histogram_estimate_s", "p99_bucket_lower_s",
    "p99_bucket_upper_s", "process_restarted",
  ];
  const rows = [];
  const rowMap = new Map();
  for (const experiment of experiments) {
    for (const metric of experiment.internal) {
      rows.push([
        experiment.name, experiment.summary.target_qps, metric.scope, metric.endpoint,
        metric.metric, metric.count_delta, metric.sum_delta_s, metric.average_s,
        metric.p99_histogram_estimate_s, metric.p99_bucket_lower_s,
        metric.p99_bucket_upper_s, metric.process_restarted,
      ]);
      if (metric.scope === "combined") {
        rowMap.set(`${experiment.name}|${metric.metric}`, rows.length + 1);
      }
    }
  }
  return { headers, rows, rowMap };
}


function writeRawSheet(workbook, name, matrix, tableName, widths, numberFormats) {
  const sheet = workbook.worksheets.add(name);
  writeMatrix(sheet, 0, 0, [matrix.headers, ...matrix.rows]);
  const rowCount = matrix.rows.length + 1;
  styleDataSheet(sheet, rowCount, matrix.headers.length, widths);
  setNumberFormats(sheet, numberFormats, rowCount);
  addTable(sheet, tableName, rowCount, matrix.headers.length);
  return sheet;
}


async function buildRoutingWorkbook(experiments, pilots) {
  const workbook = Workbook.create();
  const summarySheet = workbook.worksheets.add("RR Summary");
  const requestData = requestMatrix(experiments);
  const timeseriesData = timeseriesMatrix(experiments);
  const internalData = internalMatrix(experiments);

  const requestsSheet = writeRawSheet(workbook, "RR Requests", requestData,
    "RRRequestsTable", {
      0: 150, 1: 70, 2: 85, 3: 70, 4: 70, 5: 300, 6: 105, 7: 105,
      8: 95, 9: 250, 10: 250, 11: 65, 12: 70, 13: 105, 14: 105,
      15: 95, 16: 90, 17: 90, 18: 85, 19: 85, 20: 90, 21: 105,
      22: 95, 23: 100, 24: 230, 25: 230, 26: 220,
    }, { 3: "0.00", 8: "0.00", 13: "0.00", 14: "0.00", 22: "0.0000", 23: "0.00" });
  requestsSheet.getRange(`Y2:Z${requestData.rows.length + 1}`).format.wrapText = true;

  writeRawSheet(workbook, "vLLM Internal", internalData, "RRInternalTable", {
    0: 150, 1: 75, 2: 90, 3: 300, 4: 250, 5: 90, 6: 105, 7: 95,
    8: 125, 9: 110, 10: 110, 11: 95,
  }, { 1: "0.00", 5: "0", 6: "0.000", 7: "0.000", 8: "0.000", 9: "0.000" });
  writeRawSheet(workbook, "vLLM Timeseries", timeseriesData, "RRTimeseriesTable", {
    0: 150, 1: 75, 2: 250, 3: 90, 4: 75, 5: 300, 6: 90, 7: 90,
    8: 100, 9: 105, 10: 120, 11: 130, 12: 130, 13: 210,
  }, { 1: "0.00", 3: "0.000", 8: "0.0000" });

  const pilotRequestData = requestMatrix(pilots);
  const pilotInternalData = internalMatrix(pilots);
  writeRawSheet(workbook, "Pilot Requests", pilotRequestData, "PilotRequestsTable", {
    0: 140, 1: 70, 2: 85, 3: 70, 4: 70, 5: 300, 6: 105, 7: 105,
    8: 95, 9: 250, 10: 250, 11: 65, 12: 70, 13: 105, 14: 105,
    15: 95, 16: 90, 17: 90, 18: 85, 19: 85, 20: 90, 21: 105,
    22: 95, 23: 100, 24: 230, 25: 230, 26: 220,
  }, { 3: "0.00", 13: "0.00", 14: "0.00" });
  writeRawSheet(workbook, "Pilot Internal", pilotInternalData, "PilotInternalTable", {
    0: 140, 1: 75, 2: 90, 3: 300, 4: 250, 5: 90, 6: 105, 7: 95,
    8: 125, 9: 110, 10: 110, 11: 95,
  }, { 1: "0.00", 5: "0", 6: "0.000", 7: "0.000", 8: "0.000" });

  styleTitle(summarySheet, "A1:U1", "KT Cloud ATOM+ Round-Robin Benchmark",
    "Qwen3-4B · two independent KT Model Serving endpoints · ShareGPT · Poisson arrivals · max_tokens=64");
  const headers = [
    "Target QPS", "Requests", "Success", "Client TTFT avg (s)", "Client TTFT p99 (s)",
    "Client E2E avg (s)", "Client E2E p99 (s)", "vLLM TTFT avg (s)",
    "vLLM TTFT p99 approx (s)", "vLLM E2E avg (s)", "vLLM E2E p99 approx (s)",
    "Queue avg (s)", "Queue p99 approx (s)", "Max waiting / replica",
    "Waiting-positive samples", "Max running / replica", "Max KV-cache", "Preemptions",
    "Arrival lag avg (ms)", "Internal E2E count", "Quality",
  ];
  writeMatrix(summarySheet, 4, 0, [headers]);
  styleHeader(summarySheet, 5, headers.length, COLORS.blue);
  const sorted = [...experiments].sort((a, b) => a.summary.target_qps - b.summary.target_qps);
  writeMatrix(summarySheet, 5, 0, sorted.map((experiment) => [experiment.summary.target_qps]));
  for (let index = 0; index < sorted.length; index += 1) {
    const row = index + 6;
    const experiment = sorted[index];
    const rr = requestData.ranges.get(experiment.name);
    const ts = timeseriesData.ranges.get(experiment.name);
    const ttftRow = internalData.rowMap.get(`${experiment.name}|vllm:time_to_first_token_seconds`);
    const e2eRow = internalData.rowMap.get(`${experiment.name}|vllm:e2e_request_latency_seconds`);
    const queueRow = internalData.rowMap.get(`${experiment.name}|vllm:request_queue_time_seconds`);
    const formulas = [
      `=ROWS('RR Requests'!$A$${rr.start}:$A$${rr.end})`,
      `=SUM('RR Requests'!$L$${rr.start}:$L$${rr.end})`,
      `=AVERAGE('RR Requests'!$N$${rr.start}:$N$${rr.end})/1000`,
      `=PERCENTILE.INC('RR Requests'!$N$${rr.start}:$N$${rr.end},0.99)/1000`,
      `=AVERAGE('RR Requests'!$O$${rr.start}:$O$${rr.end})/1000`,
      `=PERCENTILE.INC('RR Requests'!$O$${rr.start}:$O$${rr.end},0.99)/1000`,
      `='vLLM Internal'!$H$${ttftRow}`,
      `='vLLM Internal'!$I$${ttftRow}`,
      `='vLLM Internal'!$H$${e2eRow}`,
      `='vLLM Internal'!$I$${e2eRow}`,
      `='vLLM Internal'!$H$${queueRow}`,
      `='vLLM Internal'!$I$${queueRow}`,
      `=MAX('vLLM Timeseries'!$H$${ts.start}:$H$${ts.end})`,
      `=COUNTIF('vLLM Timeseries'!$H$${ts.start}:$H$${ts.end},">0")/ROWS('vLLM Timeseries'!$H$${ts.start}:$H$${ts.end})`,
      `=MAX('vLLM Timeseries'!$G$${ts.start}:$G$${ts.end})`,
      `=MAX('vLLM Timeseries'!$I$${ts.start}:$I$${ts.end})`,
      `=MAX('vLLM Timeseries'!$J$${ts.start}:$J$${ts.end})-MIN('vLLM Timeseries'!$J$${ts.start}:$J$${ts.end})`,
      `=AVERAGE('RR Requests'!$I$${rr.start}:$I$${rr.end})`,
      `='vLLM Internal'!$F$${e2eRow}`,
      `=IF(AND(B${row}=C${row},B${row}=T${row},R${row}=0),"PASS","CHECK")`,
    ];
    summarySheet.getRangeByIndexes(row - 1, 1, 1, formulas.length).formulas = [formulas];
  }
  summarySheet.getRange("A6:U9").format = {
    borders: { preset: "all", style: "thin", color: COLORS.border },
    verticalAlignment: "center",
  };
  summarySheet.getRange("D6:M9").setNumberFormat("0.000");
  summarySheet.getRange("O6:O9").setNumberFormat("0.0%");
  summarySheet.getRange("Q6:Q9").setNumberFormat("0.0%");
  summarySheet.getRange("S6:S9").setNumberFormat("0.00");
  summarySheet.getRange("U6:U9").conditionalFormats.add("containsText", {
    text: "PASS", format: { fill: "#DCFCE7", font: { color: COLORS.green, bold: true } },
  });
  for (let column = 0; column < headers.length; column += 1) {
    summarySheet.getRangeByIndexes(4, column, 5, 1).format.columnWidthPx =
      column === 20 ? 85 : column === 14 ? 120 : 105;
  }
  summarySheet.freezePanes.freezeRows(5);
  addLineChart(summarySheet, "Client TTFT vs offered QPS (seconds)",
    "'RR Summary'!$A$6:$A$9", [
      { name: "Average", formula: "'RR Summary'!$D$6:$D$9", color: COLORS.blue, numberFormat: "0.0" },
      { name: "P99", formula: "'RR Summary'!$E$6:$E$9", color: COLORS.red, numberFormat: "0.0" },
    ], ["A12", "J28"]);
  addLineChart(summarySheet, "Client E2E vs offered QPS (seconds)",
    "'RR Summary'!$A$6:$A$9", [
      { name: "Average", formula: "'RR Summary'!$F$6:$F$9", color: COLORS.teal, numberFormat: "0.0" },
      { name: "P99", formula: "'RR Summary'!$G$6:$G$9", color: COLORS.amber, numberFormat: "0.0" },
    ], ["K12", "U28"]);
  addLineChart(summarySheet, "Client vs vLLM E2E p99 (seconds)",
    "'RR Summary'!$A$6:$A$9", [
      { name: "Client raw p99", formula: "'RR Summary'!$G$6:$G$9", color: COLORS.blue, numberFormat: "0.0" },
      { name: "vLLM histogram approx", formula: "'RR Summary'!$K$6:$K$9", color: COLORS.red, numberFormat: "0.0" },
    ], ["A30", "J46"]);
  addLineChart(summarySheet, "Max waiting requests per replica", "'RR Summary'!$A$6:$A$9", [
    { name: "Max waiting", formula: "'RR Summary'!$N$6:$N$9", color: COLORS.red, numberFormat: "0" },
  ], ["K30", "U46"]);

  const catalog = workbook.worksheets.add("Experiment Catalog");
  const catalogHeaders = [
    "experiment_id", "role", "included", "mode", "target_qps", "requests", "successful",
    "start_index", "seed", "max_tokens", "metrics_interval_s", "result_directory",
  ];
  const allRuns = [...experiments.map((item) => [item, "main", true]),
    ...pilots.map((item) => [item, "pilot", true])];
  const catalogRows = allRuns.map(([item, role, included]) => [
    item.name, role, included, item.summary.mode, item.summary.target_qps,
    item.summary.requests, item.summary.successful, item.summary.start_index,
    item.summary.seed, item.summary.max_tokens, item.summary.metrics_interval_s,
    path.join(resultsRoot, item.name),
  ]);
  writeMatrix(catalog, 0, 0, [catalogHeaders, ...catalogRows]);
  styleDataSheet(catalog, catalogRows.length + 1, catalogHeaders.length, {
    0: 180, 1: 80, 2: 75, 3: 75, 4: 80, 5: 80, 6: 80, 7: 85,
    8: 90, 9: 90, 10: 110, 11: 500,
  });
  addTable(catalog, "RRCatalogTable", catalogRows.length + 1, catalogHeaders.length);

  const readme = workbook.worksheets.add("README");
  styleTitle(readme, "A1:H1", "Routing Experiment Workbook", "How to interpret client and vLLM measurements");
  const notes = [
    ["Item", "Definition / decision"],
    ["Endpoints", "svc-1 and svc-2 are independent KT Model Serving endpoints, each backed by one ATOM+ session."],
    ["Routing", "Deterministic round robin: even request IDs → svc-1, odd request IDs → svc-2."],
    ["Arrival process", "Poisson arrivals with a fixed seed per run. Main runs contain 600 requests per QPS."],
    ["Client TTFT", "Wall-clock time from HTTP submission to the first non-empty content or reasoning_content SSE fragment."],
    ["Client E2E", "Wall-clock time from HTTP submission through stream completion; includes KT proxy/network/client overhead."],
    ["vLLM average", "Exact experiment-window counter delta: histogram _sum delta / _count delta."],
    ["vLLM p99", "Approximation from cumulative histogram bucket deltas, interpolated within the enclosing bucket."],
    ["Important", "Client p99 and vLLM histogram p99 are independent measurements. Do not add or average them."],
    ["Combined p99", "Buckets from both replicas are summed before quantile estimation; endpoint p99 values are not averaged."],
    ["Quality", "PASS requires client success count = request count = internal E2E count and zero preemption delta."],
    ["Observed knee", "QPS 1.25 is moderate; QPS 1.5 begins sustained queueing; QPS 2 is overload."],
    ["Model settings", "Qwen3-4B, max_tokens=64, temperature=0, max_turns=3, vLLM 0.13.0."],
    ["Source paper", "C:\\Users\\byStander\\Documents\\dnc\\byStander\\paper\\BYSTANDER.pdf"],
  ];
  writeMatrix(readme, 3, 0, notes);
  styleHeader(readme, 4, 2, COLORS.blue);
  readme.getRange(`A5:B${notes.length + 3}`).format = {
    wrapText: true, verticalAlignment: "top",
    borders: { preset: "all", style: "thin", color: COLORS.border },
  };
  readme.getRange("A1:A20").format.columnWidthPx = 155;
  readme.getRange("B1:B20").format.columnWidthPx = 760;
  readme.freezePanes.freezeRows(4);

  const inspect = await workbook.inspect({
    kind: "table", range: "RR Summary!A1:U10", include: "values,formulas",
    tableMaxRows: 12, tableMaxCols: 21, maxChars: 12000,
  });
  console.log(inspect.ndjson);
  const errors = await workbook.inspect({
    kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    range: "RR Summary!A1:U10",
    options: { useRegex: true, maxResults: 300 }, summary: "routing formula error scan",
  });
  console.log(errors.ndjson);

  const previews = [
    ["RR Summary", "A1:U46"], ["RR Requests", "A1:AA24"],
    ["vLLM Internal", "A1:L24"], ["vLLM Timeseries", "A1:N24"],
    ["Pilot Requests", "A1:AA24"], ["Pilot Internal", "A1:L24"],
    ["Experiment Catalog", "A1:L12"], ["README", "A1:H18"],
  ];
  for (const [sheetName, range] of previews) {
    await addPreview(workbook, sheetName, range, "rr");
  }
  const output = await SpreadsheetFile.exportXlsx(workbook);
  const target = path.join(outputDir, "kt_atom_rr_routing_results.xlsx");
  await output.save(target);
  return target;
}


async function buildSlmWorkbook(experiments, excluded) {
  const workbook = Workbook.create();
  const summarySheet = workbook.worksheets.add("Load Summary");
  const trainingData = requestMatrix(experiments, true);
  const timeseriesData = timeseriesMatrix(experiments);
  const internalData = internalMatrix(experiments);

  const training = writeRawSheet(workbook, "Training Dataset", trainingData,
    "TrainingDatasetTable", {
      0: 180, 1: 210, 2: 70, 3: 85, 4: 75, 5: 300, 6: 260, 7: 260,
      8: 90, 9: 105, 10: 100, 11: 90, 12: 90, 13: 90, 14: 105,
      15: 105, 16: 105, 17: 105, 18: 105, 19: 300, 20: 100, 21: 100,
      22: 105, 23: 105, 24: 95, 25: 65, 26: 100, 27: 620, 28: 250,
      29: 250, 30: 95, 31: 260, 32: 240,
    }, { 4: "0.00", 20: "0.0000", 21: "0.00", 22: "0.00", 23: "0.00", 30: "0.00" });
  const lastTrainingRow = trainingData.rows.length + 1;
  training.getRange("AA2").formulas = [[
    "=IF(AND(Z2=1,J2=K2,L2>=0,M2>=0,V2<=300),1,0)",
  ]];
  training.getRange(`AA2:AA${lastTrainingRow}`).fillDown();
  training.getRange("AB2").formulas = [[
    "=\"ATOM+_POOL status: running \"&TEXT(L2,\"0\")&\", waiting \"&TEXT(M2,\"0\")&\", tokens \"&TEXT(N2,\"0\")&\" [p99:\"&TEXT(O2,\"0\")&\", p90:\"&TEXT(P2,\"0\")&\", p75:\"&TEXT(Q2,\"0\")&\", p50:\"&TEXT(R2,\"0\")&\", p25:\"&TEXT(S2,\"0\")&\"]. Request: first150='\"&G2&\"', last150='\"&H2&\"', total_chars=\"&TEXT(I2,\"0\")&\".\"",
  ]];
  training.getRange(`AB2:AB${lastTrainingRow}`).fillDown();
  training.getRange(`G2:H${lastTrainingRow}`).format.wrapText = true;
  training.getRange(`AB2:AB${lastTrainingRow}`).format.wrapText = true;
  training.getRange(`AA2:AA${lastTrainingRow}`).conditionalFormats.add("cellIs", {
    operator: "equal", formula: 1,
    format: { fill: "#DCFCE7", font: { color: COLORS.green, bold: true } },
  });

  writeRawSheet(workbook, "vLLM Internal", internalData, "SLMInternalTable", {
    0: 180, 1: 75, 2: 90, 3: 300, 4: 250, 5: 90, 6: 105, 7: 95,
    8: 125, 9: 110, 10: 110, 11: 95,
  }, { 1: "0.00", 5: "0", 6: "0.000", 7: "0.000", 8: "0.000" });
  writeRawSheet(workbook, "vLLM Timeseries", timeseriesData, "SLMTimeseriesTable", {
    0: 180, 1: 75, 2: 250, 3: 90, 4: 75, 5: 300, 6: 90, 7: 90,
    8: 100, 9: 105, 10: 120, 11: 130, 12: 130, 13: 210,
  }, { 1: "0.00", 3: "0.000", 8: "0.0000" });

  const excludedData = requestMatrix(excluded, true);
  const excludedSheet = writeRawSheet(workbook, "Excluded Run", excludedData,
    "ExcludedRunTable", {
      0: 250, 1: 210, 2: 70, 3: 85, 4: 75, 5: 300, 6: 240, 7: 240,
      8: 90, 9: 105, 10: 100, 11: 90, 12: 90, 13: 90, 14: 105,
      15: 105, 16: 105, 17: 105, 18: 105, 19: 300, 20: 100, 21: 100,
      22: 105, 23: 105, 24: 95, 25: 65, 26: 100, 27: 300, 28: 250,
      29: 250, 30: 95, 31: 260, 32: 300,
    }, { 4: "0.00", 20: "0.0000", 21: "0.00", 22: "0.00", 23: "0.00" });
  excludedSheet.getRange("A1:AG1").format.fill = COLORS.red;

  styleTitle(summarySheet, "A1:R1", "BYSTANDER SLM Training Dataset — ATOM+",
    "ShareGPT · svc-1 / one ATOM+ · paper Eq. 4 state vector · 1,200 successful / 1,194 strict-quality eligible");
  const headers = [
    "Target QPS", "Samples", "Eligible", "Client TTFT avg (s)", "Client TTFT p99 (s)",
    "Client E2E avg (s)", "Client E2E p99 (s)", "vLLM E2E avg (s)",
    "vLLM E2E p99 approx (s)", "Avg running", "Max running", "Avg waiting",
    "Max waiting", "Waiting-positive samples", "Avg inflight", "Max inflight",
    "Avg metrics age (ms)", "Quality",
  ];
  writeMatrix(summarySheet, 4, 0, [headers]);
  styleHeader(summarySheet, 5, headers.length, COLORS.blue);
  const sorted = [...experiments].sort((a, b) => a.summary.target_qps - b.summary.target_qps);
  writeMatrix(summarySheet, 5, 0, sorted.map((experiment) => [experiment.summary.target_qps]));
  for (let index = 0; index < sorted.length; index += 1) {
    const row = index + 6;
    const experiment = sorted[index];
    const rr = trainingData.ranges.get(experiment.name);
    const e2eRow = internalData.rowMap.get(`${experiment.name}|vllm:e2e_request_latency_seconds`);
    const formulas = [
      `=ROWS('Training Dataset'!$A$${rr.start}:$A$${rr.end})`,
      `=SUM('Training Dataset'!$AA$${rr.start}:$AA$${rr.end})`,
      `=AVERAGE('Training Dataset'!$W$${rr.start}:$W$${rr.end})/1000`,
      `=PERCENTILE.INC('Training Dataset'!$W$${rr.start}:$W$${rr.end},0.99)/1000`,
      `=AVERAGE('Training Dataset'!$X$${rr.start}:$X$${rr.end})/1000`,
      `=PERCENTILE.INC('Training Dataset'!$X$${rr.start}:$X$${rr.end},0.99)/1000`,
      `='vLLM Internal'!$H$${e2eRow}`,
      `='vLLM Internal'!$I$${e2eRow}`,
      `=AVERAGE('Training Dataset'!$L$${rr.start}:$L$${rr.end})`,
      `=MAX('Training Dataset'!$L$${rr.start}:$L$${rr.end})`,
      `=AVERAGE('Training Dataset'!$M$${rr.start}:$M$${rr.end})`,
      `=MAX('Training Dataset'!$M$${rr.start}:$M$${rr.end})`,
      `=COUNTIF('Training Dataset'!$M$${rr.start}:$M$${rr.end},">0")/ROWS('Training Dataset'!$M$${rr.start}:$M$${rr.end})`,
      `=AVERAGE('Training Dataset'!$N$${rr.start}:$N$${rr.end})`,
      `=MAX('Training Dataset'!$N$${rr.start}:$N$${rr.end})`,
      `=AVERAGE('Training Dataset'!$V$${rr.start}:$V$${rr.end})`,
      `=IF(B${row}=C${row},"PASS","CHECK")`,
    ];
    summarySheet.getRangeByIndexes(row - 1, 1, 1, formulas.length).formulas = [formulas];
  }
  summarySheet.getRange("A6:R9").format = {
    borders: { preset: "all", style: "thin", color: COLORS.border },
    verticalAlignment: "center",
  };
  summarySheet.getRange("D6:I9").setNumberFormat("0.000");
  summarySheet.getRange("J6:P9").setNumberFormat("0.00");
  summarySheet.getRange("N6:N9").setNumberFormat("0.0%");
  summarySheet.getRange("Q6:Q9").setNumberFormat("0.00");
  summarySheet.getRange("R6:R9").conditionalFormats.add("containsText", {
    text: "PASS", format: { fill: "#DCFCE7", font: { color: COLORS.green, bold: true } },
  });
  for (let column = 0; column < headers.length; column += 1) {
    summarySheet.getRangeByIndexes(4, column, 5, 1).format.columnWidthPx =
      column === 13 ? 125 : 105;
  }
  summarySheet.freezePanes.freezeRows(5);
  addLineChart(summarySheet, "Training labels: client E2E by offered QPS (seconds)",
    "'Load Summary'!$A$6:$A$9", [
      { name: "Average", formula: "'Load Summary'!$F$6:$F$9", color: COLORS.teal, numberFormat: "0.0" },
      { name: "P99", formula: "'Load Summary'!$G$6:$G$9", color: COLORS.red, numberFormat: "0.0" },
    ], ["A12", "I28"]);
  addLineChart(summarySheet, "Request-time waiting state by offered QPS", "'Load Summary'!$A$6:$A$9", [
    { name: "Average waiting", formula: "'Load Summary'!$L$6:$L$9", color: COLORS.blue, numberFormat: "0.0" },
    { name: "Max waiting", formula: "'Load Summary'!$M$6:$M$9", color: COLORS.red, numberFormat: "0.0" },
  ], ["J12", "R28"]);
  addLineChart(summarySheet, "Client vs vLLM E2E p99 (seconds)", "'Load Summary'!$A$6:$A$9", [
    { name: "Client raw p99", formula: "'Load Summary'!$G$6:$G$9", color: COLORS.blue, numberFormat: "0.0" },
    { name: "vLLM histogram approx", formula: "'Load Summary'!$I$6:$I$9", color: COLORS.amber, numberFormat: "0.0" },
  ], ["A30", "I46"]);

  const readme = workbook.worksheets.add("README");
  styleTitle(readme, "A1:H1", "SLM Dataset Guide", "Feature provenance, label definition, and training cautions");
  const notes = [
    ["Item", "Definition / decision"],
    ["Successful samples", "1,200 rows: QPS 0.3, 0.5, 0.7, 0.9 × 300 requests, all served by svc-1 / one ATOM+."],
    ["Strict-quality rows", "1,194 rows meet quality_eligible=1. Six successful QPS 0.5 rows have metrics_age_ms > 300 and remain available with quality_eligible=0."],
    ["Paper feature vector", "running, waiting, inflight count, and inflight prompt-token p99/p90/p75/p50/p25."],
    ["Request feature", "First 150 and last 150 characters of the final user prompt plus total character count."],
    ["Token length source", "Qwen3-4B /tokenize API using the actual chat messages and chat template before load starts."],
    ["Inflight token source", "Client registry of this experiment's submitted-but-not-completed requests. Stock vLLM 0.13 does not expose per-request inflight token lengths."],
    ["Running/waiting source", "Cached vLLM Prometheus gauges at request submission. metrics_age_ms records snapshot staleness."],
    ["Label", "client_e2e_ms: HTTP submission through stream completion, including KT proxy/network/client overhead."],
    ["quality_eligible", "success=1, tokenize count equals usage count, gauges present, and metrics_age_ms ≤ 300."],
    ["slm_input_text", "Formula-generated natural-language input matching BYSTANDER's pool-status + request format."],
    ["Excluded run", "The first QPS 0.7 run is preserved separately because one 98,388-token prompt exceeded max_model_len=40,960. It is not part of Training Dataset."],
    ["Scale caution", "The paper used approximately 90K samples. This 1,200-row workbook validates the real ATOM+ data pipeline and supports initial training, not a final production predictor."],
    ["Network caution", "Labels are specific to the KT Model Serving path. Do not mix with direct-IP/CSP labels without a deployment-path feature or separate model."],
    ["Source paper", "C:\\Users\\byStander\\Documents\\dnc\\byStander\\paper\\BYSTANDER.pdf"],
  ];
  writeMatrix(readme, 3, 0, notes);
  styleHeader(readme, 4, 2, COLORS.blue);
  readme.getRange(`A5:B${notes.length + 3}`).format = {
    wrapText: true, verticalAlignment: "top",
    borders: { preset: "all", style: "thin", color: COLORS.border },
  };
  readme.getRange("A1:A20").format.columnWidthPx = 165;
  readme.getRange("B1:B20").format.columnWidthPx = 760;
  readme.freezePanes.freezeRows(4);

  const catalog = workbook.worksheets.add("Experiment Catalog");
  const catalogHeaders = [
    "experiment_id", "included", "reason", "mode", "target_qps", "requests",
    "successful", "oversized_prompts_skipped", "start_index", "seed", "result_directory",
  ];
  const validRows = experiments.map((item) => [
    item.name, true, "valid training run", item.summary.mode, item.summary.target_qps,
    item.summary.requests, item.summary.successful, item.summary.oversized_prompts_skipped ?? 0,
    item.summary.start_index, item.summary.seed, path.join(resultsRoot, item.name),
  ]);
  const excludedRows = excluded.map((item) => [
    item.name, false, "excluded: one 98,388-token prompt caused HTTP 400 before length filtering",
    item.summary.mode, item.summary.target_qps, item.summary.requests,
    item.summary.successful, item.summary.oversized_prompts_skipped ?? 0,
    item.summary.start_index, item.summary.seed, path.join(resultsRoot, item.name),
  ]);
  writeMatrix(catalog, 0, 0, [catalogHeaders, ...validRows, ...excludedRows]);
  styleDataSheet(catalog, validRows.length + excludedRows.length + 1,
    catalogHeaders.length, { 0: 300, 1: 80, 2: 480, 3: 75, 4: 80, 5: 80,
      6: 80, 7: 110, 8: 85, 9: 90, 10: 500 });
  addTable(catalog, "SLMCatalogTable", validRows.length + excludedRows.length + 1,
    catalogHeaders.length);

  const inspect = await workbook.inspect({
    kind: "table", range: "Load Summary!A1:R10", include: "values,formulas",
    tableMaxRows: 12, tableMaxCols: 18, maxChars: 12000,
  });
  console.log(inspect.ndjson);
  const formulaInspect = await workbook.inspect({
    kind: "table", range: "Training Dataset!A1:AG6", include: "values,formulas",
    tableMaxRows: 6, tableMaxCols: 33, maxChars: 15000,
  });
  console.log(formulaInspect.ndjson);
  const errors = await workbook.inspect({
    kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    range: "Load Summary!A1:R10",
    options: { useRegex: true, maxResults: 300 }, summary: "SLM formula error scan",
  });
  console.log(errors.ndjson);

  const previews = [
    ["Load Summary", "A1:R46"], ["Training Dataset", "A1:AG24"],
    ["vLLM Internal", "A1:L24"], ["vLLM Timeseries", "A1:N24"],
    ["Excluded Run", "A1:AG24"], ["Experiment Catalog", "A1:K10"],
    ["README", "A1:H18"],
  ];
  for (const [sheetName, range] of previews) {
    await addPreview(workbook, sheetName, range, "slm");
  }
  const output = await SpreadsheetFile.exportXlsx(workbook);
  const target = path.join(outputDir, "kt_atom_slm_training_dataset.xlsx");
  await output.save(target);
  return target;
}


await fs.mkdir(previewDir, { recursive: true });
const rrNames = ["rr_qps1_n600", "rr_qps1_25_n600", "rr_qps1_5_n600", "rr_qps2_n600"];
const pilotNames = ["pilot_rr_qps1", "pilot_rr_qps2"];
const slmNames = [
  "slm_single_qps0_3_n300", "slm_single_qps0_5_n300",
  "slm_single_qps0_7_n300", "slm_single_qps0_9_n300",
];
const excludedNames = ["slm_single_qps0_7_n300_invalid_oversized_prompt"];

const rrTarget = await buildRoutingWorkbook(
  await collectExperiments(rrNames), await collectExperiments(pilotNames));
const slmTarget = await buildSlmWorkbook(
  await collectExperiments(slmNames), await collectExperiments(excludedNames));
console.log(JSON.stringify({ rrTarget, slmTarget, previewDir }, null, 2));
