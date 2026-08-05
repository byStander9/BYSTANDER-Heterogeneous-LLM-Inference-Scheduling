#!/usr/bin/env python3
"""Run controlled ShareGPT experiments against KT Cloud vLLM endpoints.

The runner deliberately keeps raw client observations and vLLM Prometheus
histogram deltas separate.  It also records the BYSTANDER paper's state vector
at backend submission time.  Because stock vLLM does not expose the identities
or token lengths of inflight requests, those lengths are tracked by this client
after being measured with the backend's /tokenize API.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from proxy_request_qps import (convert_to_openai_messages,
                               load_dataset_auto)


HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
)
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "process_start_time_seconds",
)
SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)(?:\s+\d+)?$")
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"])*)"')


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def percentile(values: list[float], q: float) -> float:
    """Linear percentile, matching common client-side latency reporting."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def paper_percentile(values: list[int], q: float) -> float:
    """Percentile convention already used by BYSTANDER's SLM prompt code."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(int(len(ordered) * q), len(ordered) - 1)])


def parse_prometheus(text: str) -> dict[str, Any]:
    """Parse the subset of Prometheus text needed by this experiment."""
    values: dict[str, float] = {}
    histograms = {
        name: {"buckets": {}, "sum": 0.0, "count": 0.0}
        for name in HISTOGRAMS
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        labels = dict(LABEL_RE.findall(match.group("labels") or ""))
        matched_histogram = False
        for base in HISTOGRAMS:
            if name == f"{base}_bucket":
                bound_text = labels.get("le", "+Inf")
                bound = math.inf if bound_text == "+Inf" else float(bound_text)
                histograms[base]["buckets"][bound] = (
                    histograms[base]["buckets"].get(bound, 0.0) + value)
                matched_histogram = True
                break
            if name == f"{base}_sum":
                histograms[base]["sum"] += value
                matched_histogram = True
                break
            if name == f"{base}_count":
                histograms[base]["count"] += value
                matched_histogram = True
                break
        if not matched_histogram:
            values[name] = values.get(name, 0.0) + value
    return {"values": values, "histograms": histograms}


def histogram_delta(start: dict[str, Any], end: dict[str, Any],
                    name: str) -> dict[str, Any]:
    before = start["histograms"][name]
    after = end["histograms"][name]
    bounds = set(before["buckets"]) | set(after["buckets"])
    return {
        "buckets": {
            bound: after["buckets"].get(bound, 0.0)
            - before["buckets"].get(bound, 0.0)
            for bound in bounds
        },
        "sum": after["sum"] - before["sum"],
        "count": after["count"] - before["count"],
    }


def histogram_quantile(delta: dict[str, Any], q: float) -> tuple[float, float, float]:
    """Return interpolated quantile and its enclosing bucket bounds."""
    count = delta["count"]
    if count <= 0:
        return 0.0, 0.0, 0.0
    target = count * q
    previous_count = 0.0
    previous_bound = 0.0
    for bound, cumulative in sorted(delta["buckets"].items()):
        cumulative = max(cumulative, previous_count)
        if cumulative >= target:
            if math.isinf(bound):
                return previous_bound, previous_bound, math.inf
            in_bucket = cumulative - previous_count
            if in_bucket <= 0:
                estimate = bound
            else:
                fraction = (target - previous_count) / in_bucket
                estimate = previous_bound + (bound - previous_bound) * fraction
            return estimate, previous_bound, bound
        previous_count = cumulative
        previous_bound = bound
    return previous_bound, previous_bound, math.inf


def serializable_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "values": snapshot["values"],
        "histograms": {
            name: {
                "sum": hist["sum"],
                "count": hist["count"],
                "buckets": {
                    "+Inf" if math.isinf(bound) else str(bound): value
                    for bound, value in hist["buckets"].items()
                },
            }
            for name, hist in snapshot["histograms"].items()
        },
    }


def state_features(token_lengths: list[int]) -> dict[str, float]:
    return {
        "inflight_count": len(token_lengths),
        "inflight_p99_tokens": paper_percentile(token_lengths, 0.99),
        "inflight_p90_tokens": paper_percentile(token_lengths, 0.90),
        "inflight_p75_tokens": paper_percentile(token_lengths, 0.75),
        "inflight_p50_tokens": paper_percentile(token_lengths, 0.50),
        "inflight_p25_tokens": paper_percentile(token_lengths, 0.25),
    }


def select_compatible_requests(requests: list[PreparedRequest], total: int,
                               max_model_len: int,
                               max_tokens: int) -> tuple[list[PreparedRequest], int]:
    selected = []
    skipped = 0
    for item in requests:
        if item.prompt_tokens + max_tokens > max_model_len:
            skipped += 1
            continue
        selected.append(item)
        if len(selected) >= total:
            break
    for request_id, item in enumerate(selected):
        item.request_id = request_id
    return selected, skipped


@dataclass
class PreparedRequest:
    request_id: int
    dataset_index: int
    messages: list[dict[str, str]]
    prompt_text: str
    prompt_tokens: int = 0
    tokenize_error: str = ""


class StateTracker:
    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
        self.latest: list[dict[str, Any]] = [{} for _ in endpoints]
        self.inflight: list[dict[int, int]] = [{} for _ in endpoints]
        self.lock = asyncio.Lock()

    async def snapshot_and_register(self, endpoint_index: int,
                                    request_id: int,
                                    prompt_tokens: int) -> dict[str, Any]:
        async with self.lock:
            latest = dict(self.latest[endpoint_index])
            lengths = list(self.inflight[endpoint_index].values())
            features = state_features(lengths)
            features.update({
                "vllm_running": latest.get("vllm:num_requests_running", -1),
                "vllm_waiting": latest.get("vllm:num_requests_waiting", -1),
                "kv_cache_usage_perc": latest.get("vllm:kv_cache_usage_perc", -1),
                "metrics_observed_at": latest.get("observed_at", ""),
                "metrics_age_ms": max(
                    0.0,
                    (time.perf_counter() - latest.get("perf_counter", time.perf_counter()))
                    * 1000,
                ),
                "inflight_prompt_tokens_json": json.dumps(lengths),
            })
            self.inflight[endpoint_index][request_id] = prompt_tokens
            return features

    async def unregister(self, endpoint_index: int, request_id: int) -> None:
        async with self.lock:
            self.inflight[endpoint_index].pop(request_id, None)


async def fetch_metrics(client: httpx.AsyncClient, endpoint: str) -> tuple[str, dict[str, Any]]:
    response = await client.get(f"{endpoint}/metrics")
    response.raise_for_status()
    return response.text, parse_prometheus(response.text)


async def fetch_metrics_with_retry(
        client: httpx.AsyncClient, endpoint: str, attempts: int = 3,
        delay_s: float = 1.0) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fetch_metrics(client, endpoint)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(delay_s)
    assert last_error is not None
    raise last_error


async def metrics_sampler(clients: list[httpx.AsyncClient], tracker: StateTracker,
                          interval: float, rows: list[dict[str, Any]],
                          stop: asyncio.Event, started: float) -> None:
    while not stop.is_set():
        scrape_started = time.perf_counter()
        for index, (client, endpoint) in enumerate(zip(clients, tracker.endpoints)):
            row: dict[str, Any] = {
                "timestamp": now_iso(),
                "elapsed_s": time.perf_counter() - started,
                "endpoint_index": index,
                "endpoint": endpoint,
                "scrape_error": "",
            }
            try:
                _, snapshot = await fetch_metrics(client, endpoint)
                row.update({name: snapshot["values"].get(name, 0.0) for name in GAUGES})
                async with tracker.lock:
                    tracker.latest[index] = {
                        **{name: row[name] for name in GAUGES},
                        "observed_at": row["timestamp"],
                        "perf_counter": time.perf_counter(),
                    }
            except Exception as exc:  # retain a time-series row for failed scrapes
                row["scrape_error"] = str(exc)
            rows.append(row)
        delay = interval - (time.perf_counter() - scrape_started)
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.01, delay))
        except asyncio.TimeoutError:
            pass


async def tokenize_requests(requests: list[PreparedRequest], endpoint: str,
                            model: str, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=60.0) as client:
        async def tokenize(item: PreparedRequest) -> None:
            async with semaphore:
                try:
                    response = await client.post(
                        f"{endpoint}/tokenize",
                        json={"model": model, "messages": item.messages},
                    )
                    response.raise_for_status()
                    item.prompt_tokens = int(response.json()["count"])
                except Exception as exc:
                    item.prompt_tokens = max(1, len(item.prompt_text) // 4)
                    item.tokenize_error = str(exc)
        await asyncio.gather(*(tokenize(item) for item in requests))


async def send_request(item: PreparedRequest, endpoint_index: int,
                       endpoint: str, client: httpx.AsyncClient,
                       tracker: StateTracker, semaphore: asyncio.Semaphore,
                       model: str, max_tokens: int, temperature: float,
                       scheduled_elapsed_s: float, experiment_started: float,
                       mode: str, qps: float,
                       proxy_base_url: str | None = None) -> dict[str, Any]:
    async with semaphore:
        state = await tracker.snapshot_and_register(
            endpoint_index, item.request_id, item.prompt_tokens)
        submitted_perf = time.perf_counter()
        submitted_at = now_iso()
        ttft_perf: float | None = None
        http_status = 0
        completion_tokens = 0
        usage_prompt_tokens = 0
        routed_endpoint_index = -1
        routed_endpoint = ""
        error = ""
        try:
            payload = {
                "model": model,
                "messages": item.messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            timeout = httpx.Timeout(900.0, connect=30.0, read=900.0, write=30.0)
            request_base = proxy_base_url.rstrip("/") if proxy_base_url else endpoint
            async with client.stream(
                    "POST", f"{request_base}/v1/chat/completions", json=payload,
                    headers={"X-Bystander-Request-ID": str(item.request_id)},
                    timeout=timeout) as response:
                http_status = response.status_code
                routed_endpoint_index = int(
                    response.headers.get("X-Bystander-Endpoint-Index", -1))
                routed_endpoint = response.headers.get("X-Bystander-Endpoint", "")
                if http_status != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    error = f"HTTP {http_status}: {body[:500]}"
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        for choice in event.get("choices", []):
                            delta = choice.get("delta", {})
                            first_piece = delta.get("content") or delta.get("reasoning_content")
                            if ttft_perf is None and first_piece:
                                ttft_perf = time.perf_counter()
                        usage = event.get("usage") or {}
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        usage_prompt_tokens = usage.get("prompt_tokens", usage_prompt_tokens)
        except Exception as exc:
            error = str(exc)
        finally:
            completed_perf = time.perf_counter()
            completed_at = now_iso()
            await tracker.unregister(endpoint_index, item.request_id)

        e2e_ms = (completed_perf - submitted_perf) * 1000
        ttft_ms = ((ttft_perf - submitted_perf) * 1000
                   if ttft_perf is not None else e2e_ms)
        return {
            "request_id": item.request_id,
            "dataset_index": item.dataset_index,
            "mode": mode,
            "target_qps": qps,
            "endpoint_index": endpoint_index,
            "endpoint": endpoint,
            "proxy_base_url": proxy_base_url or "",
            "routed_endpoint_index": routed_endpoint_index,
            "routed_endpoint": routed_endpoint,
            "rr_route_verified": int(
                not proxy_base_url or routed_endpoint_index == endpoint_index),
            "scheduled_elapsed_s": scheduled_elapsed_s,
            "actual_submit_elapsed_s": submitted_perf - experiment_started,
            "arrival_lag_ms": (submitted_perf - experiment_started - scheduled_elapsed_s) * 1000,
            "submitted_at": submitted_at,
            "completed_at": completed_at,
            "http_status": http_status,
            "success": int(http_status == 200 and not error),
            "client_ttft_ms": ttft_ms,
            "client_e2e_ms": e2e_ms,
            "ttft_observed": int(ttft_perf is not None),
            "prompt_tokens_tokenize_api": item.prompt_tokens,
            "prompt_tokens_usage": usage_prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_chars": len(item.prompt_text),
            "prompt_first150": item.prompt_text[:150].replace("\n", " "),
            "prompt_last150": item.prompt_text[-150:].replace("\n", " "),
            "tokenize_error": item.tokenize_error,
            "state_source": "vllm_prometheus_gauges+client_registry_tokenize_api",
            **state,
            "error": error,
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_progress(path: Path, rows: list[dict[str, Any]], total: int,
                   started: float) -> None:
    successful = sum(int(row.get("success", 0)) for row in rows)
    progress = {
        "updated_at": now_iso(),
        "completed": len(rows),
        "total": total,
        "successful": successful,
        "failed": len(rows) - successful,
        "elapsed_s": time.perf_counter() - started,
    }
    path.write_text(json.dumps(progress, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def build_internal_rows(starts: list[dict[str, Any]], ends: list[dict[str, Any]],
                        endpoints: list[str]) -> list[dict[str, Any]]:
    rows = []
    combined: dict[str, dict[str, Any]] = {
        name: {"buckets": {}, "sum": 0.0, "count": 0.0}
        for name in HISTOGRAMS
    }
    for index, endpoint in enumerate(endpoints):
        restarted = (starts[index]["values"].get("process_start_time_seconds") !=
                     ends[index]["values"].get("process_start_time_seconds"))
        for metric in HISTOGRAMS:
            delta = histogram_delta(starts[index], ends[index], metric)
            estimate, lower, upper = histogram_quantile(delta, 0.99)
            rows.append({
                "scope": f"endpoint_{index + 1}",
                "endpoint": endpoint,
                "metric": metric,
                "count_delta": delta["count"],
                "sum_delta_s": delta["sum"],
                "average_s": delta["sum"] / delta["count"] if delta["count"] > 0 else 0,
                "p99_histogram_estimate_s": estimate,
                "p99_bucket_lower_s": lower,
                "p99_bucket_upper_s": "+Inf" if math.isinf(upper) else upper,
                "process_restarted": int(restarted),
            })
            if not restarted:
                combined[metric]["sum"] += delta["sum"]
                combined[metric]["count"] += delta["count"]
                for bound, value in delta["buckets"].items():
                    combined[metric]["buckets"][bound] = (
                        combined[metric]["buckets"].get(bound, 0.0) + value)
    for metric, delta in combined.items():
        estimate, lower, upper = histogram_quantile(delta, 0.99)
        rows.append({
            "scope": "combined",
            "endpoint": "all",
            "metric": metric,
            "count_delta": delta["count"],
            "sum_delta_s": delta["sum"],
            "average_s": delta["sum"] / delta["count"] if delta["count"] > 0 else 0,
            "p99_histogram_estimate_s": estimate,
            "p99_bucket_lower_s": lower,
            "p99_bucket_upper_s": "+Inf" if math.isinf(upper) else upper,
            "process_restarted": 0,
        })
    return rows


def build_client_summary(rows: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    successful = [row for row in rows if row["success"]]
    ttft = [row["client_ttft_ms"] for row in successful]
    e2e = [row["client_e2e_ms"] for row in successful]
    return {
        "requests": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0,
        "duration_s": duration_s,
        "achieved_completion_qps": len(successful) / duration_s if duration_s else 0,
        "client_ttft_average_ms": statistics.fmean(ttft) if ttft else 0,
        "client_ttft_p50_ms": percentile(ttft, 0.50),
        "client_ttft_p99_ms": percentile(ttft, 0.99),
        "client_e2e_average_ms": statistics.fmean(e2e) if e2e else 0,
        "client_e2e_p50_ms": percentile(e2e, 0.50),
        "client_e2e_p99_ms": percentile(e2e, 0.99),
        "max_vllm_waiting": max((row.get("vllm_waiting", -1) for row in rows), default=-1),
        "max_router_inflight": max((row.get("inflight_count", 0) for row in rows), default=0),
    }


async def run(args: argparse.Namespace) -> None:
    endpoints = [endpoint.rstrip("/") for endpoint in args.endpoints]
    if args.mode == "rr" and len(endpoints) < 2:
        raise SystemExit("rr mode requires at least two endpoints")
    if args.mode == "single":
        endpoints = endpoints[:1]

    records = load_dataset_auto(args.dataset, start_index=args.start_index,
                                limit=args.total * 2)
    prepared: list[PreparedRequest] = []
    for offset, record in enumerate(records):
        conversations = record.get("conversations") or record.get("conversation") or []
        messages = convert_to_openai_messages(conversations, max_turns=args.max_turns)
        if not messages:
            continue
        prompt_text = next((message["content"] for message in reversed(messages)
                            if message["role"] == "user"), "")
        if not prompt_text:
            continue
        prepared.append(PreparedRequest(len(prepared), args.start_index + offset,
                                        messages, prompt_text))
        if len(prepared) >= args.total * 2:
            break
    if len(prepared) < args.total:
        raise SystemExit(f"only {len(prepared)} usable requests were found")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{now_iso()}] tokenizing {len(prepared)} requests via {endpoints[0]}/tokenize", flush=True)
    await tokenize_requests(prepared, endpoints[0], args.model, args.tokenize_concurrency)
    prepared, oversized_skipped = select_compatible_requests(
        prepared, args.total, args.max_model_len, args.max_tokens)
    if len(prepared) < args.total:
        raise SystemExit(
            f"only {len(prepared)} requests fit max_model_len={args.max_model_len}")

    limits = httpx.Limits(max_connections=args.max_concurrent,
                          max_keepalive_connections=min(args.max_concurrent, 100))
    clients = [httpx.AsyncClient(timeout=60.0, limits=limits) for _ in endpoints]
    tracker = StateTracker(endpoints)
    starts_text: list[str] = []
    starts: list[dict[str, Any]] = []
    for client, endpoint in zip(clients, endpoints):
        raw, parsed = await fetch_metrics(client, endpoint)
        starts_text.append(raw)
        starts.append(parsed)

    started_wall = now_iso()
    started = time.perf_counter()
    metrics_rows: list[dict[str, Any]] = []
    stop_sampler = asyncio.Event()
    sampler = asyncio.create_task(metrics_sampler(
        clients, tracker, args.metrics_interval, metrics_rows, stop_sampler, started))
    await asyncio.sleep(min(0.25, args.metrics_interval * 1.1))

    rng = random.Random(args.seed)
    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = []
    checkpoint_rows: list[dict[str, Any]] = []

    def save_completed(task: asyncio.Task) -> None:
        try:
            checkpoint_rows.append(task.result())
        except Exception as exc:
            checkpoint_rows.append({
                "request_id": -1,
                "success": 0,
                "error": f"unhandled request task error: {exc}",
            })
        completed = len(checkpoint_rows)
        if completed % args.checkpoint_every == 0 or completed == len(prepared):
            ordered = sorted(checkpoint_rows,
                             key=lambda row: row.get("request_id", -1))
            write_csv(output_dir / "client_requests_checkpoint.csv", ordered)
            write_csv(output_dir / "vllm_timeseries_checkpoint.csv", metrics_rows)
            write_progress(output_dir / "progress.json", checkpoint_rows,
                           len(prepared), started)
            print(f"[{now_iso()}] CHECKPOINT completed={completed}/{len(prepared)}",
                  flush=True)

    scheduled = 0.0
    for item in prepared:
        scheduled += rng.expovariate(args.qps)
        delay = scheduled - (time.perf_counter() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        endpoint_index = item.request_id % len(endpoints) if args.mode == "rr" else 0
        task = asyncio.create_task(send_request(
            item, endpoint_index, endpoints[endpoint_index], clients[endpoint_index],
            tracker, semaphore, args.model, args.max_tokens, args.temperature,
            scheduled, started, args.mode, args.qps, args.proxy_base_url))
        task.add_done_callback(save_completed)
        tasks.append(task)
        if item.request_id and item.request_id % 100 == 0:
            done = sum(task.done() for task in tasks)
            print(f"[{now_iso()}] submitted={item.request_id + 1}/{len(prepared)} completed~={done}", flush=True)

    rows = await asyncio.gather(*tasks)
    duration_s = time.perf_counter() - started
    stop_sampler.set()
    await sampler

    ends_text: list[str] = []
    ends: list[dict[str, Any]] = []
    final_metrics_errors: list[str] = []
    for index, (client, endpoint) in enumerate(zip(clients, endpoints)):
        try:
            raw, parsed = await fetch_metrics_with_retry(client, endpoint)
            final_metrics_errors.append("")
        except Exception as exc:
            error = str(exc)
            raw = f"# Final metrics unavailable: {error}\n"
            parsed = starts[index]
            final_metrics_errors.append(error)
        ends_text.append(raw)
        ends.append(parsed)
    for client in clients:
        await client.aclose()

    write_csv(output_dir / "client_requests.csv", sorted(rows, key=lambda row: row["request_id"]))
    write_csv(output_dir / "vllm_timeseries.csv", metrics_rows)
    internal_rows = build_internal_rows(starts, ends, endpoints)
    write_csv(output_dir / "vllm_histogram_summary.csv", internal_rows)
    for index, (start_text, end_text) in enumerate(zip(starts_text, ends_text), start=1):
        (output_dir / f"endpoint_{index}_metrics_start.prom").write_text(start_text, encoding="utf-8")
        (output_dir / f"endpoint_{index}_metrics_end.prom").write_text(end_text, encoding="utf-8")
        (output_dir / f"endpoint_{index}_metrics_parsed.json").write_text(
            json.dumps({"start": serializable_snapshot(starts[index - 1]),
                        "end": serializable_snapshot(ends[index - 1])},
                       indent=2), encoding="utf-8")

    summary = {
        "experiment_started_at": started_wall,
        "experiment_completed_at": now_iso(),
        "mode": args.mode,
        "endpoints": endpoints,
        "proxy_base_url": args.proxy_base_url or "",
        "dataset": str(Path(args.dataset).resolve()),
        "start_index": args.start_index,
        "target_qps": args.qps,
        "max_tokens": args.max_tokens,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "metrics_interval_s": args.metrics_interval,
        "checkpoint_every": args.checkpoint_every,
        "final_metrics_errors": final_metrics_errors,
        "tokenize_fallback_count": sum(bool(item.tokenize_error) for item in prepared),
        "oversized_prompts_skipped": oversized_skipped,
        "state_measurement_note": (
            "running/waiting/kv_cache are cached vLLM Prometheus gauges; inflight token "
            "lengths are tracked by this exclusive experiment client from /tokenize counts"),
        "rr_proxy_route_mismatches": sum(
            1 for row in rows if args.proxy_base_url and not row["rr_route_verified"]),
        **build_client_summary(rows, duration_s),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rr", "single"), required=True)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--proxy-base-url",
                        help="actual RR proxy base URL; endpoint metrics are still sampled directly")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--model", default="Qwen3-4B")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrent", type=int, default=500)
    parser.add_argument("--metrics-interval", type=float, default=0.2)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--tokenize-concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    asyncio.run(run(build_parser().parse_args()))
