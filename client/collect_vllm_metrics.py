#!/usr/bin/env python3
import argparse
import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx


METRIC_COLUMNS = {
    "vllm:num_requests_running": "num_running",
    "vllm:num_requests_waiting": "num_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:request_success_total": "request_success_total",
    "vllm:num_preemptions_total": "num_preemptions_total",
}


def parse_metrics_text(text: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            series, raw_value = line.rsplit(None, 1)
            metric_name = series.split("{", 1)[0]
            values[metric_name].append(float(raw_value))
        except ValueError:
            continue
    return dict(values)


def snapshot_from_text(text: str) -> dict[str, float | str]:
    metrics = parse_metrics_text(text)
    process_start = metrics["process_start_time_seconds"][0]
    snapshot: dict[str, float | str] = {
        "replica_id": f"{process_start:.3f}",
    }
    for metric_name, column_name in METRIC_COLUMNS.items():
        snapshot[column_name] = sum(metrics.get(metric_name, []))
    return snapshot


def collect(metrics_url: str, interval: float, duration: float,
            output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "elapsed_s",
        "scrape_index",
        "replica_id",
        *METRIC_COLUMNS.values(),
    ]
    started = time.perf_counter()
    next_scrape = started
    scrape_index = 0

    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        with httpx.Client(timeout=10.0) as client:
            while time.perf_counter() - started < duration:
                now = time.perf_counter()
                if now < next_scrape:
                    time.sleep(next_scrape - now)
                try:
                    response = client.get(metrics_url)
                    response.raise_for_status()
                    row = snapshot_from_text(response.text)
                    row.update({
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "elapsed_s": f"{time.perf_counter() - started:.6f}",
                        "scrape_index": scrape_index,
                    })
                    writer.writerow(row)
                    target.flush()
                except Exception as exc:
                    print(f"metrics scrape {scrape_index} failed: {exc}",
                          flush=True)
                scrape_index += 1
                next_scrape += interval

    print(f"metrics saved: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.url, args.interval, args.duration, args.output)


if __name__ == "__main__":
    main()
