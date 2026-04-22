"""
Motivation Experiments for Paper

Experiment 1: WRR의 정적 가중치와 Bursty Traffic 대응 실패
  - 이기종 GPU 환경에서 WRR의 고정 가중치가 Burst 트래픽 + 이질적 워크로드에서
    약한 GPU의 Queue Explosion을 유발함을 시계열 그래프로 증명

Experiment 2: SQF의 오판(Mis-scheduling)과 Regret 분석
  - SQF가 큐 개수만 보고 결정할 때 토큰 길이·하드웨어 성능 차이를 무시하여
    발생하는 Scheduling Error Rate와 Relative Latency Penalty를 정량화

Usage:
  python motivation_experiments.py --exp 1   # Experiment 1만 실행
  python motivation_experiments.py --exp 2   # Experiment 2만 실행
  python motivation_experiments.py --exp all # 둘 다 실행
"""

import asyncio
import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

# ============================================================
# Configuration
# ============================================================

BACKEND_SERVERS = [
    {"host": "BACKEND_RTX3090_HOST", "port": 18001, "name": "RTX3090_SERVER", "gpu_type": "RTX3090"},
    {"host": "BACKEND_RTX4090_HOST", "port": 45652, "name": "RTX4090_SERVER", "gpu_type": "RTX4090"},
    {"host": "BACKEND_RTX5090_HOST", "port": 18004, "name": "RTX5090_SERVER", "gpu_type": "RTX5090"},
]

GPU_PERF_RATIOS = {"RTX3090": 1.0, "RTX4090": 2.0, "RTX5090": 2.5}

GPU_DISPLAY_NAMES = {
    "RTX3090": "GPU-A (RTX 3090)",
    "RTX4090": "GPU-B (RTX 4090)",
    "RTX5090": "GPU-C (RTX 5090)",
}

GPU_COLORS = {
    "RTX3090": "#e74c3c",
    "RTX4090": "#f39c12",
    "RTX5090": "#2ecc71",
}

RESULTS_BASE = Path("motivation_results")
RESULTS_BASE.mkdir(exist_ok=True)
RESULTS_DIR = RESULTS_BASE  # main()에서 run_id별 하위 디렉토리로 재설정

LMSYS_PROMPTS = [
    "What is the capital of France?",
    "Explain the concept of supply and demand in economics.",
    "How does photosynthesis work?",
    "What are the main differences between Python and Java?",
    "Describe the water cycle briefly.",
    "What is machine learning?",
    "How do vaccines work?",
    "Explain the theory of relativity in simple terms.",
    "What causes earthquakes?",
    "How does the internet work?",
    "What is the significance of the Turing test?",
    "Explain the greenhouse effect.",
    "What is blockchain technology?",
    "How does a refrigerator work?",
    "What is the difference between weather and climate?",
]

SHAREGPT_PROMPTS = [
    (
        "You are a senior software engineer. I want you to write a comprehensive, production-ready "
        "implementation of a distributed task queue system in Python. The system should support: "
        "1) Task prioritization with multiple priority levels, 2) Dead letter queues for failed tasks, "
        "3) Retry logic with exponential backoff, 4) Worker health monitoring, "
        "5) Graceful shutdown with in-progress task completion, "
        "6) Metrics collection and reporting. "
        "Please provide the complete implementation with all classes, error handling, "
        "type hints, and detailed docstrings. Also include unit tests for each component. "
        "Make sure the code follows SOLID principles and is ready for a code review. "
        "Additionally, explain the architectural decisions you made and why you chose them "
        "over alternative approaches. Include a discussion of trade-offs between consistency "
        "and availability in the context of distributed systems. Finally, provide a deployment "
        "guide with Docker configuration and Kubernetes manifests."
    ),
    (
        "I need you to write a detailed research paper outline about the impact of large language models "
        "on software engineering practices. Cover the following sections in depth: "
        "1) Introduction with historical context of AI in SE, "
        "2) Literature review of at least 20 related works, "
        "3) Methodology for empirical study design, "
        "4) Analysis framework for measuring developer productivity, "
        "5) Discussion of ethical implications, "
        "6) Future research directions. "
        "For each section, provide detailed bullet points, potential data sources, "
        "and analysis techniques. Include a comprehensive bibliography format. "
        "Also discuss the limitations of current evaluation metrics and propose new ones. "
        "Consider the socioeconomic impact on the software development workforce "
        "and the implications for computer science education curricula."
    ),
    (
        "Design and implement a complete real-time collaborative text editor from scratch. "
        "Use Conflict-free Replicated Data Types (CRDTs) for conflict resolution. "
        "The implementation should include: "
        "1) A CRDT-based document model supporting insert, delete, and format operations, "
        "2) An operational transformation layer for real-time synchronization, "
        "3) A WebSocket server for peer communication, "
        "4) Undo/redo support with causal ordering, "
        "5) Cursor presence and selection sharing, "
        "6) Offline editing with automatic merge on reconnect, "
        "7) Version history with branch and merge capabilities. "
        "Provide complete TypeScript implementation with comprehensive error handling. "
        "Include performance benchmarks comparing your CRDT implementation against OT-based solutions. "
        "Discuss the theoretical guarantees of your approach in terms of convergence and intention preservation."
    ),
    (
        "Create a comprehensive machine learning pipeline for natural language processing "
        "that includes data collection, preprocessing, feature engineering, model training, "
        "hyperparameter optimization, evaluation, and deployment. The pipeline should handle "
        "multiple languages, support transfer learning from pre-trained models, implement "
        "active learning for efficient annotation, and include monitoring for data drift "
        "and model degradation. Provide detailed code for each component along with "
        "architecture diagrams, performance benchmarks, and a thorough discussion of "
        "the trade-offs between different modeling approaches. Include implementations "
        "of attention mechanisms, transformer architectures, and efficient fine-tuning "
        "techniques like LoRA and QLoRA. Discuss the memory optimization strategies "
        "including gradient checkpointing, mixed precision training, and model parallelism."
    ),
    (
        "Write an exhaustive comparison of modern database architectures including "
        "relational (PostgreSQL), document (MongoDB), graph (Neo4j), time-series (TimescaleDB), "
        "and vector databases (Pinecone, Milvus). For each database type, provide: "
        "1) Internal architecture and storage engine details, "
        "2) Query optimization strategies, "
        "3) Indexing mechanisms and their complexity, "
        "4) Replication and sharding approaches, "
        "5) ACID compliance and consistency models, "
        "6) Performance benchmarks under different workload patterns, "
        "7) Real-world use cases with implementation examples, "
        "8) Cost analysis for cloud deployments. "
        "Include SQL and NoSQL query examples for common operations. "
        "Discuss the CAP theorem implications for each database and provide "
        "decision frameworks for choosing the right database for different scenarios. "
        "Also cover emerging trends like serverless databases, multi-model databases, "
        "and the convergence of OLTP and OLAP workloads."
    ),
]

MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=30.0)
HTTP_LIMITS = httpx.Limits(max_connections=2000, max_keepalive_connections=500)


# ============================================================
# Utility Functions
# ============================================================

def parse_prometheus_metrics(text: str) -> Dict:
    metrics = {"num_requests_running": 0, "num_requests_waiting": 0}
    m = re.search(r'vllm:num_requests_running(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)', text)
    if m:
        metrics["num_requests_running"] = int(float(m.group(1)))
    m = re.search(r'vllm:num_requests_waiting(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)', text)
    if m:
        metrics["num_requests_waiting"] = int(float(m.group(1)))
    return metrics


async def fetch_queue_lengths(client: httpx.AsyncClient) -> Dict[str, Dict[str, int]]:
    """각 서버의 running/waiting 큐 길이 수집"""
    results = {}
    tasks = []
    for server in BACKEND_SERVERS:
        url = f"http://{server['host']}:{server['port']}/metrics"
        tasks.append(client.get(url))

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for server, resp in zip(BACKEND_SERVERS, responses):
        if isinstance(resp, Exception):
            results[server["name"]] = {"running": -1, "waiting": -1, "total": -1}
        else:
            m = parse_prometheus_metrics(resp.text)
            r = m["num_requests_running"]
            w = m["num_requests_waiting"]
            results[server["name"]] = {"running": r, "waiting": w, "total": r + w}
    return results


def build_chat_request(prompt: str, max_tokens: int = 512, stream: bool = True) -> Dict:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream,
    }


def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================================================
# Experiment 1: WRR Static Weights Failure under QPS Burst
#   (Discrete-Event Simulation with Congestion Effect)
# ============================================================

class WRRScheduler:
    """Weighted Round Robin with static weights"""

    def __init__(self, gpu_names: List[str], weights: List[int]):
        self.gpu_names = gpu_names
        self.weights = weights
        self.sequence = []
        for idx, w in enumerate(weights):
            self.sequence.extend([idx] * w)
        self.index = 0

    def next_gpu(self) -> str:
        idx = self.sequence[self.index]
        self.index = (self.index + 1) % len(self.sequence)
        return self.gpu_names[idx]


class GPUSimulator:
    """
    이기종 GPU의 M/G/1 큐잉 모델 (congestion-dependent service time).

    현실 모델링:
    - GPU는 동시 요청(큐 깊이)이 커지면 성능이 비선형적으로 저하됨
      (KV-cache 메모리 압박, 배치 스케줄링 오버헤드, 메모리 대역폭 포화)
    - 빠른 GPU일수록 리소스 활용도가 높아 congestion에 더 민감
    - 오프라인 프로파일링은 저부하 환경에서 측정 → 고부하 성능 저하를 반영 못함
    """

    def __init__(self, name: str, base_service_time: float, congestion_factor: float):
        """
        Args:
            base_service_time: 저부하(큐=0)에서의 평균 서비스 시간
            congestion_factor: 큐 깊이 1 증가당 서비스 시간 증가 비율
                               service_time = base * (1 + factor * queue_depth) * noise
        """
        self.name = name
        self.base_service_time = base_service_time
        self.congestion_factor = congestion_factor
        self.queue: List[dict] = []
        self.current_job: Optional[dict] = None
        self.current_job_finish_time: float = float("inf")
        self.completed: List[dict] = []

    def enqueue(self, job: dict):
        self.queue.append(job)

    def queue_length(self) -> int:
        running = 1 if self.current_job is not None else 0
        return running + len(self.queue)

    def running_count(self) -> int:
        return 1 if self.current_job is not None else 0

    def waiting_count(self) -> int:
        return len(self.queue)

    def _start_next(self, now: float):
        if self.current_job is None and self.queue:
            job = self.queue.pop(0)
            ql = self.queue_length()
            degradation = 1.0 + self.congestion_factor * ql
            noise = np.random.uniform(0.8, 1.2)
            service_time = self.base_service_time * degradation * noise
            job["service_start"] = now
            job["service_time"] = service_time
            job["queue_depth_at_start"] = ql
            self.current_job = job
            self.current_job_finish_time = now + service_time

    def advance(self, now: float):
        while self.current_job is not None and now >= self.current_job_finish_time:
            job = self.current_job
            job["finish_time"] = self.current_job_finish_time
            job["e2e"] = job["finish_time"] - job["arrival_time"]
            self.completed.append(job)
            self.current_job = None
            self.current_job_finish_time = float("inf")
            self._start_next(self.current_job_finish_time if self.current_job else now)
        if self.current_job is None:
            self._start_next(now)


def run_experiment1_sim(
    phase1_rps: float = 4.0,
    phase1_duration: float = 40.0,
    phase2_rps: float = 10.0,
    phase2_duration: float = 25.0,
    cooldown_duration: float = 200.0,
    wrr_weights: List[int] = None,
    snapshot_interval: float = 0.2,
):
    """
    Experiment 1 (Discrete-Event Simulation)

    WRR의 정적 가중치가 QPS burst에서 실패하는 메커니즘:

    1. 오프라인 프로파일링은 저부하에서 측정한 성능비로 가중치를 결정
       → 저부하 성능비 = 1:2:4 → weights 1:2:4
    2. GPU 처리 속도는 부하(큐 깊이)에 따라 비선형적으로 저하
       (KV-cache 메모리 압박, 배치 오버헤드, 메모리 대역폭 포화)
    3. 빠른 GPU(RTX5090)는 가중치가 커서 트래픽이 집중되고,
       큐가 깊어지면서 congestion effect가 가장 크게 작용
    4. WRR은 실시간 큐 상태를 보지 않으므로 부하 재분배 불가
       → 가장 빠른 GPU에서 Queue Explosion + Latency 급등 (positive feedback loop)
    """
    print("=" * 70)
    print("  Experiment 1: WRR Static Weights under QPS Burst")
    print("=" * 70)

    if wrr_weights is None:
        wrr_weights = [1, 2, 4]

    # ─── GPU 모델 ───
    # base_service_time: 저부하(큐 깊이 ≈ 0)에서의 서비스 시간
    #   → 이 값으로 측정한 성능비 = 1:2:4 = WRR 가중치
    #
    # congestion_factor: 큐 깊이당 서비스 시간 증가율
    #   빠른 GPU일수록 리소스 한계에 더 빨리 도달하여 factor가 높음
    #   - RTX3090: 느리지만 이미 자원 사용률이 낮음 → 약한 congestion
    #   - RTX5090: 빠르지만 높은 자원 사용률로 운용 → 강한 congestion
    #
    gpu_configs = [
        {"name": "RTX3090", "base_time": 1.0,  "congestion": 0.01},  # 큐10→ 1.1x
        {"name": "RTX4090", "base_time": 0.5,  "congestion": 0.025}, # 큐10→ 1.25x
        {"name": "RTX5090", "base_time": 0.25, "congestion": 0.045}, # 큐10→ 1.45x
    ]
    gpu_names = [g["name"] for g in gpu_configs]
    gpu_cfg_map = {g["name"]: g for g in gpu_configs}

    scheduler = WRRScheduler(gpu_names, wrr_weights)
    gpus = {g["name"]: GPUSimulator(g["name"], g["base_time"], g["congestion"])
            for g in gpu_configs}

    print(f"  GPU Model (base service time @ low load → congestion factor):")
    for g in gpu_configs:
        cap = 1.0 / g["base_time"]
        print(f"    {g['name']}: {g['base_time']:.2f}s (cap={cap:.1f} req/s), "
              f"congestion={g['congestion']:.2f}/depth "
              f"(queue=10 → {1+g['congestion']*10:.1f}x, queue=20 → {1+g['congestion']*20:.1f}x)")
    print(f"  Low-load perf ratio: 1 : 2 : 4  (= WRR weights {':'.join(str(w) for w in wrr_weights)})")
    total_w = sum(wrr_weights)
    for n, w in zip(gpu_names, wrr_weights):
        print(f"    {n}: gets {w/total_w*100:.1f}% of traffic")
    print(f"  Phase 1 (Normal): RPS={phase1_rps}, Duration={phase1_duration}s")
    print(f"  Phase 2 (Burst):  RPS={phase2_rps}, Duration={phase2_duration}s  (×{phase2_rps/phase1_rps:.1f} spike)")
    print(f"  Cooldown: {cooldown_duration}s")
    print()

    total_time = phase1_duration + phase2_duration + cooldown_duration

    # 이벤트 생성 (동일 워크로드, QPS만 변화)
    events = []
    t = 0.0
    req_id = 0
    while t < phase1_duration + phase2_duration:
        rps = phase1_rps if t < phase1_duration else phase2_rps
        phase = "Normal" if t < phase1_duration else "Burst"

        inter = np.random.exponential(1.0 / rps)
        t += inter
        if t >= phase1_duration + phase2_duration:
            break
        req_id += 1
        events.append({"id": req_id, "arrival_time": t, "phase": phase})

    print(f"  Generated {len(events)} requests "
          f"(Phase1: {sum(1 for e in events if e['phase']=='Normal')}, "
          f"Phase2: {sum(1 for e in events if e['phase']=='Burst')})")

    # 시뮬레이션
    queue_snapshots = []
    event_idx = 0
    sim_time = 0.0
    dt = 0.01
    next_snapshot = 0.0

    while sim_time <= total_time:
        while event_idx < len(events) and events[event_idx]["arrival_time"] <= sim_time:
            ev = events[event_idx]
            target_gpu = scheduler.next_gpu()
            ev["assigned_gpu"] = target_gpu
            gpus[target_gpu].enqueue(ev)
            event_idx += 1

        for gpu in gpus.values():
            gpu.advance(sim_time)

        if sim_time >= next_snapshot:
            snap = {"time": sim_time}
            for gn in gpu_names:
                ql = gpus[gn].queue_length()
                snap[gn] = ql
                snap[f"{gn}_running"] = gpus[gn].running_count()
                snap[f"{gn}_waiting"] = gpus[gn].waiting_count()
                cfg = gpu_cfg_map[gn]
                snap[f"{gn}_est_e2e"] = cfg["base_time"] * (ql + 1) * (1 + cfg["congestion"] * ql / 2)
            if sim_time < phase1_duration:
                snap["phase"] = "Normal"
            elif sim_time < phase1_duration + phase2_duration:
                snap["phase"] = "Burst"
            else:
                snap["phase"] = "Cooldown"
            queue_snapshots.append(snap)
            next_snapshot += snapshot_interval

        sim_time += dt

    # 결과 수집
    completion_log = []
    for gpu in gpus.values():
        for job in gpu.completed:
            completion_log.append({
                "request_id": job["id"],
                "gpu_type": job["assigned_gpu"],
                "e2e": job["e2e"],
                "phase": job["phase"],
                "arrival_time": job["arrival_time"],
                "finish_time": job["finish_time"],
                "service_time": job.get("service_time", 0),
                "queue_depth": job.get("queue_depth_at_start", 0),
            })
    completion_log.sort(key=lambda x: x["finish_time"])

    unfinished = sum(gpu.queue_length() for gpu in gpus.values())
    print(f"  Completed: {len(completion_log)}, Unfinished at sim end: {unfinished}")

    # GPU별 Phase 2 통계
    for gn in gpu_names:
        burst_jobs = [c for c in completion_log if c["gpu_type"] == gn and c["phase"] == "Burst"]
        if burst_jobs:
            avg_e2e = statistics.mean(j["e2e"] for j in burst_jobs)
            max_e2e = max(j["e2e"] for j in burst_jobs)
            avg_svc = statistics.mean(j["service_time"] for j in burst_jobs)
            avg_qd = statistics.mean(j["queue_depth"] for j in burst_jobs)
            print(f"    {gn} (Burst): avg_e2e={avg_e2e:.1f}s, max_e2e={max_e2e:.1f}s, "
                  f"avg_service={avg_svc:.2f}s, avg_queue_depth={avg_qd:.1f}")

    # 저장
    csv_path = RESULTS_DIR / "exp1_queue_snapshots.csv"
    snap_fields = (["time"] + gpu_names
                   + [f"{n}_running" for n in gpu_names]
                   + [f"{n}_waiting" for n in gpu_names]
                   + [f"{n}_est_e2e" for n in gpu_names]
                   + ["phase"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=snap_fields)
        writer.writeheader()
        for snap in queue_snapshots:
            writer.writerow(snap)
    print(f"  Queue snapshots saved: {csv_path}")

    comp_fields = ["request_id", "gpu_type", "e2e", "phase", "arrival_time", "finish_time", "service_time", "queue_depth"]
    comp_csv = RESULTS_DIR / "exp1_completions.csv"
    with open(comp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comp_fields)
        writer.writeheader()
        for c in completion_log:
            writer.writerow({k: c.get(k, "") for k in comp_fields})
    print(f"  Completion log saved: {comp_csv}")

    plot_experiment1(queue_snapshots, completion_log,
                     phase1_duration, phase1_duration + phase2_duration,
                     wrr_weights, gpu_names)

    return queue_snapshots, completion_log


# ============================================================
# Experiment 1 (Real GPU): WRR Static Weights under QPS Burst
# ============================================================

SERVER_MAP = {s["gpu_type"]: s for s in BACKEND_SERVERS}


async def _send_request(client: httpx.AsyncClient, server: dict, prompt: str,
                        max_tokens: int, req_id: int, experiment_start: float,
                        phase: str, results: list):
    """Send a non-streaming request and record E2E latency."""
    url = f"http://{server['host']}:{server['port']}/v1/chat/completions"
    send_time = time.time()
    try:
        resp = await client.post(url, json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        })
        finish_time = time.time()
        e2e = finish_time - send_time
        results.append({
            "request_id": req_id,
            "gpu_type": server["gpu_type"],
            "e2e": e2e,
            "phase": phase,
            "arrival_time": send_time - experiment_start,
            "finish_time": finish_time - experiment_start,
            "max_tokens": max_tokens,
            "status": resp.status_code,
        })
    except Exception as exc:
        finish_time = time.time()
        results.append({
            "request_id": req_id,
            "gpu_type": server["gpu_type"],
            "e2e": finish_time - send_time,
            "phase": phase,
            "arrival_time": send_time - experiment_start,
            "finish_time": finish_time - experiment_start,
            "max_tokens": max_tokens,
            "status": -1,
        })


async def _monitor_queues(client: httpx.AsyncClient, gpu_names: list,
                          experiment_start: float, stop_event: asyncio.Event,
                          snapshots: list, interval: float = 0.5):
    """Periodically poll /metrics to record queue lengths."""
    last_known = {gn: {"total": 0, "running": 0, "waiting": 0} for gn in gpu_names}
    while not stop_event.is_set():
        snap = {"time": time.time() - experiment_start}
        tasks = []
        for gn in gpu_names:
            srv = SERVER_MAP[gn]
            url = f"http://{srv['host']}:{srv['port']}/metrics"
            tasks.append(client.get(url, timeout=15.0))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for gn, resp in zip(gpu_names, responses):
            if isinstance(resp, Exception):
                snap[gn] = last_known[gn]["total"]
                snap[f"{gn}_running"] = last_known[gn]["running"]
                snap[f"{gn}_waiting"] = last_known[gn]["waiting"]
            else:
                m = parse_prometheus_metrics(resp.text)
                r = m["num_requests_running"]
                w = m["num_requests_waiting"]
                snap[gn] = r + w
                snap[f"{gn}_running"] = r
                snap[f"{gn}_waiting"] = w
                last_known[gn] = {"total": r + w, "running": r, "waiting": w}
        snapshots.append(snap)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def run_experiment1_real(
    phase1_rps: float = 2.0,
    phase1_duration: float = 30.0,
    phase2_rps: float = 6.0,
    phase2_duration: float = 30.0,
    cooldown_duration: float = 120.0,
    wrr_weights: List[int] = None,
    max_tokens: int = 256,
    monitor_interval: float = 0.5,
    use_heavy_prompts: bool = False,
):
    """
    Experiment 1 (Real GPU) — WRR 정적 가중치의 QPS Burst 대응 실패

    실제 vLLM 백엔드에 요청을 보내 큐 폭발 및 E2E latency 급등을 관측.
    """
    print("=" * 70)
    print("  Experiment 1 (Real GPU): WRR Static Weights under QPS Burst")
    print("=" * 70)

    if wrr_weights is None:
        wrr_weights = [1, 2, 4]

    gpu_names = [s["gpu_type"] for s in BACKEND_SERVERS]
    scheduler = WRRScheduler(gpu_names, wrr_weights)

    total_w = sum(wrr_weights)
    for n, w in zip(gpu_names, wrr_weights):
        print(f"    {n}: weight={w} ({w/total_w*100:.1f}% of traffic)")
    print(f"  Phase 1 (Normal): RPS={phase1_rps}, Duration={phase1_duration}s")
    print(f"  Phase 2 (Burst):  RPS={phase2_rps}, Duration={phase2_duration}s  (x{phase2_rps/phase1_rps:.1f})")
    print(f"  Cooldown: {cooldown_duration}s (no new requests)")
    prompt_pool = SHAREGPT_PROMPTS if use_heavy_prompts else LMSYS_PROMPTS
    print(f"  max_tokens={max_tokens}, prompts={'ShareGPT(heavy)' if use_heavy_prompts else 'LMSYS(light)'}")
    print()

    queue_snapshots: List[dict] = []
    completion_log: List[dict] = []
    stop_monitor = asyncio.Event()

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=30.0),
                                  limits=httpx.Limits(max_connections=500, max_keepalive_connections=200)) as client:
        # Warmup
        print("  Warming up backends...")
        warmup_tasks = []
        for gn in gpu_names:
            srv = SERVER_MAP[gn]
            warmup_tasks.append(client.post(
                f"http://{srv['host']}:{srv['port']}/v1/chat/completions",
                json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hi"}],
                      "max_tokens": 4, "stream": False},
                timeout=60.0,
            ))
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        print("  Warmup done.\n")

        experiment_start = time.time()

        # Start queue monitor
        monitor_task = asyncio.create_task(
            _monitor_queues(client, gpu_names, experiment_start, stop_monitor,
                            queue_snapshots, monitor_interval)
        )

        # Phase 1 + Phase 2: send requests
        req_id = 0
        pending_tasks = []
        total_send_duration = phase1_duration + phase2_duration
        phase_start = time.time()

        while True:
            elapsed = time.time() - phase_start
            if elapsed >= total_send_duration:
                break

            if elapsed < phase1_duration:
                rps = phase1_rps
                phase = "Normal"
            else:
                rps = phase2_rps
                phase = "Burst"

            interval_wait = np.random.exponential(1.0 / rps)
            await asyncio.sleep(interval_wait)

            elapsed = time.time() - phase_start
            if elapsed >= total_send_duration:
                break

            req_id += 1
            target_gpu = scheduler.next_gpu()
            server = SERVER_MAP[target_gpu]
            prompt = random.choice(prompt_pool)

            task = asyncio.create_task(
                _send_request(client, server, prompt, max_tokens, req_id,
                              experiment_start, phase, completion_log)
            )
            pending_tasks.append(task)

            if req_id % 20 == 0:
                qls = {gn: queue_snapshots[-1].get(gn, 0) if queue_snapshots else 0
                       for gn in gpu_names}
                print(f"    [{elapsed:.0f}s] sent={req_id}, phase={phase}, "
                      f"queues: {' | '.join(f'{n}={v}' for n, v in qls.items())}")

        phase1_count = sum(1 for t in completion_log if t.get("phase") == "Normal")
        burst_count = req_id - phase1_count
        print(f"\n  All {req_id} requests sent (Phase1≈{phase1_count}, Phase2≈{burst_count})")
        print(f"  Waiting for cooldown ({cooldown_duration}s) and pending completions...")

        # Cooldown: wait for pending requests to complete
        cooldown_start = time.time()
        done_count = 0
        while True:
            remaining = [t for t in pending_tasks if not t.done()]
            if not remaining:
                print(f"  All requests completed at +{time.time()-experiment_start:.0f}s")
                break
            cooldown_elapsed = time.time() - cooldown_start
            if cooldown_elapsed >= cooldown_duration:
                print(f"  Cooldown timeout. {len(remaining)} requests still pending.")
                break
            new_done = len(pending_tasks) - len(remaining)
            if new_done > done_count + 10:
                done_count = new_done
                qls = {gn: queue_snapshots[-1].get(gn, 0) if queue_snapshots else 0
                       for gn in gpu_names}
                print(f"    [cooldown +{cooldown_elapsed:.0f}s] done={new_done}/{req_id}, "
                      f"queues: {' | '.join(f'{n}={v}' for n, v in qls.items())}")
            await asyncio.sleep(1.0)

        # Extra monitoring for a few more seconds after all done
        await asyncio.sleep(3.0)
        stop_monitor.set()
        await monitor_task

    # Sort completion log
    completion_log.sort(key=lambda x: x["finish_time"])

    # Stats
    print(f"\n  Completed: {len(completion_log)}")
    for gn in gpu_names:
        gpu_jobs = [c for c in completion_log if c["gpu_type"] == gn]
        burst_jobs = [c for c in gpu_jobs if c["phase"] == "Burst"]
        if burst_jobs:
            avg_e2e = statistics.mean(j["e2e"] for j in burst_jobs)
            max_e2e = max(j["e2e"] for j in burst_jobs)
            print(f"    {gn} (Burst {len(burst_jobs)} reqs): avg_e2e={avg_e2e:.1f}s, max_e2e={max_e2e:.1f}s")
        if gpu_jobs:
            avg_all = statistics.mean(j["e2e"] for j in gpu_jobs)
            print(f"    {gn} (All   {len(gpu_jobs)} reqs): avg_e2e={avg_all:.1f}s")

    # Assign phase to snapshots
    for snap in queue_snapshots:
        t = snap["time"]
        if t < phase1_duration:
            snap["phase"] = "Normal"
        elif t < phase1_duration + phase2_duration:
            snap["phase"] = "Burst"
        else:
            snap["phase"] = "Cooldown"

    # Save CSV
    csv_path = RESULTS_DIR / "exp1_queue_snapshots.csv"
    snap_fields = (["time"] + gpu_names
                   + [f"{n}_running" for n in gpu_names]
                   + [f"{n}_waiting" for n in gpu_names]
                   + ["phase"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=snap_fields, extrasaction="ignore")
        writer.writeheader()
        for snap in queue_snapshots:
            writer.writerow(snap)
    print(f"  Queue snapshots saved: {csv_path}")

    comp_fields = ["request_id", "gpu_type", "e2e", "phase", "arrival_time",
                   "finish_time", "max_tokens", "status"]
    comp_csv = RESULTS_DIR / "exp1_completions.csv"
    with open(comp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comp_fields)
        writer.writeheader()
        for c in completion_log:
            writer.writerow({k: c.get(k, "") for k in comp_fields})
    print(f"  Completion log saved: {comp_csv}")

    plot_experiment1(queue_snapshots, completion_log,
                     phase1_duration, phase1_duration + phase2_duration,
                     wrr_weights, gpu_names)

    return queue_snapshots, completion_log


def plot_experiment1(snapshots, completion_log, burst_start_time, burst_end_time, wrr_weights, gpu_names=None):
    """Experiment 1: 큐 길이(상단) + E2E Latency(하단) 이중 패널 시계열 그래프"""
    set_plot_style()

    if gpu_names is None:
        gpu_names = ["RTX3090", "RTX4090", "RTX5090"]

    times = [s["time"] for s in snapshots]

    fig, (ax_q, ax_l) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

    # ===== 상단: Queue Length =====
    for gt in gpu_names:
        vals = [s.get(gt, 0) for s in snapshots]
        ax_q.plot(times, vals, label=GPU_DISPLAY_NAMES.get(gt, gt), color=GPU_COLORS.get(gt, "gray"),
                  linewidth=1.8, alpha=0.9)

    for ax in (ax_q, ax_l):
        ax.axvspan(0, burst_start_time, alpha=0.05, color="blue")
        ax.axvspan(burst_start_time, burst_end_time, alpha=0.08, color="red")
        if times:
            ax.axvspan(burst_end_time, max(times), alpha=0.05, color="green")
        ax.axvline(x=burst_start_time, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(x=burst_end_time, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax_q.set_ylabel("Queue Length\n(running + waiting)")
    weight_str = ":".join(str(w) for w in wrr_weights)
    ax_q.set_title(f"Static Weight Scheduling (WRR {weight_str}) under QPS Burst", fontsize=14, fontweight="bold")
    ax_q.legend(loc="upper left", framealpha=0.9, ncol=3)
    ax_q.set_ylim(bottom=0)

    ymax_q = ax_q.get_ylim()[1]
    mid1 = burst_start_time / 2
    mid2 = (burst_start_time + burst_end_time) / 2
    label_y = ymax_q * 0.15
    ax_q.text(mid1, label_y, "Phase 1\n(Normal)",
              ha="center", va="bottom", fontsize=8, fontstyle="italic", color="navy",
              alpha=0.8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax_q.text(mid2, label_y, "Phase 2\n(QPS Burst)",
              ha="center", va="bottom", fontsize=8, fontstyle="italic", color="darkred",
              alpha=0.8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    if times:
        mid3 = (burst_end_time + max(times)) / 2
        ax_q.text(mid3, label_y, "Cooldown\n(No New Requests)",
                  ha="center", va="bottom", fontsize=8, fontstyle="italic", color="darkgreen",
                  alpha=0.8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # ===== 하단: E2E Latency (이동 평균만 표시) =====
    for gt in gpu_names:
        subset = [c for c in completion_log if c["gpu_type"] == gt]
        if not subset:
            continue
        ft = [c["finish_time"] for c in subset]
        lat = [c["e2e"] for c in subset]

        if len(ft) >= 5:
            sorted_pairs = sorted(zip(ft, lat))
            ft_s = [p[0] for p in sorted_pairs]
            lat_s = [p[1] for p in sorted_pairs]
            win = min(20, len(lat_s))
            ma = []
            for i in range(len(lat_s)):
                lo = max(0, i - win + 1)
                ma.append(statistics.mean(lat_s[lo:i + 1]))
            ax_l.plot(ft_s, ma, color=GPU_COLORS.get(gt, "gray"), linewidth=2.0, alpha=0.9,
                      label=GPU_DISPLAY_NAMES.get(gt, gt))

    ax_l.set_xlabel("Time (seconds)")
    ax_l.set_ylabel("E2E Latency (seconds)")
    ax_l.legend(loc="upper left", framealpha=0.9, ncol=3)
    ax_l.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp1_wrr_queue_explosion.pdf"
    fig.savefig(out_path)
    fig.savefig(RESULTS_DIR / "exp1_wrr_queue_explosion.png")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ============================================================
# Experiment 2: SQF Mis-scheduling & Regret Analysis
# ============================================================

class SQFScheduler:
    """Shortest Queue First (waiting 큐 기반 선택)"""

    def select(self, queue_lengths: Dict[str, Dict[str, int]]) -> Tuple[str, Dict]:
        best_server = None
        best_name = None
        min_waiting = float("inf")
        for server in BACKEND_SERVERS:
            ql = queue_lengths.get(server["name"], {})
            w = ql.get("waiting", float("inf"))
            if w < min_waiting or (w == min_waiting and random.random() < 0.5):
                min_waiting = w
                best_server = server
                best_name = server["name"]
        return best_name, best_server


async def send_probe_request(
    client: httpx.AsyncClient,
    server: Dict,
    prompt: str,
    max_tokens: int = 16,
) -> float:
    """프로브 요청: 해당 서버에 요청을 보내고 E2E latency 반환 (non-streaming)"""
    url = f"http://{server['host']}:{server['port']}/v1/chat/completions"
    req = build_chat_request(prompt, max_tokens=max_tokens, stream=False)
    t0 = time.time()
    try:
        resp = await client.post(url, json=req, timeout=120.0)
        return time.time() - t0
    except Exception:
        return float("inf")


async def inject_background_load(
    client: httpx.AsyncClient,
    server: Dict,
    n_requests: int,
    sharegpt_ratio: float,
    completion_events: list,
):
    """백그라운드 부하 주입: n_requests 개의 요청을 fire-and-forget으로 전송"""
    tasks = []
    for i in range(n_requests):
        if random.random() < sharegpt_ratio:
            prompt = random.choice(SHAREGPT_PROMPTS)
            max_tokens = random.randint(1024, 2048)
        else:
            prompt = random.choice(LMSYS_PROMPTS)
            max_tokens = random.randint(512, 1024)

        url = f"http://{server['host']}:{server['port']}/v1/chat/completions"
        req = build_chat_request(prompt, max_tokens=max_tokens, stream=True)

        async def _send(u=url, r=req):
            try:
                async with client.stream("POST", u, json=r) as resp:
                    async for _ in resp.aiter_bytes():
                        pass
            except Exception:
                pass

        t = asyncio.create_task(_send())
        tasks.append(t)
        await asyncio.sleep(0.002)

    completion_events.extend(tasks)
    return tasks


LONG_BASE_PROMPT = (
    "Write an extremely detailed essay about the complete history of computing, "
    "covering all major milestones and key figures in chronological order. "
) * 15


async def _inject_heavy_bg(
    client: httpx.AsyncClient,
    server: Dict,
    n_requests: int,
    weight_ratio: float,
    task_list: list,
):
    """waiting 큐를 형성하기 위한 고부하 배경 주입.
    weight_ratio(0~1)에 따라 모든 요청의 max_tokens를 비례 조절.
    ratio=0: mt=300~400 (light), ratio=1: mt=1200~1536 (heavy)"""
    mt_base_lo, mt_base_hi = 300, 400
    mt_heavy_lo, mt_heavy_hi = 1200, 1536
    lo = int(mt_base_lo + (mt_heavy_lo - mt_base_lo) * weight_ratio)
    hi = int(mt_base_hi + (mt_heavy_hi - mt_base_hi) * weight_ratio)

    for i in range(n_requests):
        max_tokens = random.randint(lo, hi)
        url = f"http://{server['host']}:{server['port']}/v1/chat/completions"
        req = build_chat_request(LONG_BASE_PROMPT, max_tokens=max_tokens, stream=True)

        async def _send(u=url, r=req):
            try:
                async with client.stream("POST", u, json=r) as resp:
                    async for _ in resp.aiter_bytes():
                        pass
            except Exception:
                pass

        t = asyncio.create_task(_send())
        task_list.append(t)
        await asyncio.sleep(0.002)


async def run_experiment2(
    sharegpt_ratios: List[float] = None,
    bg_requests_slow: int = 260,
    bg_requests_fast: int = 263,
    probes_per_ratio: int = 15,
    warmup_wait: float = 2.0,
    probe_max_tokens: int = 16,
):
    """
    Experiment 2: SQF Regret 분석 (비대칭 워크로드, waiting 기반)

    RTX3090(느린 GPU)에 ShareGPT 비율을 0%→100%로 변화시키고,
    RTX5090(빠른 GPU)는 항상 LMSYS만 사용.
    양쪽 모두 waiting 큐가 존재하되 RTX3090이 약간 짧게 유지 → SQF가 RTX3090 선택.
    RTX3090은 느린 하드웨어 + 무거운 워크로드 → 높은 Regret.
    """
    print("=" * 70)
    print("  Experiment 2: SQF Mis-scheduling & Regret Analysis")
    print("  (Asymmetric Workload on Low-perf GPU)")
    print("=" * 70)

    if sharegpt_ratios is None:
        sharegpt_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    sqf = SQFScheduler()
    server_slow = BACKEND_SERVERS[0]   # RTX3090 (Low-perf, varying ShareGPT)
    server_fast = BACKEND_SERVERS[2]   # RTX5090 (High-perf, always LMSYS)
    print(f"  Low-perf  GPU (varying ShareGPT): {server_slow['name']} ({server_slow['gpu_type']})")
    print(f"  High-perf GPU (always LMSYS):     {server_fast['name']} ({server_fast['gpu_type']})")
    print(f"  Background load: slow={bg_requests_slow}, fast={bg_requests_fast}")
    print(f"  Probes per ratio: {probes_per_ratio}")
    print(f"  ShareGPT ratios (Low-perf only): {[f'{r*100:.0f}%' for r in sharegpt_ratios]}")
    print()

    all_results = []

    bg_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=30.0),
        limits=httpx.Limits(max_connections=2000, max_keepalive_connections=500),
    )
    probe_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=30.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )

    async with bg_client, probe_client:

        for ratio_idx, sg_ratio in enumerate(sharegpt_ratios):
            print(f"  [{ratio_idx+1}/{len(sharegpt_ratios)}] Low-perf ShareGPT = {sg_ratio*100:.0f}%  (High-perf = 0%)")
            ratio_results = []

            for probe_idx in range(probes_per_ratio):
                bg_tasks = []

                bg_s = asyncio.create_task(
                    _inject_heavy_bg(bg_client, server_slow, bg_requests_slow, sg_ratio, bg_tasks))
                bg_f = asyncio.create_task(
                    _inject_heavy_bg(bg_client, server_fast, bg_requests_fast, 0.0, bg_tasks))
                await asyncio.gather(bg_s, bg_f)

                await asyncio.sleep(warmup_wait)

                ql = await fetch_queue_lengths(probe_client)
                ql_slow = ql[server_slow["name"]]["waiting"]
                ql_fast = ql[server_fast["name"]]["waiting"]

                sqf_ql = {
                    server_slow["name"]: ql[server_slow["name"]],
                    server_fast["name"]: ql[server_fast["name"]],
                }
                sqf_name, _ = sqf.select(sqf_ql)
                sqf_choice = "slow" if sqf_name == server_slow["name"] else "fast"

                probe_prompt = random.choice(LMSYS_PROMPTS)

                async def _probe(srv, prompt, mt):
                    url = f"http://{srv['host']}:{srv['port']}/v1/chat/completions"
                    req = build_chat_request(prompt, max_tokens=mt, stream=False)
                    t0 = time.time()
                    try:
                        await probe_client.post(url, json=req, timeout=300.0)
                        return time.time() - t0
                    except Exception:
                        return float("inf")

                t_slow_task = asyncio.create_task(_probe(server_slow, probe_prompt, probe_max_tokens))
                t_fast_task = asyncio.create_task(_probe(server_fast, probe_prompt, probe_max_tokens))
                t_slow, t_fast = await asyncio.gather(t_slow_task, t_fast_task)

                t_sqf = t_slow if sqf_choice == "slow" else t_fast
                t_opt = min(t_slow, t_fast)
                regret = t_sqf - t_opt

                result = {
                    "ratio": sg_ratio,
                    "probe_idx": probe_idx,
                    "ql_slow": ql_slow,
                    "ql_fast": ql_fast,
                    "sqf_choice": sqf_choice,
                    "t_slow": t_slow,
                    "t_fast": t_fast,
                    "t_sqf": t_sqf,
                    "t_opt": t_opt,
                    "regret": regret,
                    "is_error": t_sqf != t_opt,
                }
                ratio_results.append(result)
                all_results.append(result)

                sys.stdout.write(f"    probe {probe_idx+1}/{probes_per_ratio}: "
                    f"ql_s={ql_slow} ql_f={ql_fast} sqf={sqf_choice} "
                    f"t_s={t_slow:.1f}s t_f={t_fast:.1f}s reg={regret:.1f}s\n")
                sys.stdout.flush()

                for t in bg_tasks:
                    t.cancel()
                await asyncio.sleep(0.5)
                for drain_sec in range(120):
                    ql = await fetch_queue_lengths(probe_client)
                    total = sum(ql[s["name"]]["total"] for s in [server_slow, server_fast])
                    if total == 0:
                        break
                    await asyncio.sleep(1.0)

            errors = [r for r in ratio_results if r["is_error"]]
            ser = len(errors) / len(ratio_results) * 100 if ratio_results else 0
            avg_regret = statistics.mean(r["regret"] for r in ratio_results) if ratio_results else 0
            max_regret = max(r["regret"] for r in ratio_results) if ratio_results else 0
            avg_ql_slow = statistics.mean(r["ql_slow"] for r in ratio_results)
            avg_ql_fast = statistics.mean(r["ql_fast"] for r in ratio_results)
            print(f"    SER={ser:.1f}%, Avg Regret={avg_regret:.3f}s, Max Regret={max_regret:.3f}s, "
                  f"Avg QL: slow={avg_ql_slow:.1f}, fast={avg_ql_fast:.1f}")

    csv_path = RESULTS_DIR / "exp2_regret_data.csv"
    fieldnames = ["ratio", "probe_idx", "ql_slow", "ql_fast", "sqf_choice",
                   "t_slow", "t_fast", "t_sqf", "t_opt", "regret", "is_error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"  Raw data saved: {csv_path}")

    plot_experiment2(all_results, sharegpt_ratios)

    return all_results


def plot_experiment2(results, sharegpt_ratios):
    """Experiment 2 시각화: 비대칭 워크로드에서의 SQF 오판"""
    set_plot_style()

    ratio_stats = {}
    for ratio in sharegpt_ratios:
        subset = [r for r in results if abs(r["ratio"] - ratio) < 0.001]
        if not subset:
            continue

        errors = [r for r in subset if r["is_error"]]
        ser = len(errors) / len(subset) * 100

        penalties = [r["t_sqf"] / r["t_opt"] if r["t_opt"] > 0 else 1.0 for r in subset]
        avg_penalty = statistics.mean(penalties)
        if len(penalties) > 1:
            se = statistics.stdev(penalties) / math.sqrt(len(penalties))
            ci95 = 1.96 * se
        else:
            ci95 = 0

        regrets = [r["regret"] for r in subset]
        avg_regret = statistics.mean(regrets)
        if len(regrets) > 1:
            se_r = statistics.stdev(regrets) / math.sqrt(len(regrets))
            ci95_r = 1.96 * se_r
        else:
            ci95_r = 0

        max_regret = max(regrets) if regrets else 0

        ratio_stats[ratio] = {
            "ser": ser,
            "avg_penalty": avg_penalty,
            "ci95_penalty": ci95,
            "avg_regret": avg_regret,
            "ci95_regret": ci95_r,
            "max_regret": max_regret,
        }

    x_labels = [f"{int(r * 100)}%" for r in sorted(ratio_stats.keys())]
    x_pos = np.arange(len(x_labels))

    sers = [ratio_stats[r]["ser"] for r in sorted(ratio_stats.keys())]
    penalties = [ratio_stats[r]["avg_penalty"] for r in sorted(ratio_stats.keys())]
    ci_penalties = [ratio_stats[r]["ci95_penalty"] for r in sorted(ratio_stats.keys())]

    # --- Single Plot: Average Regret ---
    avg_regrets = [ratio_stats[r]["avg_regret"] for r in sorted(ratio_stats.keys())]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x_pos, avg_regrets, 0.55, color="#e74c3c", alpha=0.85,
                  edgecolor="#b03030", linewidth=0.8)
    ax.set_xlabel("Heavy Request Ratio", fontweight='bold', fontsize=14)
    ax.set_ylabel("Latency Penalty of SQF\nvs. Optimal (s)", fontweight='bold', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    out_path = RESULTS_DIR / "exp2_sqf_regret.pdf"
    fig.savefig(out_path, dpi=1200, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "exp2_sqf_regret.png", dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Motivation Experiments for Paper")
    parser.add_argument("--exp", type=str, default="all", choices=["1", "1r", "2", "all"],
                        help="Which experiment to run (1=sim, 1r=real GPU, 2=SQF, all=1+2)")

    # Experiment 1 parameters
    parser.add_argument("--exp1-phase1-rps", type=float, default=4.0)
    parser.add_argument("--exp1-phase1-duration", type=float, default=40.0)
    parser.add_argument("--exp1-phase2-rps", type=float, default=10.0)
    parser.add_argument("--exp1-phase2-duration", type=float, default=25.0)
    parser.add_argument("--exp1-cooldown", type=float, default=200.0)
    parser.add_argument("--exp1-weights", type=str, default="1,2,4",
                        help="WRR weights (comma-separated, e.g., '1,2,4')")
    parser.add_argument("--exp1-monitor-interval", type=float, default=0.2)

    # Experiment 1r (real GPU) parameters
    parser.add_argument("--exp1r-phase1-rps", type=float, default=2.0)
    parser.add_argument("--exp1r-phase1-duration", type=float, default=30.0)
    parser.add_argument("--exp1r-phase2-rps", type=float, default=6.0)
    parser.add_argument("--exp1r-phase2-duration", type=float, default=30.0)
    parser.add_argument("--exp1r-cooldown", type=float, default=120.0)
    parser.add_argument("--exp1r-weights", type=str, default="1,2,4")
    parser.add_argument("--exp1r-max-tokens", type=int, default=256)
    parser.add_argument("--exp1r-monitor-interval", type=float, default=0.5)
    parser.add_argument("--exp1r-heavy-prompts", action="store_true",
                        help="Use heavy (ShareGPT) prompts for more KV-cache pressure")

    # Experiment 2 parameters
    parser.add_argument("--exp2-bg-requests", type=int, default=15,
                        help="Background requests per server per ratio")
    parser.add_argument("--exp2-probes", type=int, default=30,
                        help="Number of probe requests per ratio")
    parser.add_argument("--exp2-warmup", type=float, default=3.0,
                        help="Wait time after background load injection")
    parser.add_argument("--exp2-probe-tokens", type=int, default=256,
                        help="Max tokens for probe requests")

    args = parser.parse_args()

    global RESULTS_DIR
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = RESULTS_BASE / run_id
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    latest_link = RESULTS_BASE / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_id)

    print(f"\n{'='*70}")
    print(f"  Motivation Experiments — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results directory: {RESULTS_DIR.absolute()}")
    print(f"  (latest → {run_id})")
    print(f"{'='*70}\n")

    run_info = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "experiment": args.exp,
        "args": vars(args),
    }
    with open(RESULTS_DIR / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    if args.exp in ("1", "all"):
        weights = [int(x) for x in args.exp1_weights.split(",")]
        run_experiment1_sim(
            phase1_rps=args.exp1_phase1_rps,
            phase1_duration=args.exp1_phase1_duration,
            phase2_rps=args.exp1_phase2_rps,
            phase2_duration=args.exp1_phase2_duration,
            cooldown_duration=args.exp1_cooldown,
            wrr_weights=weights,
            snapshot_interval=args.exp1_monitor_interval,
        )
        print()

    if args.exp == "1r":
        weights = [int(x) for x in args.exp1r_weights.split(",")]
        await run_experiment1_real(
            phase1_rps=args.exp1r_phase1_rps,
            phase1_duration=args.exp1r_phase1_duration,
            phase2_rps=args.exp1r_phase2_rps,
            phase2_duration=args.exp1r_phase2_duration,
            cooldown_duration=args.exp1r_cooldown,
            wrr_weights=weights,
            max_tokens=args.exp1r_max_tokens,
            monitor_interval=args.exp1r_monitor_interval,
            use_heavy_prompts=args.exp1r_heavy_prompts,
        )
        print()

    if args.exp in ("2", "all"):
        await run_experiment2(
            bg_requests_per_server=args.exp2_bg_requests,
            probes_per_ratio=args.exp2_probes,
            warmup_wait=args.exp2_warmup,
            probe_max_tokens=args.exp2_probe_tokens,
        )
        print()

    print(f"\n  All experiments complete. Results in: {RESULTS_DIR.absolute()}\n")


if __name__ == "__main__":
    asyncio.run(main())
