# BYSTANDER: Heterogeneous LLM Inference Scheduling

This repository contains the research source code for **BYSTANDER**, a system
that performs predictive scheduling (routing) of LLM inference workloads on
heterogeneous GPU clusters.

The key idea: given the hardware heterogeneity of each GPU node, the
characteristics of the incoming request (prompt, max_tokens, etc.), and the
real-time load signals of each node (running / waiting request counts,
in-flight prompt-token distributions, etc.), a pre-fine-tuned
**Small Language Model (SLM)** predicts the **end-to-end latency** that would
be observed on every GPU type, and the proxy routes each request to the
backend with the best predicted latency.

A Korean version of this document is available at
[`README_KOR.md`](./README_KOR.md).

## System Overview

```
┌────────────────────┐       ┌──────────────────────────────┐       ┌───────────────────────────┐
│                    │       │  Proxy Server                │       │  Backend Pool (vLLM)      │
│  Client            │──────▶│  - OpenAI-compatible API     │──────▶│  RTX3090 × N              │
│  (load generator)  │  HTTP │  - Routing algorithms        │       │  RTX4090 × M              │
│                    │       │  - SLM-based latency predict │       │  RTX5090 × K              │
│  proxy_request_qps │       │  - Metrics collection        │       │  (customized vLLM image)  │
└────────────────────┘       └──────────────────────────────┘       └───────────────────────────┘
          │                             │                                       │
          │  QPS / algorithm / preset   │  collects running, waiting,           │  exposes /metrics
          │  dataset, total_requests    │  inflight tokens per backend          │  (Prometheus + JSON)
          └────────────────────────────▶                                        │
                                        ◀───── e2e latency, ttft ─────────────  │
```

- **Client node** — the load generator. It streams real LLM requests
  (ShareGPT / LMSYS datasets) to the proxy server following a Poisson arrival
  process and collects the results.
- **Proxy server node** — an OpenAI-compatible chat-completions router.
  It supports several routing algorithms (Round Robin, Weighted Round Robin,
  Shortest Queue First, SLM Adaptive, Fisher-Jenks SQF) and runs SLM
  inference to estimate the expected e2e latency per GPU type before
  dispatching each request.
- **Backend nodes** — a customized vLLM image running inside containers on
  the GPU nodes. Standard Prometheus scrapes use `/metrics`, while the proxy
  uses `/metrics?format=json` for running, waiting, and in-flight prompt-token
  length data in one request.
  (The container image and SLM model weights are distributed separately and
  are **not** included in this repository.)

## Repository Layout

```
.
├── client/                                   # Load generator (runs on the client node)
│   ├── proxy_request_qps.py                  # Main QPS-controlled streaming client
│   ├── run_repeated_experiments.sh           # Repeated-experiment driver (with K8s / VastAI restart)
│   ├── run_single_experiment_with_restart.sh # Single-experiment runner
│   ├── preprocess_lmsys.py                   # LMSYS-chat-1m preprocessing (English filter + shuffle)
│   └── requirements.txt
│
├── proxy/                                    # Proxy server (runs on the node that hosts the SLM)
│   ├── proxy_server.py                       # Main router (SLM loading + every routing algorithm)
│   ├── config.py                             # Backend list · routing options · experiment presets
│   ├── requirements.txt
│   ├── README.md                             # Detailed proxy-server API document
│   ├── .gitignore
│   └── motivation/                           # Motivation experiments from the paper (WRR/SQF limits)
│       ├── proxy_server_motivation.py        # Proxy variant used for the motivation experiments
│       └── motivation_experiments.py         # Motivation experiment runner + plotting
│
├── README.md                                 # This document
└── README_KOR.md                             # Korean version
```

## Routing Algorithms

Select the algorithm either statically through `ROUTING_CONFIG["algorithm"]`
in `proxy/config.py`, or dynamically by calling the proxy's `/set_algorithm`
endpoint from the client.

| ID | Name                   | Description                                                                       |
|----|------------------------|-----------------------------------------------------------------------------------|
| 1  | `round_robin`          | Simple round-robin across all backends.                                           |
| 2  | `weighted_round_robin` | Weighted RR using either GPU performance ratios or calibrated weights.            |
| 3  | `shortest_queue_first` | Picks the backend with the minimum waiting (or total) request count.              |
| 4  | `slm_adaptive`         | Picks the backend with the minimum SLM-predicted e2e latency.                     |
| 5  | `fisher_jenks_sqf`     | Clusters the SLM-predicted latency diffs (Fisher-Jenks / top-N / percent-of-min / EWMA / fixed-threshold) to select a *candidate GPU group*, then applies SQF within that group to pick the final backend. **This is the method proposed by the paper.** |

## Configuration — Values You Must Fill In Before Running

For security reasons the repository does **not** contain real IP addresses,
API keys, or instance IDs. Replace the placeholders below with your own
values (or inject them via environment variables) before launching anything.

### `proxy/config.py`

```python
BACKEND_SERVERS = [
    {"host": "BACKEND_RTX3090_A_HOST", "port": <PORT>, "name": "RTX3090_A", "gpu_type": "RTX3090"},
    {"host": "BACKEND_RTX4090_A_HOST", "port": <PORT>, "name": "RTX4090_A", "gpu_type": "RTX4090"},
    {"host": "BACKEND_RTX5090_A_HOST", "port": <PORT>, "name": "RTX5090_A", "gpu_type": "RTX5090"},
    # ... additional backends
]

ROUTING_CONFIG = {
    "algorithm": "fisher_jenks_sqf",
    "slm_model_path": "model/final_regression_multi_gpu",  # SLM checkpoint (distributed separately)
    # ...
}
```

### `client/run_repeated_experiments.sh`

You can either edit the top of the file or export the variables before
running the script:

```bash
export PROXY_HOST="<proxy server IP>"
export PROXY_PORT=<PROXY_PORT>
export K8S_MASTER="<k8s master node IP>"
export K8S_SSH_PORT=<K8S_SSH_PORT>
export K8S_SSH_USER="<ssh user>"

# Only if you use VastAI-based instance restarts
export VASTAI_API_KEY="<your-api-key>"
export VASTAI_INSTANCE_IDS="INSTANCE_ID_1 INSTANCE_ID_2 ..."   # space-separated

./run_repeated_experiments.sh
```

## How to Run

### 1. Bring Up the Backends

On each GPU node, start a container based on the customized vLLM image
(not included in this repository). Every backend must expose the following
endpoints:

- `POST /v1/chat/completions` — OpenAI-compatible (streaming and
  non-streaming).
- `GET /metrics` — standard Prometheus format.
- `GET /metrics?format=json` — JSON containing
  `engine_running_requests`, `engine_waiting_requests`, and
  `inflight_prompt_token_lengths`. Required by SLM Adaptive and
  Fisher-Jenks SQF.
- `POST /reset_custom_metrics` — clears BYSTANDER's custom in-memory request
  state between experiments. Call only when no inference requests are active.

### 2. Start the Proxy Server

```bash
cd proxy
python -m venv .venv && source .venv/bin/activate   # or: uv venv --seed
pip install -r requirements.txt
# Place the SLM model directory under model/ (distributed separately)
python proxy_server.py
```

The proxy listens on `0.0.0.0:<PROXY_PORT>` by default. Change this in
`UVICORN_CONFIG` inside `config.py` if needed.

### 3. Launch the Client

```bash
cd client
pip install -r requirements.txt

# Prepare datasets
python preprocess_lmsys.py                       # LMSYS (English filter + shuffle)
# sharegpt_shuffled.json must be prepared separately (not included)

# Full repeated-experiment loop (with K8s DaemonSet + VastAI instance restarts)
./run_repeated_experiments.sh
```

Or run a single experiment directly:

```bash
python proxy_request_qps.py \
    --proxy-host "$PROXY_HOST" --proxy-port "$PROXY_PORT" \
    --qps 50 --total 4000 \
    --algorithm 5 \
    --dataset ./lmsys_english_shuffled.json \
    --output results/run.xlsx
```

`--algorithm` accepts the IDs (1..5) from the table above. `--dataset` also
accepts the `sharegpt` and `lmsys` aliases when `BYSTANDER_DATASET_DIR` points
to the directory containing the prepared files. Use `--dry-run` to validate a
dataset and command without contacting the proxy. See `client/README.md` for
the complete client workflow. The legacy `--sharegpt` flag remains supported.

## Collected Metrics

For each request, the proxy aggregates the following values during streaming
and writes them to a per-experiment CSV/XLSX file:

- `request_id`, `timestamp`, `target_server`, `target_server_port`
- Snapshot of every backend's running / waiting counts at forward time
- **TTFT (Time To First Token)** — time until the first token arrives.
- **E2E latency** — total elapsed time for the request.
- (When an SLM-based algorithm is active) the per-GPU-type predicted
  latency and the final scheduling decision.

`proxy/README.md` documents every endpoint, the CSV schema, and the exact
collection timing.

## Not Included in This Repository

The following assets are distributed through separate channels:

- The customized vLLM backend container image
  (e.g. `vllm-openai-myimage*.tar`).
- The fine-tuned SLM weights (to be placed under `proxy/model/...`).
- Large shuffled datasets derived from ShareGPT / LMSYS.
- Raw experiment results (CSV / XLSX) and the analysis / visualization
  scripts used to produce the figures in the paper.
- Non-workload analysis code; only the motivation experiments from the
  paper are kept here.

## License

Released for academic research. Final license terms will be decided together
with the authors.
