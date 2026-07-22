# vLLM Proxy Server

> 한국어 버전은 [README_KOR.md](README_KOR.md)에서 확인하실 수 있습니다.

Proxy server that forwards inference requests to multiple vLLM backends.
Supports both streaming and non-streaming OpenAI Chat Completions traffic
and serves as the scheduling layer for the BYSTANDER research system.

## Features

- OpenAI-compatible Chat Completions API (`/v1/chat/completions`)
- Streaming and non-streaming request forwarding
- Multiple routing algorithms (Round Robin, Weighted Round Robin,
  Shortest Queue First, SLM Adaptive, Fisher-Jenks SQF)
- Per-request metric collection saved as CSV
  - Backend state (running / waiting requests at forwarding time)
  - TTFT (Time To First Token)
  - E2E Latency
- Real-time metric query endpoints
- HTTP/2 support and connection pool reuse
- Graceful shutdown (drains in-flight requests before closing)
- Experiment-scoped metric files with automatic filename generation

## Backend Servers

Each entry in `config.py`'s `BACKEND_SERVERS` list points to one vLLM
instance. Hosts and ports are kept out of the repository — replace the
placeholders with your own values before running:

```python
BACKEND_SERVERS = [
    {"host": "BACKEND_RTX3090_HOST", "port": <PORT>, "name": "RTX3090_SERVER", "gpu_type": "RTX3090"},
    {"host": "BACKEND_RTX5090_HOST", "port": <PORT>, "name": "RTX5090_SERVER", "gpu_type": "RTX5090"},
]
```

## Installation

```bash
pip install -r requirements.txt
```

## Performance Settings

Performance-related knobs live in `config.py`.

### Main limits

| Item | Default | Meaning |
|-----|--------|---------|
| Max concurrent connections | 10,000 | Upper bound on simultaneous client connections to the proxy |
| Max connections per host | 500 | Max simultaneous streaming requests to each backend |
| Throughput (RPS) | ~500-1,000 | Approximate requests per second a single worker can serve |

## Running

```bash
python proxy_server.py

# or run directly with uvicorn
uvicorn proxy_server:app --host 0.0.0.0 --port <PROXY_PORT>
```

## API Usage Examples

In the snippets below, replace `<PROXY_PORT>` with the port you bind the
proxy to.

### 1. Health check

```bash
curl http://localhost:<PROXY_PORT>/health
```

### 2. Chat Completions (streaming)

```bash
curl -X POST http://localhost:<PROXY_PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### 3. Chat Completions (non-streaming)

```bash
curl -X POST http://localhost:<PROXY_PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

### 4. Stats

```bash
curl http://localhost:<PROXY_PORT>/stats
```

### 5. Current metrics snapshot

```bash
curl http://localhost:<PROXY_PORT>/metrics/current
```

### 6. Metric history

```bash
# Last 100 records (default)
curl http://localhost:<PROXY_PORT>/metrics/history

# Last 50 records
curl "http://localhost:<PROXY_PORT>/metrics/history?limit=50"
```

### 7. Finalize experiment (graceful shutdown)

```bash
curl -X POST http://localhost:<PROXY_PORT>/finalize \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "client_1",
    "experiment_name": "my_experiment",
    "total_requests": 100,
    "timeout": 300
  }'
```

Behavior:
1. Stop accepting new requests (returns 503)
2. Wait for all in-flight requests to finish
3. Persist the metric log to a dedicated file
   (`metrics_log_{experiment}_{timestamp}_{client}.csv`)
4. Reset the primary CSV file (keep header only)
5. Reset the request counter (next experiment starts at `request_id = 1`)
6. Return to the normal state for the next experiment

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info and status |
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Chat Completions API (OpenAI-compatible) |
| `/stats` | GET | Proxy statistics (including active request count) |
| `/metrics/current` | GET | Current snapshot of backend metrics |
| `/metrics/history` | GET | CSV-backed metric history |
| `/finalize` | POST | Finish experiment and graceful shutdown |

## Graceful Shutdown

The proxy finishes in-flight requests cleanly at the end of each
experiment.

### Sequence

1. Client calls `/finalize`.
2. The proxy flips a `shutting_down` flag; new requests are rejected
   with HTTP 503.
3. The proxy waits for every active request to complete (up to the
   configured `timeout`).
4. The metric log is saved as
   `metrics_log_<experiment>_<timestamp>_<client>.csv`.
5. The primary `metrics_log.csv` is truncated down to its header row.
6. The request counter is reset so the next experiment starts at
   `request_id = 1`.
7. The proxy returns to its normal state and is ready for the next
   experiment.

### Request tracking

Every request increments and decrements `active_requests`:

```python
active_requests += 1

active_requests -= 1  # guaranteed by a finally block
```

### Example

```python
import httpx
import asyncio

async def run_experiment():
    for i in range(100):
        response = await client.post(
            "http://localhost:<PROXY_PORT>/v1/chat/completions",
            json={"model": "...", "messages": [...]}
        )

    finalize_response = await client.post(
        "http://localhost:<PROXY_PORT>/finalize",
        json={
            "client_id": "experiment_1",
            "experiment_name": "baseline",
            "total_requests": 100,
            "timeout": 300
        }
    )

    print(finalize_response.json())
    # {
    #   "status": "success",
    #   "saved_file": "metrics_log_baseline_20250107_153045_experiment_1.csv",
    #   "record_count": 100,
    #   "completed_gracefully": true,
    #   "csv_reset": true,
    #   "next_request_id_starts_from": 1
    # }
```

## Metric Collection

The proxy collects backend metrics when a request is forwarded and then
records the request's TTFT and E2E latency when it finishes.

### Collected metrics

#### 1. Backend state (at forward time)

Pulled as JSON from each backend's `/metrics?format=json` endpoint:

- `engine_running_requests`: number of currently running requests
- `engine_waiting_requests`: number of waiting requests
- `inflight_prompt_token_lengths`: prompt lengths used by SLM Adaptive and
  Fisher-Jenks SQF

#### 2. Request latency (at completion time)

- TTFT (Time To First Token) — time until the first token chunk arrives
  (streaming), or the full response (non-streaming).
- E2E Latency — total elapsed time for the request.

### CSV layout

Metrics are written to `metrics_log.csv`:

| Column | Description |
|--------|-------------|
| `request_id` | Sequential request id |
| `timestamp` | Metric collection time (ISO 8601) |
| `target_server` | Backend the request was forwarded to |
| `target_server_port` | Backend port |
| `<server>_running` | Running requests at forward time (per backend) |
| `<server>_waiting` | Waiting requests at forward time (per backend) |
| `ttft_seconds` | Time To First Token (4 decimal places) |
| `e2e_latency_seconds` | E2E Latency (4 decimal places) |

Example (ports and other runtime values are masked):

```csv
request_id,timestamp,target_server,target_server_port,RTX3090_SERVER_running,RTX3090_SERVER_waiting,RTX5090_SERVER_running,RTX5090_SERVER_waiting,ttft_seconds,e2e_latency_seconds
1,2024-01-06T10:30:15.123456,RTX3090_SERVER,<PORT>,2,3,1,0,0.3521,2.4567
2,2024-01-06T10:30:16.234567,RTX5090_SERVER,<PORT>,2,3,2,1,0.2834,3.1245
3,2024-01-06T10:30:17.345678,RTX3090_SERVER,<PORT>,3,2,2,1,0.4123,1.9876
```

### Timing summary

- Backend state (running/waiting) is captured immediately before the
  request is forwarded to a backend, giving the load snapshot at the
  routing decision point.
- Latency metrics (TTFT / E2E) are measured during and after the
  forwarded response.

Streaming request (high-level):

```
(inbound request)
    ↓ collect backend metrics (running/waiting)
    ↓ forward to backend
    ↓
    ⏱️ first chunk received → record TTFT
    ↓ stream chunks...
    ↓
    ⏱️ last chunk completed → record E2E Latency
    ↓ write CSV row
```

Non-streaming request (high-level):

```
(inbound request)
    ↓ collect backend metrics (running/waiting)
    ↓ forward to backend
    ↓
    ⏱️ full response received → TTFT = E2E Latency
    ↓ write CSV row
```

## Project Layout

```
proxy/
├── proxy_server.py             # main proxy server
├── config.py                   # configuration (routing, presets, backends)
├── requirements.txt            # Python dependencies
├── README.md                   # this file
├── README_KOR.md               # Korean version
├── .gitignore
└── motivation/                 # motivation experiment scripts
    ├── motivation_experiments.py
    └── proxy_server_motivation.py
```

## Logging

The server logs:

- Request forwarding information (target server, endpoint)
- Errors and exceptions
- Startup / shutdown events

## Troubleshooting

### Backend connection failure

- Make sure the backend servers are running.
- Check firewall rules.
- Verify network connectivity.

### Timeout

- Default request timeout is 300 seconds.
- Adjust `httpx.AsyncClient(timeout=300.0)` in the code if you need a
  different value.

## License

MIT
