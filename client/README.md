# BYSTANDER Experiment Client

`proxy_request_qps.py` reproduces the proxy workload used by the BYSTANDER
experiments. It preserves the existing Poisson-style QPS scheduling, dynamic
QPS ranges, request payload, streaming TTFT/E2E measurement, final 5% burst,
and CSV/XLSX result schemas.

## Setup

```bash
cd client
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset selection

Both ShareGPT and LMSYS JSON arrays or JSONL files are accepted:

```bash
# Explicit paths
python proxy_request_qps.py --dataset /data/sharegpt.json --dry-run
python proxy_request_qps.py --dataset /data/lmsys.jsonl --dry-run

# Short aliases
export BYSTANDER_DATASET_DIR=/data/bystander
python proxy_request_qps.py --dataset sharegpt --dry-run
python proxy_request_qps.py --dataset lmsys --dry-run
```

The aliases resolve to `sharegpt_shuffled.json` and
`lmsys_english_shuffled.json`. Search order is the explicit path,
`--dataset-dir`, `BYSTANDER_DATASET_DIR`, the current directory, and this
client directory. The legacy `--sharegpt` option remains an alias for
`--dataset`, so existing experiment scripts continue to work.

Large JSON arrays are streamed with `ijson`; JSONL is always processed one
record at a time. Record order and `--start-index` semantics are unchanged.

## Run one experiment

Validate the configuration without contacting the proxy:

```bash
python proxy_request_qps.py \
  --dataset lmsys \
  --qps 50 --total 4000 \
  --algorithm 5 \
  --dry-run
```

Run the experiment:

```bash
python proxy_request_qps.py \
  --proxy-host "$PROXY_HOST" --proxy-port "${PROXY_PORT:-8012}" \
  --dataset lmsys \
  --qps 50 --total 4000 \
  --algorithm 5 \
  --output results/lmsys_fj.xlsx
```

To send the same scheduled streaming workload directly to an OpenAI-compatible
HTTPS endpoint, provide its base URL. Direct endpoint mode does not call the
custom proxy's `/finalize` API:

```bash
python proxy_request_qps.py \
  --base-url https://qwen3-4b.proxy.ainexus.ktcloud.com/ \
  --dataset /data/sharegpt.json \
  --model Qwen3-4B \
  --qps 4 --max-concurrent 8 --total 20 --max-tokens 256 \
  --output results/qwen3-4b.csv
```

Collect each vLLM replica's Prometheus metrics through a load-balanced
endpoint while the workload runs:

```bash
python collect_vllm_metrics.py \
  --url https://qwen3-4b.proxy.ainexus.ktcloud.com/metrics \
  --interval 0.25 --duration 180 \
  --output results/qwen3-4b-metrics.csv
```

The collector identifies replicas by their stable
`process_start_time_seconds` value and records KV-cache usage, running and
waiting request counts, token counters, successful requests, and preemptions.

For a dynamic QPS range, use `--qps 45-75`. Add `--seed 42` only when a
reproducible sequence of dynamic QPS values is desired; omitting it preserves
the original non-deterministic behavior.

Useful environment variables:

- `PROXY_HOST`, `PROXY_PORT`
- `BYSTANDER_BASE_URL`
- `BYSTANDER_DATASET` (`sharegpt`, `lmsys`, or a path)
- `BYSTANDER_DATASET_DIR`
- `BYSTANDER_MODEL`
- `BYSTANDER_OUTPUT`

## Start an experiment with clean backend state

The client talks only to the proxy, so reset custom vLLM3 state directly on
each backend before a clean experiment run:

```bash
curl -X POST "http://<BACKEND_HOST>:<BACKEND_PORT>/reset_custom_metrics"
```

Call the endpoint only after previous inference requests have finished. It
clears BYSTANDER's custom in-memory request data without restarting the
container. It does not replace the proxy's `/finalize`, which saves and resets
the proxy-side experiment log.

## Repeated paper experiments

`run_repeated_experiments.sh` retains the original experiment matrices,
algorithm settings, QPS values, and restart flow. Set its environment values
and dataset paths, then run it from this directory:

```bash
PROXY_HOST=proxy.example \
PROXY_PORT=8012 \
bash run_repeated_experiments.sh
```

## LMSYS preprocessing

The raw Hugging Face dataset can be converted to the expected shuffled JSON
format with:

```bash
pip install datasets
python preprocess_lmsys.py \
  --input /data/lmsys-chat-1m/processed \
  --output /data/bystander/lmsys_english_shuffled.json
```
