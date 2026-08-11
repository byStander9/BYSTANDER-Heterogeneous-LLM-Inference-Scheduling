# BYSTANDER: State-Aware End-to-End Latency Prediction for Heterogeneous LLM Inference Scheduling

> 본 연구는 **IEEE CLOUD 2026**(2026 IEEE International Conference on
> Cloud Computing)에 발표된 논문입니다.

이 레포지토리는 이기종 GPU 클러스터 환경에서 LLM 추론 워크로드를 예측적으로
스케줄링(라우팅)하는 **BYSTANDER** 시스템의 연구용 소스 코드입니다.

각 GPU 노드의 하드웨어 성능 차이, 요청 특성(prompt/max_tokens 등), 노드의
실시간 부하 정보(running / waiting 요청 수, inflight prompt tokens 분포 등)를
사전에 fine-tuning된 **Small Language Model (SLM)** 의 입력으로 주어
각 GPU 노드에서의 **end-to-end latency** 를 예측하고, 이를 기반으로
요청을 어느 백엔드에 라우팅할지 결정합니다.

영문판은 [`README.md`](./README.md) 를 참고하세요.

## 시스템 구성

```
┌────────────────────┐       ┌──────────────────────────────┐       ┌───────────────────────────┐
│                    │       │  Proxy Server                │       │  Backend Pool (vLLM)      │
│  Client            │──────▶│  - OpenAI-compatible API     │──────▶│  RTX3090 × N              │
│  (load generator)  │  HTTP │  - Routing algorithms        │       │  RTX4090 × M              │
│                    │       │  - SLM-based latency predict │       │  RTX5090 × K              │
│  proxy_request_qps │       │  - Metrics collection        │       │  (customized vLLM3 image) │
└────────────────────┘       └──────────────────────────────┘       └───────────────────────────┘
          │                             │                                       │
          │  QPS / algorithm / preset   │  collects running, waiting,           │  exposes /metrics
          │  dataset, total_requests    │  inflight tokens per backend          │  (Prometheus + JSON)
          └────────────────────────────▶                                        │
                                        ◀───── e2e latency, ttft ─────────────  │
```

- **Client node** — 부하 생성기. ShareGPT / LMSYS 데이터셋으로 실제 LLM 요청을
  Poisson 프로세스에 따라 프록시 서버에 스트리밍으로 전송하고 결과를 수집합니다.
- **Proxy server node** — OpenAI 호환 chat completions API를 제공하는 라우터.
  Round Robin, Weighted Round Robin, Shortest Queue First, SLM Adaptive,
  Fisher-Jenks SQF 등 다양한 라우팅 알고리즘을 선택 가능하며, SLM 추론을 통해
  각 GPU 종류별 예상 e2e latency를 산출해 최적 백엔드를 결정합니다.
- **Backend nodes** — 커스터마이징된 vLLM3 이미지가 컨테이너로 동작.
  Prometheus는 표준 `/metrics`를 사용하고, 프록시는
  `/metrics?format=json`을 한 번 호출하여 running, waiting 및 inflight
  prompt token 길이 정보를 함께 수집합니다.
  (컨테이너 이미지와 SLM 모델 파일은 본 레포 외부에서 별도 제공)

## 디렉토리 구조

```
.
├── .gitattributes
├── .gitignore
├── client/                                   # 부하 생성기 (클라이언트 노드에서 실행)
│   ├── preprocess_lmsys.py                   # LMSYS-chat-1m 데이터셋 전처리 (영어 필터 + shuffle)
│   ├── proxy_request_qps.py                  # QPS 제어 스트리밍 요청 발송 메인 스크립트
│   ├── README.md                             # Client 상세 사용법
│   ├── requirements.txt
│   ├── run_repeated_experiments.sh           # 알고리즘·QPS 반복 및 K8s/VastAI 재시작
│   └── tests/
│       ├── test_client_cli.py                # CLI·dry-run·결과 저장 테스트
│       └── test_dataset_loading.py           # ShareGPT/LMSYS 로딩 테스트
│
├── proxy/                                    # 프록시 서버 (SLM이 동작하는 노드에서 실행)
│   ├── .gitignore
│   ├── config.py                             # 백엔드 목록 · 라우팅 설정 · 실험 프리셋
│   ├── proxy_server.py                       # 메인 라우터 (SLM 로딩 + 모든 라우팅 알고리즘)
│   ├── README.md                             # 프록시 서버 API 상세 문서
│   ├── README_KOR.md                         # 프록시 서버 한글 문서
│   ├── requirements.txt
│   └── tests/
│       └── test_metrics_collection.py        # vLLM3 메트릭 회귀 테스트
│
├── README.md                                 # 영문판 (GitHub 기본 노출)
└── README_KOR.md                             # 본 문서 (한글판)
```

메인 `proxy/proxy_server.py`는 아래 설명된 vLLM3 통합 메트릭 API를
사용합니다. 별도의 motivation 실험용 프록시 변형은 포함하지 않습니다.

## 라우팅 알고리즘

`proxy/config.py` 의 `ROUTING_CONFIG["algorithm"]` 으로 선택하거나, 클라이언트가
`/set_algorithm` API로 런타임에 변경합니다.

| ID | 이름                 | 설명                                                                                |
|----|----------------------|-------------------------------------------------------------------------------------|
| 1  | `round_robin`        | 단순 순회                                                                           |
| 2  | `weighted_round_robin` | GPU 성능비 또는 캘리브레이션된 가중치 기반                                       |
| 3  | `shortest_queue_first` | 각 백엔드의 waiting(또는 total) 요청 수 최소 선택                                |
| 4  | `slm_adaptive`       | SLM이 예측한 GPU별 e2e latency 최소 선택                                             |
| 5  | `fisher_jenks_sqf`   | SLM 예측 latency diff 분포를 Fisher-Jenks / top-N / percent-of-min 등으로 clustering하여 후보 GPU 그룹을 선별한 뒤 SQF로 최종 서버 선택 (논문의 제안 기법) |

## 설정 — 배포 전 반드시 수정할 값

보안상의 이유로 저장소에는 실제 IP, API 키, 인스턴스 ID 등이 제거되어 있습니다.
실험 환경에 맞춰 아래 자리표시자를 채워야 동작합니다.

### `proxy/config.py`

```python
BACKEND_SERVERS = [
    {"host": "BACKEND_RTX3090_A_HOST", "port": <PORT>, "name": "RTX3090_A", "gpu_type": "RTX3090"},
    {"host": "BACKEND_RTX4090_A_HOST", "port": <PORT>, "name": "RTX4090_A", "gpu_type": "RTX4090"},
    {"host": "BACKEND_RTX5090_A_HOST", "port": <PORT>, "name": "RTX5090_A", "gpu_type": "RTX5090"},
    # ... 추가 백엔드
]

ROUTING_CONFIG = {
    "algorithm": "fisher_jenks_sqf",
    "slm_model_path": "model/final_regression_multi_gpu",  # SLM 모델 디렉토리 (별도 제공)
    # ...
}
```

### `client/run_repeated_experiments.sh`

환경변수로 주입하거나 파일 상단을 직접 편집합니다.

```bash
export PROXY_HOST="<proxy 서버 IP>"
export PROXY_PORT=<PROXY_PORT>
export K8S_MASTER="<k8s 마스터 노드 IP>"
export K8S_SSH_PORT=<K8S_SSH_PORT>
export K8S_SSH_USER="<ssh 계정>"

# Vast.ai 인스턴스 재시작을 사용하는 경우
export VASTAI_API_KEY="<your-api-key>"
export VASTAI_INSTANCE_IDS="INSTANCE_ID_1 INSTANCE_ID_2 ..."   # 공백 구분

./run_repeated_experiments.sh
```

## 실행 방법

### 1. Backend 준비

본 레포에 포함되지 않은 커스터마이징 vLLM3 이미지로 각 GPU 노드에서 컨테이너를
기동합니다. 각 백엔드는 다음 엔드포인트를 노출해야 합니다.

- `POST /v1/chat/completions` — OpenAI 호환 (streaming / non-streaming)
- `GET /metrics` — 표준 Prometheus 형식
- `GET /metrics?format=json` — `engine_running_requests`,
  `engine_waiting_requests`, `inflight_prompt_token_lengths`를 포함하는 JSON
  (SLM Adaptive / Fisher-Jenks SQF 사용 시 필요)
- `POST /reset_custom_metrics` — 실험 사이에 BYSTANDER가 추가한 메모리 내
  요청 상태를 초기화. 진행 중인 추론 요청이 없을 때만 호출

쿼리 파라미터가 없는 `/metrics`는 계속 Prometheus 텍스트를 반환하므로
기존 Prometheus 수집에는 영향이 없습니다. JSON 메트릭 수집이 한 번
실패하면 프록시는 해당 백엔드의 running/waiting 값을 수집 실패로 표시하고,
마지막으로 정상 수집한 inflight token 목록은 유지한 채 다음 주기에 계속
수집합니다.

### 2. Proxy server 기동

```bash
cd proxy
python -m venv .venv && source .venv/bin/activate    # or: uv venv --seed
pip install -r requirements.txt
# SLM 모델 디렉토리를 model/ 하위에 배치 (별도 제공)
python proxy_server.py
```

기본 listen 주소는 `0.0.0.0:<PROXY_PORT>` 이며 `config.py` 의 `UVICORN_CONFIG` 에서 변경합니다.

### 3. Client에서 실험 실행

```bash
cd client
pip install -r requirements.txt

# 데이터셋 준비
pip install datasets                                  # LMSYS 원본 전처리 시에만 필요
python preprocess_lmsys.py                            # LMSYS (영어 필터 + shuffle)
# sharegpt_shuffled.json 은 별도로 준비 (리포에 미포함)

# 반복 실험 자동 실행 (K8s 데몬셋 · Vast.ai 인스턴스 재시작 루프 포함)
./run_repeated_experiments.sh
```

또는 단발 실행:

```bash
python proxy_request_qps.py \
    --proxy-host "$PROXY_HOST" --proxy-port "$PROXY_PORT" \
    --qps 50 --total 4000 \
    --algorithm 5 \
    --dataset ./lmsys_english_shuffled.json \
    --output results/run.xlsx
```

`--algorithm` 은 위 표의 ID(1~5)를 사용합니다. `BYSTANDER_DATASET_DIR`에
전처리된 파일 디렉터리를 지정하면 `--dataset sharegpt` 또는
`--dataset lmsys` 별칭도 사용할 수 있습니다. `--dry-run`은 프록시에
요청하지 않고 데이터셋과 옵션만 검증합니다. 전체 사용법은
`client/README.md`를 참고하세요. 기존 `--sharegpt` 옵션도 계속 지원합니다.

## 수집되는 메트릭

각 요청의 스트리밍 처리 과정에서 프록시가 아래 값을 집계해 실험별 CSV / XLSX로
저장합니다.

- request_id, timestamp, target_server, target_server_port
- 모든 백엔드의 running / waiting 요청 수 (forward 시점 스냅샷)
- **TTFT (Time To First Token)** — 첫 번째 토큰이 도착하기까지의 시간
- **E2E latency** — 해당 요청의 총 소요 시간
- (SLM 알고리즘 사용 시) SLM 예측 latency per GPU 종류 및 최종 결정

`proxy/README.md` 에 엔드포인트별 수집 타이밍과 CSV 스키마가 상세히 기술되어
있습니다.

## 제외된 파일

본 레포에는 아래 자산이 **포함되지 않습니다**. 별도 채널로 제공됩니다.

- 백엔드 vLLM 커스텀 컨테이너 이미지 (`vllm-openai-myimage*.tar`)
- Fine-tuning 된 SLM 가중치 (`proxy/model/*`)
- ShareGPT / LMSYS 원본 및 셔플된 대용량 JSON 데이터셋
- 실험 결과 원본 CSV/XLSX 및 결과 분석 스크립트
- Motivation 실험 및 기타 분석·시각화 코드

## 라이선스

학술 연구용으로 배포됩니다. 세부 라이선스는 저자와 협의 후 결정.
