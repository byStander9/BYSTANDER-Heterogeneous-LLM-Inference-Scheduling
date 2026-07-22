# vLLM 프록시 서버

> For the English version, see [README.md](README.md).

여러 대의 vLLM 백엔드로 추론 요청을 전달하는 프록시 서버입니다. OpenAI
Chat Completions 형식의 스트리밍 및 비스트리밍 트래픽을 모두 지원하며,
BYSTANDER 연구 시스템의 스케줄링 계층 역할을 합니다.

## 기능

- OpenAI 호환 Chat Completions API (`/v1/chat/completions`)
- 스트리밍 및 비스트리밍 요청 포워딩
- 다양한 라우팅 알고리즘 지원 (Round Robin, Weighted Round Robin,
  Shortest Queue First, SLM Adaptive, Fisher-Jenks SQF)
- 요청별 메트릭을 CSV로 저장
  - 포워딩 시점의 백엔드 상태 (running / waiting)
  - TTFT (Time To First Token)
  - E2E Latency
- 실시간 메트릭 조회 API
- HTTP/2 지원 및 연결 풀 재사용
- Graceful shutdown (진행 중인 요청 완료 후 종료)
- 실험 단위 메트릭 파일 자동 생성

## 백엔드 서버

`config.py`의 `BACKEND_SERVERS` 리스트에 각 vLLM 인스턴스를 등록합니다.
호스트와 포트는 저장소에 올리지 않고, 실행 전 사용자 환경에 맞는 값으로
치환해서 사용하시면 됩니다.

```python
BACKEND_SERVERS = [
    {"host": "BACKEND_RTX3090_HOST", "port": <PORT>, "name": "RTX3090_SERVER", "gpu_type": "RTX3090"},
    {"host": "BACKEND_RTX5090_HOST", "port": <PORT>, "name": "RTX5090_SERVER", "gpu_type": "RTX5090"},
]
```

## 설치

```bash
pip install -r requirements.txt
```

## 성능 설정

성능 관련 설정은 `config.py`에서 조정합니다.

### 주요 제한값

| 항목 | 기본값 | 의미 |
|-----|--------|------|
| 최대 동시 연결 | 10,000 | 프록시가 받을 수 있는 최대 클라이언트 연결 수 |
| 호스트당 최대 연결 | 500 | 각 백엔드 서버당 최대 동시 스트리밍 요청 수 |
| 처리량 (RPS) | ~500-1,000 | 단일 워커 기준 초당 처리 가능한 요청 수 |

## 실행

```bash
python proxy_server.py

# 또는 uvicorn으로 직접 실행
uvicorn proxy_server:app --host 0.0.0.0 --port <PROXY_PORT>
```

## API 사용 예제

아래 예시의 `<PROXY_PORT>`에는 프록시 서버가 바인딩한 포트를 입력해
사용하시면 됩니다.

### 1. 헬스 체크

```bash
curl http://localhost:<PROXY_PORT>/health
```

### 2. Chat Completions (스트리밍)

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

### 3. Chat Completions (비스트리밍)

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

### 4. 통계 확인

```bash
curl http://localhost:<PROXY_PORT>/stats
```

### 5. 현재 메트릭 조회

```bash
curl http://localhost:<PROXY_PORT>/metrics/current
```

### 6. 메트릭 히스토리 조회

```bash
# 최근 100개 (기본값)
curl http://localhost:<PROXY_PORT>/metrics/history

# 최근 50개
curl "http://localhost:<PROXY_PORT>/metrics/history?limit=50"
```

### 7. 실험 완료 (Graceful Shutdown)

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

동작:
1. 새로운 요청 수락 중단 (503 반환)
2. 진행 중인 요청이 모두 완료될 때까지 대기
3. 메트릭 로그를 실험 단위 파일로 저장
   (`metrics_log_{experiment}_{timestamp}_{client}.csv`)
4. 기본 CSV 파일 초기화 (헤더만 유지)
5. 요청 카운터 초기화 (다음 실험은 `request_id = 1`부터)
6. 정상 상태로 복구하여 다음 실험을 받을 준비 완료

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 서버 정보 및 상태 |
| `/health` | GET | 헬스 체크 |
| `/v1/chat/completions` | POST | Chat Completions API (OpenAI 호환) |
| `/stats` | GET | 프록시 통계 (활성 요청 수 포함) |
| `/metrics/current` | GET | 모든 백엔드의 현재 메트릭 |
| `/metrics/history` | GET | CSV 기반 메트릭 히스토리 |
| `/finalize` | POST | 실험 완료 및 graceful shutdown |

## Graceful Shutdown 기능

실험 종료 시 진행 중인 요청을 안전하게 소화하는 graceful shutdown을
제공합니다.

### 동작 순서

1. 클라이언트가 `/finalize` 호출
2. `shutting_down` 플래그 설정 → 신규 요청은 HTTP 503으로 거부
3. 활성 요청이 모두 끝날 때까지 대기 (`timeout` 한도 내)
4. 메트릭 로그를
   `metrics_log_<experiment>_<timestamp>_<client>.csv`로 저장
5. 기본 `metrics_log.csv`는 헤더만 남기고 초기화
6. 요청 카운터 리셋 (다음 실험은 `request_id = 1`부터)
7. 정상 상태로 복구하여 다음 실험을 받을 준비 완료

### 요청 추적

모든 요청은 `active_requests`를 증감시켜 수명을 추적합니다.

```python
active_requests += 1

active_requests -= 1  # finally 블록에서 보장
```

### 사용 예시

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

## 메트릭 수집 기능

프록시는 요청을 백엔드로 포워딩하기 직전에 각 백엔드의 상태를 수집하고,
요청 완료 시점에 해당 요청의 TTFT와 E2E Latency를 기록합니다.

### 수집되는 메트릭

#### 1. 백엔드 서버 상태 (포워딩 시점)

각 백엔드 `/metrics?format=json` 엔드포인트에서 JSON으로 수집:
- `engine_running_requests`: 현재 실행 중인 요청 수
- `engine_waiting_requests`: 대기 중인 요청 수
- `inflight_prompt_token_lengths`: SLM Adaptive와 Fisher-Jenks SQF가
  사용하는 프롬프트 길이 목록

#### 2. 요청 레이턴시 (완료 시점)

- TTFT (Time To First Token): 첫 번째 토큰이 도착하기까지의 시간
  (스트리밍 기준, 비스트리밍은 전체 응답 수신 시점)
- E2E Latency: 해당 요청의 총 소요 시간

### CSV 파일 구조

메트릭은 `metrics_log.csv` 파일에 저장됩니다.

| 컬럼 | 설명 |
|------|------|
| `request_id` | 요청 고유 번호 |
| `timestamp` | 메트릭 수집 시각 (ISO 8601) |
| `target_server` | 요청을 전달한 대상 서버 이름 |
| `target_server_port` | 대상 서버 포트 |
| `<server>_running` | 해당 백엔드의 실행 중인 요청 수 (포워딩 시점) |
| `<server>_waiting` | 해당 백엔드의 대기 중인 요청 수 (포워딩 시점) |
| `ttft_seconds` | TTFT (초, 소수점 4자리) |
| `e2e_latency_seconds` | E2E Latency (초, 소수점 4자리) |

예시 (포트 및 런타임 값은 마스킹):

```csv
request_id,timestamp,target_server,target_server_port,RTX3090_SERVER_running,RTX3090_SERVER_waiting,RTX5090_SERVER_running,RTX5090_SERVER_waiting,ttft_seconds,e2e_latency_seconds
1,2024-01-06T10:30:15.123456,RTX3090_SERVER,<PORT>,2,3,1,0,0.3521,2.4567
2,2024-01-06T10:30:16.234567,RTX5090_SERVER,<PORT>,2,3,2,1,0.2834,3.1245
3,2024-01-06T10:30:17.345678,RTX3090_SERVER,<PORT>,3,2,2,1,0.4123,1.9876
```

### 메트릭 수집 타이밍

- 백엔드 상태 메트릭(running/waiting)은 요청을 백엔드로 포워딩하기
  직전에 수집되어, 라우팅 결정 시점의 부하 스냅샷을 제공합니다.
- 레이턴시 메트릭(TTFT / E2E)은 포워딩된 응답을 처리하는 동안과 완료
  시점에 측정됩니다.

스트리밍 요청(개략):

```
(요청 수신)
    ↓ 백엔드 메트릭 수집 (running/waiting)
    ↓ 백엔드로 전달
    ↓
    ⏱️ 첫 chunk 수신 → TTFT 기록
    ↓ chunk 스트리밍...
    ↓
    ⏱️ 마지막 chunk 완료 → E2E Latency 기록
    ↓ CSV 기록
```

비스트리밍 요청(개략):

```
(요청 수신)
    ↓ 백엔드 메트릭 수집 (running/waiting)
    ↓ 백엔드로 전달
    ↓
    ⏱️ 전체 응답 수신 → TTFT = E2E Latency
    ↓ CSV 기록
```

## 구조

```
proxy/
├── proxy_server.py             # 메인 프록시 서버
├── config.py                   # 설정 (라우팅, 프리셋, 백엔드)
├── requirements.txt            # Python 의존성
├── README.md                   # 영어 버전
├── README_KOR.md               # 본 문서
├── .gitignore
└── motivation/                 # Motivation 실험 스크립트
    ├── motivation_experiments.py
    └── proxy_server_motivation.py
```

## 로그

서버는 다음 정보를 로깅합니다.

- 요청 포워딩 정보 (대상 서버, 엔드포인트)
- 에러 및 예외
- 서버 시작 / 종료 이벤트

## 문제 해결

### 백엔드 서버 연결 실패

- 백엔드가 실행 중인지 확인
- 방화벽 설정 확인
- 네트워크 연결 확인

### 타임아웃

- 기본 요청 타임아웃은 300초입니다.
- 필요 시 `httpx.AsyncClient(timeout=300.0)` 값을 조정해 사용하세요.

## 라이선스

MIT
