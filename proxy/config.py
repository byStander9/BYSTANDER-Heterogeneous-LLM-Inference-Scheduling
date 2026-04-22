"""
프록시 서버 설정 파일

┌──────────┐           ┌─────────────┐           ┌──────────────┐
│ 클라이언트 │ ========> │ 프록시 서버  │ ========> │ 백엔드 vLLM  │
└──────────┘           └─────────────┘           └──────────────┘
              ⬅️                          ➡️
         UVICORN_CONFIG          HTTP_CLIENT_CONFIG
        (인바운드 연결)           (아웃바운드 연결)
"""

# 백엔드 서버 설정
# gpu_type: 해당 서버의 GPU 종류 (같은 종류의 GPU가 여러 대일 수 있음)
BACKEND_SERVERS = [
    {"host": "BACKEND_RTX3090_A_HOST", "port": 18001, "name": "RTX3090_A", "gpu_type": "RTX3090"},
    {"host": "BACKEND_RTX3090_B_HOST", "port": 40321, "name": "RTX3090_B", "gpu_type": "RTX3090"},
    {"host": "BACKEND_RTX3090_C_HOST", "port": 28094, "name": "RTX3090_C", "gpu_type": "RTX3090"},
    {"host": "BACKEND_RTX4090_A_HOST", "port": 40556, "name": "RTX4090_A", "gpu_type": "RTX4090"},
    {"host": "BACKEND_RTX4090_B_HOST", "port": 54296, "name": "RTX4090_B", "gpu_type": "RTX4090"},
    {"host": "BACKEND_RTX5090_A_HOST", "port": 18004, "name": "RTX5090_A", "gpu_type": "RTX5090"},
    {"host": "BACKEND_RTX5090_B_HOST", "port": 20329, "name": "RTX5090_B", "gpu_type": "RTX5090"},
]

# GPU 종류 목록 (SLM 모델 출력 순서와 일치해야 함)
# 모델이 logits[0, 0]=RTX3090, logits[0, 1]=RTX4090, logits[0, 2]=RTX5090 순서로 출력
GPU_TYPES = ["RTX3090", "RTX4090", "RTX5090"]

# GPU 성능 비율 (RTX3090 기준, SLM 프롬프트에서 하드웨어 정보로 사용)
GPU_PERF_RATIOS = {
    "RTX3090": 1.0,
    "RTX4090": 2.0,
    "RTX5090": 2.5,
}

# 라우팅 알고리즘 설정
ROUTING_CONFIG = {
    # 알고리즘 선택: "round_robin", "weighted_round_robin", "shortest_queue_first", "slm_adaptive", "fisher_jenks_sqf"
    "algorithm": "fisher_jenks_sqf",
    
    # Weighted Round Robin 가중치 (서버 순서대로: RTX3090_A, RTX4090_A, RTX5090_A)
    "weights": [1, 3, 2],  # 자동 계산 사용 시 이 값은 무시됨
    
    # Shortest Queue First 설정
    "sqf_metric": "waiting",  # "waiting" or "total" (running+waiting)
    "sqf_fallback": "round_robin",  # 메트릭 수집 실패 시 fallback 알고리즘
    
    # SLM Adaptive Routing 설정
    "slm_model_path": "model/final_regression_multi_gpu",
    "slm_window_size": 20,
    "slm_activation_threshold": 15.0,
    "slm_deactivation_threshold": 10.0,
    "slm_base_algorithm": "shortest_queue_first",
    "slm_fallback": "shortest_queue_first",
    "slm_cache_timeout": 300,

    # SLM 배치 추론 설정
    "slm_batch_enabled": True,
    "slm_batch_max_size": 16,
    "slm_batch_max_wait_ms": 5.0,

    # Fisher-Jenks SQF 설정
    # SLM 예측 latency의 diff를 수집하여 후보 GPU 그룹을 선별한 뒤 SQF로 최종 서버 선택
    "fj_window_size": 50,              # diff 값 슬라이딩 윈도우 크기
    "fj_min_samples": 15,              # Fisher-Jenks 계산 최소 샘플 수
    "fj_default_threshold": 0.0,       # 샘플 부족 시 사용할 기본 diff 임계값 (초)
    "fj_window_reset_on_mode_change": True,  # act/deact 전환 시 윈도우 리셋 여부
    
    # 후보 선정 방법 (fisher_jenks_sqf 알고리즘 내에서 Step 5를 교체)
    # "fisher_jenks"    : (기본) 2-Class FJ Natural Breaks로 split point 산출
    # "percent_of_min"  : diff ≤ min_latency × fj_percent_threshold 이면 후보
    # "top_n"           : 예측 latency 상위 N개 GPU 종류 무조건 후보
    # "ewma"            : diff ≤ EWMA(diff) × fj_ewma_multiplier 이면 후보
    # "fixed_threshold" : diff ≤ fj_fixed_threshold 이면 후보
    "fj_candidate_method": "fisher_jenks",
    
    # 각 방법의 파라미터 (기본값 = ablation study 4th round에서 도출된 oracle 값)
    "fj_percent_threshold": 0.07,      # percent_of_min용 (sharegpt 기준, 논문 §V.D)
    "fj_top_n": 2,                     # top_n용
    "fj_ewma_alpha": 0.2,             # ewma용
    "fj_ewma_multiplier": 1.4,        # ewma용
    "fj_fixed_threshold": 10.0,        # fixed_threshold용 (sharegpt 기준)
}

# ============================================================================
# FJ 대안 비교 실험 프리셋 (7-GPU 실험 데이터에서 oracle 도출)
# ============================================================================
# 클라이언트가 /set_algorithm 호출 시 아래 딕셔너리의 값을 전달
#
# 파라미터 도출 근거:
#   pctmin = median(split_point) / median(min_latency)  → FJ 경계를 비율로 표현
#   fixed  = median(split_point)                        → FJ 경계의 절대값
#   lo/hi  = mid × 0.5 / mid × 1.5                     → ±50% sweep
#
#   ShareGPT : pctmin=0.07 (논문 §V.D)
#   LMSYS    : pctmin=0.22 (논문 §V.D)
#
EXPERIMENT_PRESETS = {
    # ── ShareGPT (QPS: 22-38, 24, 30, 36) ──
    "sharegpt_pctmin_lo":   {"fj_candidate_method": "percent_of_min", "fj_percent_threshold": 0.035},
    "sharegpt_pctmin_mid":  {"fj_candidate_method": "percent_of_min", "fj_percent_threshold": 0.07},
    "sharegpt_pctmin_hi":   {"fj_candidate_method": "percent_of_min", "fj_percent_threshold": 0.105},
    
    "sharegpt_topn_1":      {"fj_candidate_method": "top_n", "fj_top_n": 1},
    "sharegpt_topn_2":      {"fj_candidate_method": "top_n", "fj_top_n": 2},
    
    "sharegpt_fixed_lo":    {"fj_candidate_method": "fixed_threshold", "fj_fixed_threshold": 5.4},
    "sharegpt_fixed_mid":   {"fj_candidate_method": "fixed_threshold", "fj_fixed_threshold": 10.8},
    "sharegpt_fixed_hi":    {"fj_candidate_method": "fixed_threshold", "fj_fixed_threshold": 16.3},
    
    # ── LMSYS (QPS: 40, 45-75, 50, 60, 70) ──
    "lmsys_pctmin_lo":      {"fj_candidate_method": "percent_of_min", "fj_percent_threshold": 0.11},
    "lmsys_pctmin_mid":     {"fj_candidate_method": "percent_of_min", "fj_percent_threshold": 0.22},
    "lmsys_pctmin_hi":      {"fj_candidate_method": "percent_of_min", "fj_percent_threshold": 0.33},
    
    "lmsys_topn_1":         {"fj_candidate_method": "top_n", "fj_top_n": 1},
    "lmsys_topn_2":         {"fj_candidate_method": "top_n", "fj_top_n": 2},
    
    "lmsys_fixed_lo":       {"fj_candidate_method": "fixed_threshold", "fj_fixed_threshold": 5.9},
    "lmsys_fixed_mid":      {"fj_candidate_method": "fixed_threshold", "fj_fixed_threshold": 11.8},
    "lmsys_fixed_hi":       {"fj_candidate_method": "fixed_threshold", "fj_fixed_threshold": 17.7},
}

# ============================================================================
# 실험 매트릭스 (데이터셋 × 방법 × 파라미터)
# ============================================================================
# 클라이언트가 /set_algorithm 호출 시 아래 값들을 사용:
#
#  ┌─────────────┬──────────────┬────────────────────────────────────────────┐
#  │ Dataset     │ QPS          │ 실험할 프리셋 (위 EXPERIMENT_PRESETS 키)   │
#  ├─────────────┼──────────────┼────────────────────────────────────────────┤
#  │ ShareGPT    │ 22-38, 24,   │ sharegpt_pctmin_{lo,mid,hi}              │
#  │             │ 30, 36       │ sharegpt_topn_{1,2}                       │
#  │             │              │ sharegpt_fixed_{lo,mid,hi}                │
#  ├─────────────┼──────────────┼────────────────────────────────────────────┤
#  │ LMSYS       │ 40, 45-75,   │ lmsys_pctmin_{lo,mid,hi}                 │
#  │             │ 50, 60, 70   │ lmsys_topn_{1,2}                         │
#  │             │              │ lmsys_fixed_{lo,mid,hi}                   │
#  └─────────────┴──────────────┴────────────────────────────────────────────┘
#
#  총 실험 수: 8개 프리셋 × 데이터셋당 QPS 수 × 반복 횟수
#
#  /set_algorithm 호출 예시:
#    POST /set_algorithm
#    {
#      "algorithm": "fisher_jenks_sqf",
#      "slm_activation_threshold": 15,
#      "slm_deactivation_threshold": 10,
#      "fj_candidate_method": "percent_of_min",
#      "fj_percent_threshold": 0.17
#    }
#
# ============================================================================

# HTTP 클라이언트 설정 (프록시 → 백엔드 vLLM 서버)
HTTP_CLIENT_CONFIG = {
    # httpx.Limits 설정
    # 백엔드 서버로의 아웃바운드 연결 제한
    
    "max_connections": 100000,
    # 전체 최대 연결 수
    # 모든 백엔드 서버에 대한 총 연결 수 제한
    # ⚠️ None 불가! 무제한 원하면 매우 큰 값 사용 (예: 100000)
    
    "max_keepalive_connections": 50000,
    # 각 호스트(백엔드 서버)당 최대 Keep-Alive 연결 수
    # ⭐ 이 값이 각 백엔드 서버당 최대 동시 스트리밍 요청 수를 결정!
    # 예: 500 = 각 서버당 최대 500개의 동시 스트리밍
    # ⚠️ None 불가! 무제한 원하면 매우 큰 값 사용 (예: 50000)
    
    # httpx.Timeout 설정
    "timeout_connect": 10.0,      # 백엔드 서버 연결 타임아웃
    "timeout_read": 300.0,         # 응답 읽기 타임아웃 (무제한)
    "timeout_write": 10.0,        # 요청 쓰기 타임아웃
    "timeout_pool": 10.0,         # 연결 풀 대기 타임아웃
}

# Uvicorn 서버 설정 (클라이언트 → 프록시 서버)
UVICORN_CONFIG = {
    # 인바운드 연결 설정 (클라이언트로부터 받는 요청)
    
    "host": "0.0.0.0",
    "port": 8000,
    
    "workers": 1,
    # 워커 프로세스 수
    # 1 = 단일 프로세스 (상태 공유 가능)
    # 2+ = 멀티 프로세스 (CPU 병렬 처리, 상태 공유 불가)
    
    "log_level": "info",
    
    "limit_concurrency": 100000,
    # 애플리케이션 레벨: 최대 동시 처리 요청 수
    # 이 값을 초과하면 503 에러 즉시 반환
    
    "limit_max_requests": None,
    # 워커당 최대 처리 요청 수 (None = 무제한)
    # 설정 시: 해당 수만큼 처리 후 워커 재시작 (메모리 누수 방지)
    
    "backlog": 4096,
    # OS/TCP 레벨: TCP 연결은 됐지만 accept() 대기 중인 연결 큐
    # limit_concurrency와는 다른 레벨! (훨씬 더 앞 단계)
    # 서버가 일시적으로 바쁠 때 TCP 연결을 버퍼링
    
    "timeout_keep_alive": 5,
    # 클라이언트 Keep-Alive 타임아웃 (초)
    # 유휴 연결 유지 시간
}

# 메트릭 수집 설정
METRICS_CONFIG = {
    "csv_path": "metrics_log.csv",
    "collection_timeout": 5.0,     # 메트릭 수집 타임아웃
    "enabled": True,               # 메트릭 수집 활성화
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# 성능 모니터링
PERFORMANCE_CONFIG = {
    "log_slow_requests": True,     # 느린 요청 로깅
    "slow_request_threshold": 60.0, # 느린 요청 기준 (초)
}

