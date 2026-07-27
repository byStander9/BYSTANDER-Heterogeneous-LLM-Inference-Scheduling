"""
vLLM 추론 서버 프록시 서버

기능:
- OpenAI 호환 Chat Completions API (/v1/chat/completions)
- 스트리밍/비스트리밍 지원
- 라우팅 알고리즘: Round Robin / Weighted Round Robin / Shortest Queue First
- 메트릭 수집: 서버 상태(running/waiting), TTFT, E2E latency
- Graceful shutdown 지원

설정 (config.py):
- ROUTING_CONFIG["algorithm"]: "round_robin", "weighted_round_robin", "shortest_queue_first"
- ROUTING_CONFIG["weights"]: [RTX3090_weight, RTX5090_weight]
  예: [1, 18] = RTX5090에 18배 더 많은 요청 전달
- ROUTING_CONFIG["sqf_metric"]: "waiting" (대기 요청 수) 또는 "total" (전체 요청 수)
"""

import asyncio
import atexit
import json
import logging
import csv
import os
import re
import signal
import shutil
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from collections import deque

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 설정 파일 임포트
try:
    from config import (
        BACKEND_SERVERS,
        HTTP_CLIENT_CONFIG,
        UVICORN_CONFIG,
        METRICS_CONFIG,
        LOGGING_CONFIG,
        PERFORMANCE_CONFIG,
        ROUTING_CONFIG,
        GPU_TYPES,
        GPU_PERF_RATIOS,
        EXPERIMENT_PRESETS
    )
except ImportError:
    # 기본 설정 (config.py가 없는 경우)
    BACKEND_SERVERS = [
        {"host": "BACKEND_RTX3090_HOST", "port": 18001, "name": "RTX3090_SERVER", "gpu_type": "RTX3090"},
        {"host": "BACKEND_RTX5090_HOST", "port": 18004, "name": "RTX5090_SERVER", "gpu_type": "RTX5090"},
    ]
    HTTP_CLIENT_CONFIG = {
        "max_connections": 1000,
        "max_keepalive_connections": 500,
        "timeout_connect": 10.0,
        "timeout_read": 600.0,
        "timeout_write": 10.0,
        "timeout_pool": 10.0,
    }
    UVICORN_CONFIG = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "log_level": "info",
        "limit_concurrency": 10000,
        "backlog": 4096,
        "timeout_keep_alive": 5,
    }
    METRICS_CONFIG = {
        "csv_path": "metrics_log.csv",
        "collection_timeout": 5.0,
        "enabled": True,
    }
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    PERFORMANCE_CONFIG = {
        "log_slow_requests": True,
        "slow_request_threshold": 30.0,
    }
    ROUTING_CONFIG = {
        "algorithm": "round_robin",
        "weights": [1, 18],
    }
    GPU_TYPES = ["RTX3090", "RTX5090"]
    GPU_PERF_RATIOS = {"RTX3090": 1.0, "RTX5090": 2.5}

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# httpx/httpcore의 HTTP 요청 로그 억제 (백그라운드 메트릭 수집 시 대량 로그 방지)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def _percentile(sorted_list, p):
    """정렬된 리스트에서 백분위수 계산 (numpy 없이)"""
    if not sorted_list:
        return 0
    n = len(sorted_list)
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_list[-1]
    return sorted_list[f] + (k - f) * (sorted_list[c] - sorted_list[f])


def generate_summary_csv(raw_csv_path: str, summary_csv_path: str, backend_servers: list, gpu_types: list = None):
    """
    원본 메트릭 CSV에서 summary CSV를 생성
    
    ⚠️ 위치 기반 파싱 사용: CSV 헤더와 데이터의 서버 수가 불일치할 수 있으므로
    DictReader 대신 backend_servers로부터 열 위치를 계산하여 파싱합니다.
    
    데이터 형식 (열 순서):
      request_id, timestamp, target_server, target_server_port,
      [server_running, server_waiting] × num_servers,
      ttft_seconds, e2e_latency_seconds, arrival_timestamp, proxy_overhead_seconds, slm_inference_seconds,
      slm_mode_active, slm_prediction_used, slm_latency_diff, slm_predicted_selected_server_latency,
      [slm_predicted_<gpu_type>] × num_gpu_types, recent_avg_e2e
    
    포함 내용:
    - 전체 통계: 레코드 수, 성공률, E2E/TTFT (Avg, P50, P90, P95, P99), 초당 처리량
    - 프록시 오버헤드 통계: Proxy Overhead (Avg, P50, P90, P95, P99) - 핸들러 도착→백엔드 요청 전송
    - SLM 추론 시간 통계: SLM Inference (Avg, P50, P90, P95, P99) - SLM 추론 소요 시간 및 비중
    - GPU별 통계: 부하(running/waiting/total), 라우팅 비율, E2E/TTFT (Avg, P50, P90, P95, P99)
    - SLM 통계 (해당 시)
    """
    import csv as csv_mod
    from pathlib import Path
    
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        return
    
    # 서버명 리스트
    server_names = [s['name'] for s in backend_servers]
    num_servers = len(backend_servers)
    
    # === 열 위치 계산 (backend_servers 기준) ===
    COL_REQUEST_ID = 0
    COL_TIMESTAMP = 1
    COL_TARGET_SERVER = 2
    COL_TARGET_SERVER_PORT = 3
    COL_FIRST_SERVER_METRIC = 4  # 여기서부터 [running, waiting] × num_servers
    COL_TTFT = COL_FIRST_SERVER_METRIC + num_servers * 2
    COL_E2E = COL_TTFT + 1
    COL_ARRIVAL_TIMESTAMP = COL_E2E + 1
    COL_PROXY_OVERHEAD = COL_ARRIVAL_TIMESTAMP + 1
    COL_SLM_INFERENCE = COL_PROXY_OVERHEAD + 1
    COL_SLM_MODE_ACTIVE = COL_SLM_INFERENCE + 1
    COL_SLM_PREDICTION_USED = COL_SLM_MODE_ACTIVE + 1
    
    # 서버 인덱스 → running/waiting 열 위치 매핑
    server_col_map = {}
    for i, name in enumerate(server_names):
        server_col_map[name] = {
            'running': COL_FIRST_SERVER_METRIC + i * 2,
            'waiting': COL_FIRST_SERVER_METRIC + i * 2 + 1
        }
    
    # GPU 종류 → 서버 매핑
    type_to_servers = {}
    for s in backend_servers:
        gtype = s.get('gpu_type', s['name'])
        if gtype not in type_to_servers:
            type_to_servers[gtype] = []
        type_to_servers[gtype].append(s['name'])
    
    def safe_float(row, col_idx):
        """행에서 안전하게 float 값을 추출"""
        if col_idx >= len(row):
            return None
        val = row[col_idx].strip() if row[col_idx] else ''
        if not val or val == 'None' or val == '':
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    
    def safe_str(row, col_idx):
        """행에서 안전하게 문자열 값을 추출"""
        if col_idx >= len(row):
            return ''
        return row[col_idx].strip() if row[col_idx] else ''
    
    # 데이터 수집
    per_server_ttft = {name: [] for name in server_names}
    per_server_e2e = {name: [] for name in server_names}
    per_server_running = {name: [] for name in server_names}
    per_server_waiting = {name: [] for name in server_names}
    per_server_count = {name: 0 for name in server_names}
    
    all_ttft = []
    all_e2e = []
    all_proxy_overhead = []
    all_slm_inference = []
    all_arrival_times = []  # arrival_timestamp (ISO) → datetime 변환용
    slm_mode_active_count = 0
    slm_prediction_used_count = 0
    total_rows = 0
    success_count = 0
    
    first_timestamp = None
    last_timestamp = None
    
    with open(raw_path, 'r') as f:
        reader = csv_mod.reader(f)
        header = next(reader, None)  # 헤더 행 건너뛰기
        if header is None:
            return
        
        for row in reader:
            if not row or len(row) < COL_E2E + 1:
                continue  # 최소한 E2E 열까지 있어야 함
            
            total_rows += 1
            
            target = safe_str(row, COL_TARGET_SERVER)
            
            # 타임스탬프 추적
            ts = safe_str(row, COL_TIMESTAMP)
            if ts:
                if first_timestamp is None:
                    first_timestamp = ts
                last_timestamp = ts
            
            # TTFT, E2E (위치 기반)
            ttft = safe_float(row, COL_TTFT)
            e2e = safe_float(row, COL_E2E)
            
            if ttft is not None and e2e is not None:
                success_count += 1
                all_ttft.append(ttft)
                all_e2e.append(e2e)
                
                if target in per_server_ttft:
                    per_server_ttft[target].append(ttft)
                    per_server_e2e[target].append(e2e)
            
            # 서버별 요청 수
            if target in per_server_count:
                per_server_count[target] += 1
            
            # 서버별 running/waiting (위치 기반)
            for name in server_names:
                cols = server_col_map[name]
                r_val = safe_float(row, cols['running'])
                w_val = safe_float(row, cols['waiting'])
                if r_val is not None:
                    per_server_running[name].append(r_val)
                if w_val is not None:
                    per_server_waiting[name].append(w_val)
            
            # Proxy Overhead (위치 기반)
            po = safe_float(row, COL_PROXY_OVERHEAD)
            if po is not None:
                all_proxy_overhead.append(po)
            
            # SLM Inference Time (위치 기반)
            si = safe_float(row, COL_SLM_INFERENCE)
            if si is not None:
                all_slm_inference.append(si)
            
            # Arrival timestamp 수집 (도착률 계산용)
            arrival_ts = safe_str(row, COL_ARRIVAL_TIMESTAMP)
            if arrival_ts:
                try:
                    from datetime import datetime as dt2
                    all_arrival_times.append(dt2.fromisoformat(arrival_ts))
                except:
                    pass
            
            # SLM 통계 (위치 기반)
            slm_active = safe_str(row, COL_SLM_MODE_ACTIVE)
            slm_used = safe_str(row, COL_SLM_PREDICTION_USED)
            if slm_active.lower() == 'true':
                slm_mode_active_count += 1
            if slm_used.lower() == 'true':
                slm_prediction_used_count += 1
    
    if total_rows == 0:
        return
    
    # 정렬 (백분위수 계산용)
    all_ttft_sorted = sorted(all_ttft)
    all_e2e_sorted = sorted(all_e2e)
    all_proxy_overhead_sorted = sorted(all_proxy_overhead) if all_proxy_overhead else []
    all_slm_inference_sorted = sorted(all_slm_inference) if all_slm_inference else []
    
    # 실험 시간 계산
    duration_sec = None
    if first_timestamp and last_timestamp:
        try:
            from datetime import datetime as dt
            t1 = dt.fromisoformat(first_timestamp)
            t2 = dt.fromisoformat(last_timestamp)
            duration_sec = (t2 - t1).total_seconds()
        except:
            pass
    
    # Summary 작성
    lines = []
    
    def add(metric, value):
        lines.append((metric, value))
    
    def add_blank():
        lines.append(('', ''))
    
    # === 전체 통계 ===
    add('=== Overall Statistics ===', '')
    add('Total Records', total_rows)
    add('Successful Requests', success_count)
    add('Success Rate', f"{(success_count / total_rows * 100):.1f}%")
    if duration_sec and duration_sec > 0:
        add('Experiment Duration (s)', f"{duration_sec:.1f}")
        add('Effective Throughput (req/s)', f"{success_count / duration_sec:.2f}")
    
    add_blank()
    add('--- E2E Latency (seconds) ---', '')
    if all_e2e_sorted:
        add('E2E Avg', f"{sum(all_e2e) / len(all_e2e):.4f}")
        add('E2E P50', f"{_percentile(all_e2e_sorted, 50):.4f}")
        add('E2E P90', f"{_percentile(all_e2e_sorted, 90):.4f}")
        add('E2E P95', f"{_percentile(all_e2e_sorted, 95):.4f}")
        add('E2E P99', f"{_percentile(all_e2e_sorted, 99):.4f}")
        add('E2E Min', f"{all_e2e_sorted[0]:.4f}")
        add('E2E Max', f"{all_e2e_sorted[-1]:.4f}")
    
    add_blank()
    add('--- TTFT (seconds) ---', '')
    if all_ttft_sorted:
        add('TTFT Avg', f"{sum(all_ttft) / len(all_ttft):.4f}")
        add('TTFT P50', f"{_percentile(all_ttft_sorted, 50):.4f}")
        add('TTFT P90', f"{_percentile(all_ttft_sorted, 90):.4f}")
        add('TTFT P95', f"{_percentile(all_ttft_sorted, 95):.4f}")
        add('TTFT P99', f"{_percentile(all_ttft_sorted, 99):.4f}")
        add('TTFT Min', f"{all_ttft_sorted[0]:.4f}")
        add('TTFT Max', f"{all_ttft_sorted[-1]:.4f}")
    
    # === Proxy Overhead 통계 (핸들러 도착 → 백엔드 HTTP 요청 전송) ===
    add_blank()
    add('--- Proxy Overhead (seconds) ---', '')
    add('(Handler arrival → Backend HTTP request sent)', '')
    if all_proxy_overhead_sorted:
        add('Proxy Overhead Avg', f"{sum(all_proxy_overhead) / len(all_proxy_overhead):.4f}")
        add('Proxy Overhead P50', f"{_percentile(all_proxy_overhead_sorted, 50):.4f}")
        add('Proxy Overhead P90', f"{_percentile(all_proxy_overhead_sorted, 90):.4f}")
        add('Proxy Overhead P95', f"{_percentile(all_proxy_overhead_sorted, 95):.4f}")
        add('Proxy Overhead P99', f"{_percentile(all_proxy_overhead_sorted, 99):.4f}")
        add('Proxy Overhead Min', f"{all_proxy_overhead_sorted[0]:.4f}")
        add('Proxy Overhead Max', f"{all_proxy_overhead_sorted[-1]:.4f}")
        add('Proxy Overhead Count', len(all_proxy_overhead))
    else:
        add('Proxy Overhead', 'N/A (arrival_time not recorded)')
    
    # === SLM Inference 통계 (SLM 추론 시간) ===
    add_blank()
    add('--- SLM Inference Time (seconds) ---', '')
    if all_slm_inference_sorted:
        slm_avg = sum(all_slm_inference) / len(all_slm_inference)
        add('SLM Inference Avg', f"{slm_avg:.4f}")
        add('SLM Inference P50', f"{_percentile(all_slm_inference_sorted, 50):.4f}")
        add('SLM Inference P90', f"{_percentile(all_slm_inference_sorted, 90):.4f}")
        add('SLM Inference P95', f"{_percentile(all_slm_inference_sorted, 95):.4f}")
        add('SLM Inference P99', f"{_percentile(all_slm_inference_sorted, 99):.4f}")
        add('SLM Inference Min', f"{all_slm_inference_sorted[0]:.4f}")
        add('SLM Inference Max', f"{all_slm_inference_sorted[-1]:.4f}")
        add('SLM Inference Count', len(all_slm_inference))
        # SLM 추론이 프록시 오버헤드에서 차지하는 비율
        if all_proxy_overhead_sorted:
            po_avg = sum(all_proxy_overhead) / len(all_proxy_overhead)
            if po_avg > 0:
                add('SLM Inference / Proxy Overhead (%)', f"{(slm_avg / po_avg * 100):.1f}%")
                # 요청별 비율의 중앙값도 계산 (매칭 가능한 경우)
                paired_ratios = []
                for i in range(min(len(all_proxy_overhead), len(all_slm_inference))):
                    if all_proxy_overhead[i] > 0:
                        paired_ratios.append(all_slm_inference[i] / all_proxy_overhead[i])
                if paired_ratios:
                    paired_sorted = sorted(paired_ratios)
                    add('Per-Request SLM/Overhead Ratio P50', f"{_percentile(paired_sorted, 50) * 100:.1f}%")
    else:
        add('SLM Inference', 'N/A (non-SLM algorithm or no inference data)')
    
    # === Arrival Rate 통계 (arrival_timestamp 기반 실제 도착률) ===
    if len(all_arrival_times) >= 2:
        add_blank()
        add('--- Arrival Rate (based on arrival_timestamp) ---', '')
        all_arrival_times_sorted = sorted(all_arrival_times)
        arrival_duration = (all_arrival_times_sorted[-1] - all_arrival_times_sorted[0]).total_seconds()
        if arrival_duration > 0:
            overall_arrival_rate = (len(all_arrival_times) - 1) / arrival_duration
            add('Overall Arrival Rate (req/s)', f"{overall_arrival_rate:.2f}")
            add('Arrival Duration (s)', f"{arrival_duration:.1f}")
            add('Arrival Count', len(all_arrival_times))
            
            # 초당 도착 수 분포 (첫 도착 기준 버킷)
            t0 = all_arrival_times_sorted[0]
            per_second_counts = {}
            for at in all_arrival_times_sorted:
                sec_bucket = int((at - t0).total_seconds())
                per_second_counts[sec_bucket] = per_second_counts.get(sec_bucket, 0) + 1
            
            if per_second_counts:
                counts = list(per_second_counts.values())
                counts_sorted = sorted(counts)
                add('Per-Second Arrival Min', min(counts))
                add('Per-Second Arrival Max', max(counts))
                add('Per-Second Arrival Avg', f"{sum(counts) / len(counts):.1f}")
                add('Per-Second Arrival P50', f"{_percentile(counts_sorted, 50):.0f}")
    
    # === GPU별 통계 ===
    add_blank()
    for name in server_names:
        # GPU 종류 찾기
        gpu_type = name
        for s in backend_servers:
            if s['name'] == name:
                gpu_type = s.get('gpu_type', name)
                break
        
        pool_size = len(type_to_servers.get(gpu_type, [name]))
        pool_label = f" (pool: {gpu_type}, {pool_size}대)" if pool_size > 1 else ""
        
        add(f'=== {name} Statistics{pool_label} ===', '')
        
        # 부하 정보
        r_list = per_server_running[name]
        w_list = per_server_waiting[name]
        avg_r = sum(r_list) / len(r_list) if r_list else 0
        avg_w = sum(w_list) / len(w_list) if w_list else 0
        max_r = max(r_list) if r_list else 0
        max_w = max(w_list) if w_list else 0
        
        add('Avg Running', f"{avg_r:.2f}")
        add('Avg Waiting', f"{avg_w:.2f}")
        add('Avg Total', f"{avg_r + avg_w:.2f}")
        add('Max Running', f"{max_r:.0f}")
        add('Max Waiting', f"{max_w:.0f}")
        
        # 라우팅 비율
        cnt = per_server_count[name]
        pct = (cnt / total_rows * 100) if total_rows > 0 else 0
        add('Requests Sent', cnt)
        add('Requests Percent', f"{pct:.1f}%")
        
        # GPU별 E2E / TTFT
        s_e2e = sorted(per_server_e2e[name])
        s_ttft = sorted(per_server_ttft[name])
        
        if s_e2e:
            add('E2E Avg', f"{sum(per_server_e2e[name]) / len(per_server_e2e[name]):.4f}")
            add('E2E P50', f"{_percentile(s_e2e, 50):.4f}")
            add('E2E P90', f"{_percentile(s_e2e, 90):.4f}")
            add('E2E P95', f"{_percentile(s_e2e, 95):.4f}")
            add('E2E P99', f"{_percentile(s_e2e, 99):.4f}")
        
        if s_ttft:
            add('TTFT Avg', f"{sum(per_server_ttft[name]) / len(per_server_ttft[name]):.4f}")
            add('TTFT P50', f"{_percentile(s_ttft, 50):.4f}")
            add('TTFT P90', f"{_percentile(s_ttft, 90):.4f}")
            add('TTFT P95', f"{_percentile(s_ttft, 95):.4f}")
            add('TTFT P99', f"{_percentile(s_ttft, 99):.4f}")
        
        add_blank()
    
    # === GPU 종류(풀)별 통합 통계 ===
    if any(len(svrs) > 1 for svrs in type_to_servers.values()):
        # 풀 내 서버가 2대 이상인 경우에만 풀 통합 통계 표시
        for gtype, svrs in type_to_servers.items():
            if len(svrs) < 2:
                continue
            
            add(f'=== {gtype} Pool Combined ({len(svrs)}대) ===', '')
            
            # 풀 전체 요청 수
            pool_count = sum(per_server_count.get(s, 0) for s in svrs)
            pool_pct = (pool_count / total_rows * 100) if total_rows > 0 else 0
            add('Pool Total Requests', pool_count)
            add('Pool Requests Percent', f"{pool_pct:.1f}%")
            
            # 풀 전체 E2E / TTFT
            pool_e2e = []
            pool_ttft = []
            for s in svrs:
                pool_e2e.extend(per_server_e2e.get(s, []))
                pool_ttft.extend(per_server_ttft.get(s, []))
            
            pool_e2e_sorted = sorted(pool_e2e)
            pool_ttft_sorted = sorted(pool_ttft)
            
            if pool_e2e_sorted:
                add('E2E Avg', f"{sum(pool_e2e) / len(pool_e2e):.4f}")
                add('E2E P50', f"{_percentile(pool_e2e_sorted, 50):.4f}")
                add('E2E P90', f"{_percentile(pool_e2e_sorted, 90):.4f}")
                add('E2E P95', f"{_percentile(pool_e2e_sorted, 95):.4f}")
                add('E2E P99', f"{_percentile(pool_e2e_sorted, 99):.4f}")
            
            if pool_ttft_sorted:
                add('TTFT Avg', f"{sum(pool_ttft) / len(pool_ttft):.4f}")
                add('TTFT P50', f"{_percentile(pool_ttft_sorted, 50):.4f}")
                add('TTFT P90', f"{_percentile(pool_ttft_sorted, 90):.4f}")
                add('TTFT P95', f"{_percentile(pool_ttft_sorted, 95):.4f}")
                add('TTFT P99', f"{_percentile(pool_ttft_sorted, 99):.4f}")
            
            # 풀 내 평균 부하
            pool_avg_r = sum(sum(per_server_running[s]) / len(per_server_running[s]) if per_server_running[s] else 0 for s in svrs) / len(svrs)
            pool_avg_w = sum(sum(per_server_waiting[s]) / len(per_server_waiting[s]) if per_server_waiting[s] else 0 for s in svrs) / len(svrs)
            add('Avg Running (per server)', f"{pool_avg_r:.2f}")
            add('Avg Waiting (per server)', f"{pool_avg_w:.2f}")
            add('Avg Total (per server)', f"{pool_avg_r + pool_avg_w:.2f}")
            
            add_blank()
    
    # === SLM 통계 ===
    if slm_mode_active_count > 0 or slm_prediction_used_count > 0:
        add('=== SLM Adaptive Statistics ===', '')
        base_count = total_rows - slm_mode_active_count
        add('Base Algorithm Mode Count', base_count)
        add('Base Algorithm Mode Percent', f"{(base_count / total_rows * 100):.1f}%")
        add('SLM Mode Count', slm_mode_active_count)
        add('SLM Mode Percent', f"{(slm_mode_active_count / total_rows * 100):.1f}%")
        add('SLM Prediction Used Count', slm_prediction_used_count)
        add('SLM Prediction Used Percent', f"{(slm_prediction_used_count / total_rows * 100):.1f}%")
    
    # 파일 쓰기
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv_mod.writer(f)
        writer.writerow(['Metric', 'Value'])
        for metric, value in lines:
            writer.writerow([metric, value])

# FastAPI 앱 생성
app = FastAPI(title="vLLM Proxy Server", version="1.0.0")


class SLMBatchInferencer:
    """
    SLM 배치 추론기
    
    여러 요청의 SLM 추론을 모아서 한번에 GPU에서 처리합니다.
    - 최대 max_batch_size개까지 모아서 배치 추론
    - max_wait_ms 이내에 배치가 안 차면 현재까지 모인 것으로 추론
    - 이벤트 루프를 블로킹하지 않음 (GPU 추론은 전용 단일 스레드에서 실행)
    
    효과: batch_size=1 추론 380ms → batch_size=8이면 ~400ms에 8개 처리 (~8배 처리량)
    """
    
    def __init__(self, max_batch_size: int = 16, max_wait_ms: float = 50.0,
                 executor: ThreadPoolExecutor = None):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.model = None
        self.tokenizer = None
        self._device = None
        self._executor = executor  # 전용 단일 스레드 executor
        self._queue: asyncio.Queue = None
        self._task = None
        self._stats = {
            'total_batches': 0,
            'total_items': 0,
            'max_batch_seen': 0,
        }
    
    def set_model(self, model, tokenizer, device=None):
        """모델과 토크나이저 설정 (lazy loading 후 호출)"""
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
    
    async def start(self):
        """배치 추론 루프 시작"""
        if self._task is not None:
            return
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._batch_loop())
        logger.info(f"[SLM-BATCH] Batch inferencer started (max_batch={self.max_batch_size}, max_wait={self.max_wait_ms}ms)")
    
    async def stop(self):
        """배치 추론 루프 중지"""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("[SLM-BATCH] Batch inferencer stopped")
    
    async def predict(self, slm_prompt: str) -> tuple:
        """
        배치 큐에 추론 요청 제출 후 결과 대기
        
        Args:
            slm_prompt: SLM에 입력할 프롬프트 문자열
            
        Returns:
            tuple: (logits[1, num_gpu_types], inference_time_ms)
        """
        if self._queue is None:
            await self.start()
        
        submit_time = time.perf_counter()
        queue_depth = self._queue.qsize()
        
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((slm_prompt, future))
        result = await future
        
        queue_wait_ms = (time.perf_counter() - submit_time) * 1000
        self._stats['last_queue_depth'] = queue_depth
        self._stats['last_queue_wait_ms'] = queue_wait_ms
        
        return result
    
    async def _batch_loop(self):
        """배치 수집 + 추론 루프"""
        while True:
            try:
                batch = []
                
                # 첫 번째 아이템 대기 (무한 대기)
                item = await self._queue.get()
                batch.append(item)
                
                # 추가 아이템 수집: 즉시 drain → 적응적 대기
                # Phase 1: 큐에 이미 쌓여있는 아이템을 대기 없이 즉시 drain
                while len(batch) < self.max_batch_size:
                    try:
                        item = self._queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break
                
                # Phase 2: drain 후 큐가 비어있으면 짧은 대기로 동시 도착 요청만 수집
                # - 배치가 2개 이상이면 이미 부하가 있으므로 즉시 처리 (대기 없음)
                # - 배치가 1개이고 큐가 비어있으면 max_wait_ms만 대기
                #   → 이 대기 시간 안에 도착하는 요청이 있으면 함께 배치 처리
                #   → 없으면 즉시 1개로 처리
                if len(batch) == 1 and len(batch) < self.max_batch_size and self._queue.empty():
                    if self.max_wait_ms > 0:
                        try:
                            item = await asyncio.wait_for(
                                self._queue.get(), timeout=self.max_wait_ms / 1000.0
                            )
                            batch.append(item)
                            while len(batch) < self.max_batch_size:
                                try:
                                    item = self._queue.get_nowait()
                                    batch.append(item)
                                except asyncio.QueueEmpty:
                                    break
                        except asyncio.TimeoutError:
                            pass
                
                # 배치 추론 실행
                prompts = [p for p, f in batch]
                futures = [f for p, f in batch]
                batch_size = len(batch)
                
                try:
                    logits_batch, inference_time = await self._run_batch_inference(prompts)
                    
                    # 각 요청에 결과 전달
                    per_item_time = inference_time / batch_size
                    for i, future in enumerate(futures):
                        if not future.cancelled():
                            future.set_result((logits_batch[i:i+1], per_item_time))
                    
                    # 통계 업데이트
                    self._stats['total_batches'] += 1
                    self._stats['total_items'] += batch_size
                    self._stats['max_batch_seen'] = max(self._stats['max_batch_seen'], batch_size)
                    
                    if self._stats['total_batches'] % 50 == 0:
                        avg_batch = self._stats['total_items'] / self._stats['total_batches']
                        logger.info(f"[SLM-BATCH] Stats: {self._stats['total_batches']} batches, "
                                   f"avg_size={avg_batch:.1f}, max_size={self._stats['max_batch_seen']}, "
                                   f"last={batch_size}, inference={inference_time:.1f}ms")
                    
                except Exception as e:
                    logger.error(f"[SLM-BATCH] Batch inference error: {e}")
                    for future in futures:
                        if not future.cancelled():
                            future.set_exception(e)
                            
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[SLM-BATCH] Batch loop error: {e}")
    
    async def _run_batch_inference(self, prompts: List[str]) -> tuple:
        """배치 추론 실행 (전용 단일 스레드에서)"""
        model = self.model
        tokenizer = self.tokenizer
        device = self._device
        
        def _inference():
            full_start = time.perf_counter()
            inputs = tokenizer(
                prompts,
                max_length=512,
                padding=True,  # 배치 내 최대 길이로만 패딩 (512 전체 패딩 방지)
                truncation=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits  # shape: [batch_size, num_gpu_types]
            inference_time = (time.perf_counter() - full_start) * 1000  # ms
            
            return logits.cpu(), inference_time
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _inference)
    
    def get_stats(self) -> Dict:
        """배치 추론 통계 반환"""
        stats = dict(self._stats)
        if stats['total_batches'] > 0:
            stats['avg_batch_size'] = round(stats['total_items'] / stats['total_batches'], 2)
        return stats


class ProxyServer:
    """프록시 서버 클래스"""
    
    def __init__(self, backend_servers: List[Dict], metrics_csv_path: str = None):
        self.backend_servers = backend_servers
        self.current_server_index = 0
        self.request_count = 0
        self.metrics_csv_path = metrics_csv_path or METRICS_CONFIG["csv_path"]
        
        # GPU 풀 구성 (동일 종류 GPU를 그룹화)
        self.gpu_types = GPU_TYPES  # SLM 모델 출력 순서와 일치
        self.gpu_perf_ratios = GPU_PERF_RATIOS
        self.gpu_pools = {}  # {gpu_type: [server_dict, ...]}
        for server in self.backend_servers:
            gpu_type = server.get('gpu_type', server['name'])
            if gpu_type not in self.gpu_pools:
                self.gpu_pools[gpu_type] = []
            self.gpu_pools[gpu_type].append(server)
        
        pool_info = ', '.join([f"{t}: {len(s)}대" for t, s in self.gpu_pools.items()])
        logger.info(f"GPU Pools: [{pool_info}]")
        
        # 현재 실험에서 사용 중인 데이터셋 (finalize 시 결과 저장 경로에 사용)
        self.current_dataset = None  # "sharegpt", "lmsys-chat-1m" 등
        self.current_qps = None  # 클라이언트가 설정한 QPS 값
        
        # 라우팅 알고리즘 설정
        self.routing_algorithm = ROUTING_CONFIG.get("algorithm", "round_robin")
        self.routing_weights = ROUTING_CONFIG.get("weights", [1] * len(backend_servers))
        self.sqf_metric = ROUTING_CONFIG.get("sqf_metric", "waiting")
        self.sqf_fallback = ROUTING_CONFIG.get("sqf_fallback", "round_robin")
        
        # Weighted Round Robin용 가중치 순환 리스트 생성
        # weight를 정수로 변환하여 순환 리스트 생성
        # 예: weights=[1, 2, 2.5] → 10배 스케일링 → [10, 20, 25] → [0]*10 + [1]*20 + [2]*25
        self.weighted_server_sequence = []
        self._build_weighted_sequence()
        self.weighted_index = 0
        
        # 진행 중인 요청 추적
        self.active_requests = 0
        self.active_requests_lock = asyncio.Lock()
        
        # 요청 도착률 추적 (병목 진단용)
        self.request_arrival_times = deque(maxlen=5000)  # 최근 5000개 도착 시각
        self._arrival_log_interval = 5  # N초마다 도착률 로그 출력
        self._last_arrival_log_time = None
        self._last_arrival_log_count = 0
        
        # 종료 상태 관리
        self.shutting_down = False
        self.shutdown_event = asyncio.Event()
        
        # HTTP 클라이언트 연결 풀 설정
        self.http_limits = httpx.Limits(
            max_connections=HTTP_CLIENT_CONFIG["max_connections"],
            max_keepalive_connections=HTTP_CLIENT_CONFIG["max_keepalive_connections"]
        )
        
        self.http_timeout = httpx.Timeout(
            connect=HTTP_CLIENT_CONFIG["timeout_connect"],
            read=HTTP_CLIENT_CONFIG["timeout_read"],
            write=HTTP_CLIENT_CONFIG["timeout_write"],
            pool=HTTP_CLIENT_CONFIG["timeout_pool"]
        )
        self.total_request_timeout = HTTP_CLIENT_CONFIG["timeout_read"]
        
        # 공유 HTTP 클라이언트 (연결 풀 재사용)
        self._http_client = None
        
        # 메트릭 수집용 공유 HTTP 클라이언트 (매번 생성하지 않고 재사용)
        self._metrics_http_client = None
        
        # 백그라운드 메트릭 캐시 (이벤트 루프 부하 감소)
        self._cached_metrics = None  # Dict[str, Dict] - 가장 최근 수집된 메트릭
        self._cached_metrics_time = 0.0  # 캐시된 시각 (time.time())
        # 알고리즘별 캐시 갱신 주기: SQF/SLM은 라우팅 정확도를 위해 더 짧은 주기
        if self.routing_algorithm in ("shortest_queue_first", "slm_adaptive", "fisher_jenks_sqf"):
            self._metrics_cache_interval = 0.2  # 200ms 주기 (라우팅용)
        else:
            self._metrics_cache_interval = 0.5  # 500ms 주기 (로깅용)
        self._metrics_cache_task = None  # 백그라운드 태스크
        
        # SLM 공통 설정 (slm_adaptive, fisher_jenks_sqf 모두 사용)
        if self.routing_algorithm in ("slm_adaptive", "fisher_jenks_sqf"):
            self.slm_model_path = ROUTING_CONFIG.get("slm_model_path")
            self.slm_window_size = ROUTING_CONFIG.get("slm_window_size", 20)
            self.slm_activation_threshold = ROUTING_CONFIG.get("slm_activation_threshold", 25.0)
            self.slm_deactivation_threshold = ROUTING_CONFIG.get("slm_deactivation_threshold", 15.0)
            self.slm_base_algorithm = ROUTING_CONFIG.get("slm_base_algorithm", "weighted_round_robin")
            self.slm_fallback = ROUTING_CONFIG.get("slm_fallback", "weighted_round_robin")

            # SLM 상태 추적
            self.slm_active = False
            self.recent_e2e_latencies = deque(maxlen=self.slm_window_size)
            self.slm_mode_switches = 0

            self.slm_prediction_stats = {
                'total_predictions': 0,
            }
            # 각 서버별 선택 카운터 동적 추가
            for server in self.backend_servers:
                self.slm_prediction_stats[f"{server['name']}_selected"] = 0
            
            # Inflight 토큰 추적 (SLM 추론에 필요) - 동적으로 생성
            self.inflight_tokens = {}
            for server in self.backend_servers:
                self.inflight_tokens[server['name']] = deque(maxlen=500)
            
            # SLM 모델 로드 (lazy loading)
            self.slm_model = None
            self.slm_tokenizer = None
            self._slm_load_lock = asyncio.Lock()
            self._slm_device = None  # 모델 로드 후 캐싱
            
            # SLM 전용 ThreadPoolExecutor (max_workers=1)
            # GIL 경합과 CUDA stream 직렬화를 고려하여 단일 스레드로 운영.
            # 다수 스레드가 동시에 접근하면 컨텍스트 스위칭 + GIL 대기만 증가하므로
            # 단일 스레드가 오히려 처리량이 높다.
            self._slm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slm-inference")
            
            # SLM 배치 추론 설정 — 기본 활성화 (개별 추론 대비 처리량 대폭 향상)
            self.slm_batch_enabled = ROUTING_CONFIG.get("slm_batch_enabled", True)
            self.slm_batch_inferencer = SLMBatchInferencer(
                max_batch_size=ROUTING_CONFIG.get("slm_batch_max_size", 16),
                max_wait_ms=ROUTING_CONFIG.get("slm_batch_max_wait_ms", 50.0),
                executor=self._slm_executor
            )
            
            logger.info(f"  SLM Adaptive Routing enabled:")
            logger.info(f"    Model path: {self.slm_model_path}")
            logger.info(f"    Window size: {self.slm_window_size}")
            logger.info(f"    Activation threshold: {self.slm_activation_threshold}s")
            logger.info(f"    Deactivation threshold: {self.slm_deactivation_threshold}s")
            logger.info(f"    Base algorithm: {self.slm_base_algorithm}")
            logger.info(f"    Fallback algorithm: {self.slm_fallback}")
            logger.info(f"    Initial mode: {self.slm_base_algorithm.upper()} (SLM inactive)")
            logger.info(f"    Batch inference: {'ENABLED' if self.slm_batch_enabled else 'DISABLED (individual run_in_executor)'}")
        
        # Fisher-Jenks SQF 전용 설정
        if self.routing_algorithm == "fisher_jenks_sqf":
            self.fj_window_size = ROUTING_CONFIG.get("fj_window_size", 50)
            self.fj_min_samples = ROUTING_CONFIG.get("fj_min_samples", 15)
            self.fj_default_threshold = ROUTING_CONFIG.get("fj_default_threshold", 0.0)
            self.fj_window_reset_on_mode_change = ROUTING_CONFIG.get("fj_window_reset_on_mode_change", True)
            
            # 후보 선정 방법
            self.fj_candidate_method = ROUTING_CONFIG.get("fj_candidate_method", "fisher_jenks")
            self.fj_percent_threshold = ROUTING_CONFIG.get("fj_percent_threshold", 0.09)
            self.fj_top_n = ROUTING_CONFIG.get("fj_top_n", 2)
            self.fj_ewma_alpha = ROUTING_CONFIG.get("fj_ewma_alpha", 0.2)
            self.fj_ewma_multiplier = ROUTING_CONFIG.get("fj_ewma_multiplier", 1.4)
            self.fj_fixed_threshold = ROUTING_CONFIG.get("fj_fixed_threshold", 10.8)
            self.fj_ewma_value = 0.0  # EWMA 현재 값
            self.fj_ewma_initialized = False
            
            # Fisher-Jenks diff 버퍼 (1등 제외, 나머지 GPU의 diff 값만 저장)
            self.fj_diff_buffer = deque(maxlen=self.fj_window_size)
            self.fj_current_split_point = self.fj_default_threshold  # 초기값 = 기본 threshold
            self.fj_split_history = []  # split point 변화 추적 (디버깅용)
            
            # Fisher-Jenks 통계 (GPU 종류별 후보 그룹 통계)
            self.fj_stats = {
                'total_fj_computations': 0,
                'total_candidates_selected': 0,
                'fj_fallback_used': 0,
            }
            for gpu_type in self.gpu_types:
                self.fj_stats[f"{gpu_type}_in_candidate_group"] = 0
            
            logger.info(f"  Fisher-Jenks SQF enabled:")
            logger.info(f"    FJ candidate method: {self.fj_candidate_method}")
            logger.info(f"    FJ window size: {self.fj_window_size} (diff values)")
            logger.info(f"    FJ min samples: {self.fj_min_samples}")
            logger.info(f"    FJ default threshold: {self.fj_default_threshold}s")
            logger.info(f"    FJ window reset on mode change: {self.fj_window_reset_on_mode_change}")
            if self.fj_candidate_method == "percent_of_min":
                logger.info(f"    FJ percent threshold: {self.fj_percent_threshold}")
            elif self.fj_candidate_method == "top_n":
                logger.info(f"    FJ top_n: {self.fj_top_n}")
            elif self.fj_candidate_method == "ewma":
                logger.info(f"    FJ EWMA alpha: {self.fj_ewma_alpha}, multiplier: {self.fj_ewma_multiplier}")
            elif self.fj_candidate_method == "fixed_threshold":
                logger.info(f"    FJ fixed threshold: {self.fj_fixed_threshold}s")
        
        # ===== 병목 진단용 타이밍 수집기 =====
        self._diag = {
            # --- Handler 단계 (chat_completions) ---
            'handler_json_parse_ms': deque(maxlen=500),       # arrival → JSON 파싱 완료
            'handler_routing_ms': deque(maxlen=500),           # JSON 파싱 → 서버 선택 완료
            'handler_total_ms': deque(maxlen=500),             # arrival → 백엔드 포워딩 직전 (전체 프록시 오버헤드)
            
            # --- _predict_with_slm 내부 단계 ---
            'slm_metrics_ms': deque(maxlen=500),              # 메트릭 수집 (캐시 hit/miss)
            'slm_prompt_gen_ms': deque(maxlen=500),           # pool 집계 + 프롬프트 생성
            'slm_inference_ms': deque(maxlen=500),            # SLM 모델 추론 (토크나이징+포워드)
            'slm_total_ms': deque(maxlen=500),                # _predict_with_slm 전체
            
            # --- 배치 추론기 ---
            'batch_queue_wait_ms': deque(maxlen=500),         # 큐 제출 → 결과 수신 (큐 대기+추론)
            'batch_queue_depth': deque(maxlen=500),           # 제출 시점 큐 깊이
            'batch_size': deque(maxlen=500),                   # 배치 크기
            
            # --- 카운터 ---
            'total_diagnosed': 0,
            'diag_start_time': time.time(),
            'last_summary_request_id': 0,
            'summary_interval': 100,                           # N개 요청마다 요약 출력
        }
        
        self._init_csv_file()
        
        logger.info(f"ProxyServer initialized:")
        logger.info(f"  Routing algorithm: {self.routing_algorithm}")
        if self.routing_algorithm == "weighted_round_robin":
            logger.info(f"  Routing weights: {self.routing_weights}")
            logger.info(f"  Weight ratio: {':'.join(map(str, self.routing_weights))}")
        elif self.routing_algorithm == "shortest_queue_first":
            logger.info(f"  SQF metric: {self.sqf_metric}")
            logger.info(f"  SQF fallback: {self.sqf_fallback}")
        logger.info(f"  HTTP limits: max_connections={HTTP_CLIENT_CONFIG['max_connections']}, "
                   f"max_keepalive_connections={HTTP_CLIENT_CONFIG['max_keepalive_connections']}")
    
    async def get_http_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 가져오기 (연결 풀 재사용)"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                limits=self.http_limits,
                timeout=self.http_timeout,
                http2=True  # HTTP/2 지원 (성능 향상)
            )
        return self._http_client
    
    async def get_metrics_http_client(self) -> httpx.AsyncClient:
        """메트릭 수집용 공유 HTTP 클라이언트 (매번 생성하지 않고 재사용)"""
        if self._metrics_http_client is None:
            self._metrics_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(2.0, connect=1.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return self._metrics_http_client
    
    async def start_metrics_cache_task(self):
        """백그라운드 메트릭 캐시 태스크 시작 (모든 알고리즘에서 사용)"""
        if self._metrics_cache_task is not None:
            return  # 이미 실행 중
        
        self._metrics_cache_task = asyncio.create_task(self._metrics_cache_loop())
        logger.info(f"[METRICS-CACHE] Background metrics cache started "
                   f"(algorithm: {self.routing_algorithm}, interval: {self._metrics_cache_interval}s)")
    
    async def _metrics_cache_loop(self):
        """백그라운드에서 주기적으로 메트릭 수집하여 캐시"""
        _consecutive_errors = 0
        while True:
            try:
                metrics = await self._collect_all_metrics_direct(background=True)
                if metrics:
                    self._cached_metrics = metrics
                    self._cached_metrics_time = time.time()
                    if _consecutive_errors > 0:
                        logger.info(f"[METRICS-CACHE] Background metrics collection recovered after {_consecutive_errors} errors")
                    _consecutive_errors = 0
            except Exception as e:
                _consecutive_errors += 1
                # 연속 에러 시 처음 1회만 warning, 이후 100회마다 1회만 로그
                if _consecutive_errors == 1 or _consecutive_errors % 100 == 0:
                    logger.warning(f"[METRICS-CACHE] Background metrics error (count={_consecutive_errors}): {e}")
            await asyncio.sleep(self._metrics_cache_interval)
    
    def get_cached_metrics(self, allow_stale: bool = False) -> Optional[Dict]:
        """
        캐시된 메트릭 반환.
        
        Args:
            allow_stale: True면 오래된 캐시도 반환 (요청 경로에서 사용).
                         False면 TTL 초과 시 None 반환 (백그라운드 갱신 판단용).
        """
        if self._cached_metrics is None:
            return None
        if allow_stale:
            return self._cached_metrics
        age = time.time() - self._cached_metrics_time
        if age > self._metrics_cache_interval * 3:
            return None
        return self._cached_metrics
    
    def _build_weighted_sequence(self):
        """
        Weight 값에 따라 순환 서버 리스트 생성 (라운드 로빈 방식)
        
        동작 방식:
        - 각 라운드마다 모든 GPU에 하나씩 분배
        - Weight를 다 채운 GPU는 제외하고 나머지끼리 계속 분배
        
        예시: weights=[1, 2, 3]
        - Round 1: 3090(1회) → 4090(1회) → 5090(1회)
        - Round 2: 4090(1회) → 5090(1회)  (3090은 weight 소진)
        - Round 3: 5090(1회)  (4090도 weight 소진)
        
        결과 순서: 3090 → 4090 → 5090 → 4090 → 5090 → 5090 → 3090 → ...
        """
        self.weighted_server_sequence = []
        
        num_servers = len(self.backend_servers)
        if not self.routing_weights:
            self.weighted_server_sequence = list(range(num_servers))
            return
        
        # weights 길이가 서버 수와 다르면 보정
        if len(self.routing_weights) != num_servers:
            logger.warning(f"[WRR] weights length ({len(self.routing_weights)}) != server count ({num_servers}), padding with 1")
            padded = list(self.routing_weights)
            while len(padded) < num_servers:
                padded.append(1)
            self.routing_weights = padded[:num_servers]
        
        weights_remaining = [int(w) for w in self.routing_weights]
        max_weight = max(weights_remaining)
        
        # 라운드별로 분배
        for round_num in range(max_weight):
            # 이번 라운드에 참여할 서버들 (weight가 남은 서버들)
            active_servers = [idx for idx, w in enumerate(weights_remaining) if w > 0]
            
            # 활성 서버들에게 순서대로 하나씩 분배
            for idx in active_servers:
                self.weighted_server_sequence.append(idx)
                weights_remaining[idx] -= 1
        
        logger.info(f"[WRR] Weighted sequence built: {len(self.weighted_server_sequence)} slots")
        
        # 분배 비율 계산
        sequence_counts = [self.weighted_server_sequence.count(i) for i in range(len(self.backend_servers))]
        total_slots = len(self.weighted_server_sequence)
        
        logger.info(f"[WRR] Distribution:")
        for i, (count, server) in enumerate(zip(sequence_counts, self.backend_servers)):
            percentage = (count / total_slots) * 100 if total_slots > 0 else 0
            # weights 길이 체크 (weights가 서버 수보다 적을 수 있음)
            weight_value = int(self.routing_weights[i]) if i < len(self.routing_weights) else 0
            logger.info(f"       {server['name']}: weight={weight_value} → {count}/{total_slots} slots ({percentage:.1f}%)")

    def _calculate_weights_from_rr(self, qps, dataset=None) -> list:
        """
        RR 실험 결과에서 GPU 풀별 평균 total(running+waiting)을 계산하여 정수 weight 비율 반환
        
        계산 방식:
        1. RR CSV에서 각 물리 서버의 avg total 산출
        2. GPU 종류(풀)별로 avg total을 평균 → 풀 대표값
        3. 풀 대표값의 역수 비율로 weight 계산 (total 낮은 풀 → weight 높게)
        4. 같은 풀 내 모든 서버에 동일 weight 부여
        5. 최소 weight = 1 (base 풀도 서버당 1 이상 → 모든 GPU가 cycle당 최소 1회 라우팅)
        
        Args:
            qps: QPS 값 (예: 12.0, 18.0)
            dataset: 데이터셋 이름 (예: "sharegpt", "lmsys-chat-1m"). None이면 하위 폴더 없이 접근
            
        Returns:
            list: 정수 weight 리스트 (backend_servers 순서). 실패 시 빈 리스트 반환
        """
        num_gpus = len(self.backend_servers)
        base_dir = Path(__file__).parent / f"regressor_results_{num_gpus}gpu"
        if dataset:
            base_dir = base_dir / f"used_{dataset}"
        rr_dir = base_dir / f"QPS{qps}_round_robin"
        
        if not rr_dir.is_dir():
            found = False
            if base_dir.is_dir():
                qps_str = str(qps).replace(" ", "")
                norm_input = qps_str.replace(".0", "").replace(".", "")
                
                # 입력 QPS를 float로 변환 (범위 매칭용)
                try:
                    qps_float = float(str(qps).split("-")[0])
                except (ValueError, TypeError):
                    qps_float = None
                
                for candidate in sorted(base_dir.iterdir()):
                    if not candidate.is_dir() or not candidate.name.endswith("_round_robin"):
                        continue
                    if "weighted" in candidate.name:
                        continue
                    
                    cand_qps = candidate.name.replace("QPS", "").replace("_round_robin", "")
                    
                    # 1차: 정규화 문자열 매칭 (10-20 == 10.0-20.0)
                    norm_cand = cand_qps.replace(".0", "").replace(".", "")
                    if norm_input == norm_cand:
                        rr_dir = candidate
                        found = True
                        logger.info(f"[WRR Auto] Fuzzy matched: QPS{qps} → {candidate.name}")
                        break
                    
                    # 2차: 범위 폴더 매칭 (qps=10.0 → QPS10.0-20.0 범위에 포함)
                    if qps_float is not None and "-" in cand_qps:
                        try:
                            parts = cand_qps.split("-")
                            range_min = float(parts[0])
                            range_max = float(parts[1])
                            if range_min <= qps_float <= range_max:
                                rr_dir = candidate
                                found = True
                                logger.info(f"[WRR Auto] Range matched: QPS{qps} in {candidate.name} ({range_min}-{range_max})")
                                break
                        except (ValueError, IndexError):
                            pass
                
            if not found:
                logger.warning(f"[WRR Auto] RR result folder not found: {rr_dir}")
                return []
        
        # 원본 CSV 파일만 수집 (summary, load_analysis 등 제외)
        raw_csvs = []
        for f in sorted(rr_dir.iterdir()):
            if (f.suffix == '.csv' and
                '_summary' not in f.name and
                '_load_analysis' not in f.name and
                '_mae_analysis' not in f.name and
                '_prediction_comparison' not in f.name):
                raw_csvs.append(f)
        
        if not raw_csvs:
            logger.warning(f"[WRR Auto] No raw CSV files in {rr_dir}")
            return []
        
        # Step 1: CSV 헤더에서 실제 서버 이름 자동 감지 후 GPU 타입별 avg total 집계
        # CSV의 서버 이름(RTX3090_SERVER 등)과 현재 config의 서버 이름(RTX3090_A 등)이
        # 다를 수 있으므로, CSV 헤더에서 _running 컬럼을 직접 탐색하여 GPU 타입으로 매핑
        first_csv = raw_csvs[0]
        try:
            with open(first_csv, 'r') as f:
                csv_headers = csv.DictReader(f).fieldnames or []
        except Exception as e:
            logger.warning(f"[WRR Auto] Error reading headers from {first_csv.name}: {e}")
            return []
        
        csv_server_names = []
        for h in csv_headers:
            if h.endswith('_running'):
                csv_server_names.append(h.replace('_running', ''))
        
        if not csv_server_names:
            logger.warning(f"[WRR Auto] No _running columns found in CSV headers")
            return []
        
        csv_server_to_gpu_type = {}
        for csv_name in csv_server_names:
            for gpu_type in self.gpu_types:
                if csv_name.upper().startswith(gpu_type.upper()):
                    csv_server_to_gpu_type[csv_name] = gpu_type
                    break
        
        logger.info(f"[WRR Auto] CSV server → GPU type mapping: {csv_server_to_gpu_type}")
        
        per_run_gpu_totals = []  # [{gpu_type: [total1, total2, ...]}, ...]
        
        for csv_path in raw_csvs:
            totals = {name: 0.0 for name in csv_server_names}
            row_count = 0
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_count += 1
                        for name in csv_server_names:
                            running = float(row.get(f'{name}_running', 0) or 0)
                            waiting = float(row.get(f'{name}_waiting', 0) or 0)
                            totals[name] += running + waiting
                
                if row_count > 0:
                    avg_totals = {name: totals[name] / row_count for name in csv_server_names}
                    per_run_gpu_totals.append(avg_totals)
            except Exception as e:
                logger.warning(f"[WRR Auto] Error reading {csv_path.name}: {e}")
                continue
        
        if not per_run_gpu_totals:
            logger.warning(f"[WRR Auto] Failed to parse any CSV files")
            return []
        
        # CSV 서버별 전체 실험 평균
        csv_server_avg = {}
        for name in csv_server_names:
            total = sum(run[name] for run in per_run_gpu_totals)
            csv_server_avg[name] = total / len(per_run_gpu_totals)
        
        logger.info(f"[WRR Auto] RR avg total (running+waiting) from {len(per_run_gpu_totals)} runs (QPS {qps}):")
        for name, avg in csv_server_avg.items():
            gpu_type = csv_server_to_gpu_type.get(name, '?')
            logger.info(f"[WRR Auto]   {name} ({gpu_type}): avg total = {avg:.2f}")
        
        # Step 2: GPU 종류(풀)별로 avg total 평균 → 풀 대표값
        pool_avg_totals = {}
        for gpu_type in self.gpu_types:
            matching_avgs = [csv_server_avg[n] for n, gt in csv_server_to_gpu_type.items() if gt == gpu_type]
            if matching_avgs:
                pool_avg_totals[gpu_type] = sum(matching_avgs) / len(matching_avgs)
            else:
                pool_avg_totals[gpu_type] = 0.0
        
        logger.info(f"[WRR Auto] Pool avg totals:")
        for gpu_type, avg in pool_avg_totals.items():
            pool_size = len(self.gpu_pools.get(gpu_type, []))
            logger.info(f"[WRR Auto]   {gpu_type} ({pool_size}대): pool avg total = {avg:.2f}")
        
        # Step 3: 풀 대표값의 역수 비율로 weight 계산
        max_pool_total = max(pool_avg_totals.values())
        if max_pool_total <= 0:
            logger.warning(f"[WRR Auto] Max pool total is 0, cannot calculate ratios")
            return []
        
        pool_inverse = {}
        for gpu_type, avg in pool_avg_totals.items():
            if avg > 0:
                pool_inverse[gpu_type] = 1.0 / avg
            else:
                pool_inverse[gpu_type] = 1.0 / 0.1  # total 0이면 매우 높은 처리 능력으로 간주
        
        min_inverse = min(pool_inverse.values())
        
        # 풀별 서버당 weight (base 풀 = 1, 나머지는 비율)
        pool_per_server_weight = {}
        for gpu_type in self.gpu_types:
            ratio = pool_inverse.get(gpu_type, 1.0) / min_inverse
            pool_per_server_weight[gpu_type] = max(1, round(ratio))
        
        # Step 4: 각 물리 서버에 자신의 풀 weight 부여
        weights = []
        for server in self.backend_servers:
            gpu_type = server.get('gpu_type', server['name'])
            w = pool_per_server_weight.get(gpu_type, 1)
            weights.append(w)
        
        # 로그 출력
        logger.info(f"[WRR Auto] Pool-based weights:")
        for gpu_type in self.gpu_types:
            pool_size = len(self.gpu_pools.get(gpu_type, []))
            per_server_w = pool_per_server_weight.get(gpu_type, 1)
            pool_total_w = per_server_w * pool_size
            logger.info(f"[WRR Auto]   {gpu_type}: 서버당 weight={per_server_w}, "
                        f"풀 총 weight={pool_total_w} ({pool_size}대 × {per_server_w})")
        logger.info(f"[WRR Auto] Final weights (server order): {weights}")
        return weights

    def _aggregate_pool_states(self, all_metrics: Dict) -> Dict[str, Dict]:
        """
        GPU 종류별로 메트릭을 집계하여 대표값 생성 (SLM/Fisher-Jenks 프롬프트용)
        
        - running/waiting: 같은 종류 GPU들의 평균
        - inflight_tokens: 같은 종류 GPU들의 리스트를 모두 합침 (merge)
          → 풀 전체의 토큰 분포를 보존하여 더 정확한 통계 산출
        
        Returns:
            {gpu_type: {'running': avg, 'waiting': avg, 'inflight_tokens': merged_list}}
        """
        pool_states = {}
        for gpu_type in self.gpu_types:
            servers = self.gpu_pools.get(gpu_type, [])
            if not servers:
                pool_states[gpu_type] = {'running': 0, 'waiting': 0, 'inflight_tokens': []}
                continue
            
            running_list = []
            waiting_list = []
            merged_tokens = []
            
            for server in servers:
                server_name = server['name']
                metrics = all_metrics.get(server_name, {})
                
                running = metrics.get('num_requests_running', 0)
                waiting = metrics.get('num_requests_waiting', 0)
                
                # -1은 수집 실패 → 제외
                if running >= 0:
                    running_list.append(running)
                if waiting >= 0:
                    waiting_list.append(waiting)
                
                # inflight tokens 합치기 (풀 전체 분포 보존)
                inflight = list(self.inflight_tokens.get(server_name, []))
                merged_tokens.extend(inflight)
            
            avg_running = sum(running_list) / len(running_list) if running_list else 0
            avg_waiting = sum(waiting_list) / len(waiting_list) if waiting_list else 0
            
            pool_states[gpu_type] = {
                'running': avg_running,
                'waiting': avg_waiting,
                'inflight_tokens': merged_tokens
            }
        
        return pool_states

    def _select_best_server_in_pool(self, gpu_type: str, all_metrics: Optional[Dict]) -> Dict:
        """
        지정된 GPU 종류의 풀에서 total requests(running+waiting)가 가장 적은 서버 선택
        
        Args:
            gpu_type: GPU 종류명 (e.g., "RTX3090")
            all_metrics: 모든 서버의 메트릭 정보
        
        Returns:
            선택된 서버 dict
        """
        pool = self.gpu_pools.get(gpu_type, [])
        if not pool:
            logger.warning(f"No servers in pool for GPU type: {gpu_type}")
            return self.backend_servers[0]  # fallback
        
        if len(pool) == 1:
            return pool[0]
        
        if not all_metrics:
            return random.choice(pool)
        
        best_server = None
        min_total = float('inf')
        
        for server in pool:
            server_name = server['name']
            metrics = all_metrics.get(server_name, {})
            running = metrics.get('num_requests_running', 0)
            waiting = metrics.get('num_requests_waiting', 0)
            
            if running < 0 or waiting < 0:
                total = float('inf')
            else:
                total = running + waiting
            
            if total < min_total:
                min_total = total
                best_server = server
        
        if best_server:
            if len(pool) > 1:
                logger.debug(f"[Pool] {gpu_type}: selected {best_server['name']} (total={min_total}) "
                            f"from {len(pool)} servers")
            return best_server
        return random.choice(pool)

    async def increment_active_requests(self):
        """활성 요청 수 증가"""
        async with self.active_requests_lock:
            self.active_requests += 1
            logger.debug(f"Active requests: {self.active_requests} (↑)")
    
    async def decrement_active_requests(self):
        """활성 요청 수 감소"""
        async with self.active_requests_lock:
            self.active_requests -= 1
            logger.debug(f"Active requests: {self.active_requests} (↓)")
            
            # 종료 대기 중이고 모든 요청이 완료되었으면 이벤트 설정
            if self.shutting_down and self.active_requests == 0:
                logger.info("All active requests completed. Ready for shutdown.")
                self.shutdown_event.set()
    
    async def wait_for_all_requests(self, timeout: float = 300.0):
        """
        모든 활성 요청이 완료될 때까지 대기
        
        Args:
            timeout: 최대 대기 시간 (초)
        
        Returns:
            bool: 모든 요청이 완료되면 True, 타임아웃이면 False
        """
        try:
            async with asyncio.timeout(timeout):
                while self.active_requests > 0:
                    logger.info(f"Waiting for {self.active_requests} active requests to complete...")
                    await asyncio.sleep(1)
                return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for requests to complete. {self.active_requests} requests still active.")
            return False
    
    def _diag_stats(self, data: deque) -> Dict:
        """deque의 통계를 계산하여 dict로 반환"""
        if not data:
            return {'count': 0, 'avg': 0, 'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'max': 0, 'min': 0}
        sorted_d = sorted(data)
        n = len(sorted_d)
        return {
            'count': n,
            'avg': round(sum(sorted_d) / n, 2),
            'p50': round(_percentile(sorted_d, 50), 2),
            'p90': round(_percentile(sorted_d, 90), 2),
            'p95': round(_percentile(sorted_d, 95), 2),
            'p99': round(_percentile(sorted_d, 99), 2),
            'max': round(sorted_d[-1], 2),
            'min': round(sorted_d[0], 2),
        }
    
    def _log_diag_summary(self):
        """
        주기적 진단 요약을 로그로 출력 + 파일로 자동 저장.
        로그는 에이전트에게 복사해서 보여줄 용도,
        파일은 강제 종료 시에도 남아있도록 하는 안전장치.
        """
        d = self._diag
        total = d['total_diagnosed']
        elapsed = time.time() - d['diag_start_time']
        throughput = total / elapsed if elapsed > 0 else 0
        
        slm_device = getattr(self, '_slm_device', 'N/A')
        batch_enabled = getattr(self, 'slm_batch_enabled', 'N/A')
        
        lines = [
            f"\n{'='*80}",
            f"[DIAG SUMMARY] 최근 {total}개 요청 진단 (경과: {elapsed:.0f}초, 처리량: {throughput:.1f} req/s)",
            f"  SLM device: {slm_device} | Batch: {batch_enabled}",
            f"{'='*80}",
        ]
        
        stage_names = [
            ('handler_json_parse_ms',  'Handler: JSON 파싱'),
            ('handler_routing_ms',     'Handler: 라우팅 결정 (get_next_server)'),
            ('handler_total_ms',       'Handler: 전체 프록시 오버헤드 (arrival→forward)'),
            ('slm_metrics_ms',         '  SLM: 메트릭 수집 (캐시)'),
            ('slm_prompt_gen_ms',      '  SLM: Pool 집계 + 프롬프트 생성'),
            ('slm_inference_ms',       '  SLM: 모델 추론 (토크나이징+포워드)'),
            ('slm_total_ms',           '  SLM: _predict_with_slm 전체'),
            ('batch_queue_wait_ms',    '  Batch: 큐 대기+추론 (submit→result)'),
            ('batch_queue_depth',      '  Batch: 제출 시 큐 깊이'),
        ]
        
        for key, label in stage_names:
            data = d.get(key, deque())
            if not data:
                continue
            st = self._diag_stats(data)
            if key == 'batch_queue_depth':
                lines.append(f"  {label:50s} | avg={st['avg']:.1f}  p50={st['p50']:.0f}  p90={st['p90']:.0f}  p99={st['p99']:.0f}  max={st['max']:.0f}")
            else:
                lines.append(f"  {label:50s} | avg={st['avg']:.1f}ms  p50={st['p50']:.1f}ms  p90={st['p90']:.1f}ms  p99={st['p99']:.1f}ms  max={st['max']:.1f}ms")
        
        # 배치 추론기 통계
        if hasattr(self, 'slm_batch_inferencer') and self.slm_batch_inferencer._stats['total_batches'] > 0:
            bs = self.slm_batch_inferencer._stats
            avg_bs = bs['total_items'] / bs['total_batches'] if bs['total_batches'] > 0 else 0
            lines.append(f"  {'Batch: 통계':50s} | total_batches={bs['total_batches']}  avg_size={avg_bs:.1f}  max_size={bs['max_batch_seen']}")
        
        lines.append(f"{'='*80}")
        
        logger.info('\n'.join(lines))
        
        # 파일 자동 저장 (강제 종료 시에도 최신 진단 데이터가 남도록)
        try:
            live_path = Path(self.metrics_csv_path).parent / "_live_diagnostics.csv"
            self.save_diagnostics_csv(str(live_path))
        except Exception as e:
            logger.debug(f"[DIAG] Live diagnostics save failed: {e}")
    
    def get_diagnostics(self) -> Dict:
        """진단 데이터를 JSON 직렬화 가능한 dict로 반환 (/diagnostics 엔드포인트용)"""
        d = self._diag
        total = d['total_diagnosed']
        elapsed = time.time() - d['diag_start_time']
        
        result = {
            'total_requests_diagnosed': total,
            'elapsed_seconds': round(elapsed, 1),
            'throughput_rps': round(total / elapsed, 2) if elapsed > 0 else 0,
            'stages': {},
        }
        
        stage_keys = [
            'handler_json_parse_ms', 'handler_routing_ms', 'handler_total_ms',
            'slm_metrics_ms', 'slm_prompt_gen_ms', 'slm_inference_ms', 'slm_total_ms',
            'batch_queue_wait_ms', 'batch_queue_depth',
        ]
        
        for key in stage_keys:
            data = d.get(key, deque())
            if data:
                result['stages'][key] = self._diag_stats(data)
        
        # 배치 추론기 통계
        if hasattr(self, 'slm_batch_inferencer'):
            result['batch_inferencer'] = self.slm_batch_inferencer.get_stats()
        
        return result
    
    def save_diagnostics_csv(self, csv_path: str):
        """
        진단 데이터를 CSV 파일로 저장
        
        구조: 2개 섹션
        1) 단계별 타이밍 통계 (Avg, P50, P90, P95, P99, Max, Min)
        2) 메타 정보 (처리량, 배치 추론기 통계 등)
        """
        import csv as csv_mod
        
        d = self._diag
        total = d['total_diagnosed']
        elapsed = time.time() - d['diag_start_time']
        throughput = total / elapsed if elapsed > 0 else 0
        
        stage_labels = [
            ('handler_json_parse_ms',  'Handler: JSON Parse'),
            ('handler_routing_ms',     'Handler: Routing (get_next_server)'),
            ('handler_total_ms',       'Handler: Total Proxy Overhead (arrival→forward)'),
            ('slm_metrics_ms',         'SLM: Metrics Collection (cache)'),
            ('slm_prompt_gen_ms',      'SLM: Pool Aggregation + Prompt Generation'),
            ('slm_inference_ms',       'SLM: Model Inference (tokenize+forward)'),
            ('slm_total_ms',           'SLM: _predict_with_slm Total'),
            ('batch_queue_wait_ms',    'Batch: Queue Wait+Inference (submit→result)'),
            ('batch_queue_depth',      'Batch: Queue Depth at Submit'),
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv_mod.writer(f)
            
            # 메타 정보
            writer.writerow(['=== Diagnostics Summary ===', ''])
            writer.writerow(['Total Requests Diagnosed', total])
            writer.writerow(['Elapsed Seconds', round(elapsed, 1)])
            writer.writerow(['Throughput (req/s)', round(throughput, 2)])
            writer.writerow(['Batch Inference Enabled', getattr(self, 'slm_batch_enabled', 'N/A')])
            writer.writerow(['Routing Algorithm', self.routing_algorithm])
            writer.writerow(['SLM Active', getattr(self, 'slm_active', 'N/A')])
            writer.writerow(['SLM Device', getattr(self, '_slm_device', 'N/A')])
            writer.writerow(['SLM Model Path', getattr(self, 'slm_model_path', 'N/A')])
            writer.writerow(['', ''])
            
            # 배치 추론기 통계
            if hasattr(self, 'slm_batch_inferencer') and self.slm_batch_inferencer._stats.get('total_batches', 0) > 0:
                bs = self.slm_batch_inferencer._stats
                avg_bs = bs['total_items'] / bs['total_batches'] if bs['total_batches'] > 0 else 0
                writer.writerow(['=== Batch Inferencer Stats ===', ''])
                writer.writerow(['Total Batches', bs['total_batches']])
                writer.writerow(['Total Items', bs['total_items']])
                writer.writerow(['Avg Batch Size', round(avg_bs, 2)])
                writer.writerow(['Max Batch Size', bs['max_batch_seen']])
                writer.writerow(['', ''])
            
            # 단계별 타이밍 통계 테이블
            writer.writerow(['=== Stage Timing Statistics ===', ''])
            unit_label = 'ms'
            writer.writerow(['Stage', 'Unit', 'Count', 'Avg', 'P50', 'P90', 'P95', 'P99', 'Max', 'Min'])
            
            for key, label in stage_labels:
                data = d.get(key, deque())
                if not data:
                    continue
                st = self._diag_stats(data)
                unit = '' if key == 'batch_queue_depth' else 'ms'
                writer.writerow([
                    label, unit,
                    st['count'], st['avg'], st['p50'], st['p90'], st['p95'], st['p99'], st['max'], st['min']
                ])
            
            # Raw 데이터 (최근 500개 샘플) — 시계열 분석용
            writer.writerow(['', ''])
            writer.writerow(['=== Raw Samples (recent, per-request) ===', ''])
            
            # 모든 단계의 raw 데이터를 열로 나열
            active_keys = [(k, l) for k, l in stage_labels if d.get(k)]
            if active_keys:
                header = ['sample_index'] + [label for _, label in active_keys]
                writer.writerow(header)
                
                max_len = max(len(d[k]) for k, _ in active_keys)
                lists = {k: list(d[k]) for k, _ in active_keys}
                
                for i in range(max_len):
                    row = [i]
                    for k, _ in active_keys:
                        vals = lists[k]
                        row.append(round(vals[i], 3) if i < len(vals) else '')
                    writer.writerow(row)
        
        logger.info(f"[DIAG] Diagnostics saved to: {csv_path}")
    
    def reset_diagnostics(self):
        """진단 데이터 초기화 (다음 실험용)"""
        for key, val in self._diag.items():
            if isinstance(val, deque):
                val.clear()
        self._diag['total_diagnosed'] = 0
        self._diag['diag_start_time'] = time.time()
        self._diag['last_summary_request_id'] = 0
    
    def emergency_save(self):
        """
        서버 강제 종료(Ctrl+C / SIGTERM) 시 현재까지 기록된 결과를 저장.
        finalize 요청 없이 중단되는 경우를 위한 안전장치.
        동기 함수로 구현 (shutdown 시점에서 이벤트 루프가 불안정할 수 있으므로).
        """
        import csv as csv_mod
        
        source_csv = Path(self.metrics_csv_path)
        if not source_csv.exists():
            logger.warning("[EMERGENCY-SAVE] No metrics CSV file found. Nothing to save.")
            return
        
        # 레코드가 있는지 확인
        try:
            with open(source_csv, 'r') as f:
                reader = csv_mod.reader(f)
                row_count = sum(1 for _ in reader) - 1
            if row_count <= 0:
                logger.info("[EMERGENCY-SAVE] Metrics CSV is empty (header only). Skipping save.")
                return
        except Exception:
            return
        
        logger.info(f"[EMERGENCY-SAVE] Saving {row_count} records before shutdown...")
        
        # 알고리즘 이름 및 suffix 구성 (finalize와 동일 로직)
        algorithm_name = self.routing_algorithm
        threshold_suffix = ""
        
        if algorithm_name == "slm_adaptive":
            act = getattr(self, 'slm_activation_threshold', 15.0)
            deact = getattr(self, 'slm_deactivation_threshold', 10.0)
            threshold_suffix = f"_act{act}_deact{deact}"
        elif algorithm_name == "fisher_jenks_sqf":
            act = getattr(self, 'slm_activation_threshold', 15.0)
            deact = getattr(self, 'slm_deactivation_threshold', 10.0)
            method = getattr(self, 'fj_candidate_method', 'fisher_jenks')
            
            if method == "fisher_jenks":
                fjw = getattr(self, 'fj_window_size', 50)
                fjm = getattr(self, 'fj_min_samples', 15)
                fjd = getattr(self, 'fj_default_threshold', 0.0)
                threshold_suffix = f"_act{act}_deact{deact}_fjw{fjw}_fjm{fjm}_fjd{fjd}"
            elif method == "percent_of_min":
                pct = getattr(self, 'fj_percent_threshold', 0.09)
                threshold_suffix = f"_act{act}_deact{deact}_pctmin{pct}"
            elif method == "top_n":
                n = getattr(self, 'fj_top_n', 2)
                threshold_suffix = f"_act{act}_deact{deact}_topn{n}"
            elif method == "ewma":
                alpha = getattr(self, 'fj_ewma_alpha', 0.2)
                mult = getattr(self, 'fj_ewma_multiplier', 1.4)
                threshold_suffix = f"_act{act}_deact{deact}_ewma{alpha}_m{mult}"
            elif method == "fixed_threshold":
                ft = getattr(self, 'fj_fixed_threshold', 10.8)
                threshold_suffix = f"_act{act}_deact{deact}_fixed{ft}"
            else:
                threshold_suffix = f"_act{act}_deact{deact}_{method}"
        elif algorithm_name == "weighted_round_robin":
            weights = getattr(self, 'routing_weights', None)
            if weights:
                weight_str = "_".join(str(int(w)) for w in weights)
                threshold_suffix = f"_w{weight_str}"
        
        # 디렉토리 구성
        num_gpus = len(self.backend_servers)
        if algorithm_name == "fisher_jenks_sqf":
            base_results_dir = Path(__file__).parent / "fj_ablation_study"
        else:
            base_results_dir = Path(__file__).parent / f"regressor_results_{num_gpus}gpu"
        dataset = getattr(self, 'current_dataset', None)
        if dataset:
            base_results_dir = base_results_dir / f"used_{dataset}"
        
        req_count = self.request_count
        req_suffix = f"_req{req_count}" if req_count > 0 else ""
        
        qps = getattr(self, 'current_qps', None)
        
        # "INTERRUPTED" 태그를 붙여 finalize된 것과 구분
        if qps:
            results_dir = base_results_dir / f"QPS{qps}_{algorithm_name}{threshold_suffix}"
            base_filename = f"results_QPS{qps}_{algorithm_name}{threshold_suffix}{req_suffix}_INTERRUPTED"
        else:
            results_dir = base_results_dir / f"{algorithm_name}{threshold_suffix}"
            base_filename = f"results_{algorithm_name}{threshold_suffix}{req_suffix}_INTERRUPTED"
        
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 충돌 방지
        extension = source_csv.suffix
        dest_csv = results_dir / f"{base_filename}{extension}"
        counter = 2
        while dest_csv.exists():
            dest_csv = results_dir / f"{base_filename}_{counter}{extension}"
            counter += 1
        
        if counter > 2:
            base_filename = f"{base_filename}_{counter - 1}"
        
        try:
            # 메트릭 CSV 복사
            shutil.copy2(source_csv, dest_csv)
            logger.info(f"[EMERGENCY-SAVE] Metrics saved: {dest_csv}")
            
            # Summary 생성
            summary_path = results_dir / f"{base_filename}_summary.csv"
            try:
                generate_summary_csv(str(dest_csv), str(summary_path), self.backend_servers, self.gpu_types)
                logger.info(f"[EMERGENCY-SAVE] Summary saved: {summary_path}")
            except Exception as e:
                logger.error(f"[EMERGENCY-SAVE] Summary generation failed: {e}")
            
            # Diagnostics 저장
            diag_path = results_dir / f"{base_filename}_diagnostics.csv"
            try:
                self._log_diag_summary()
                self.save_diagnostics_csv(str(diag_path))
                logger.info(f"[EMERGENCY-SAVE] Diagnostics saved: {diag_path}")
            except Exception as e:
                logger.error(f"[EMERGENCY-SAVE] Diagnostics save failed: {e}")
            
            logger.info(f"[EMERGENCY-SAVE] All files saved to: {results_dir}")
            
        except Exception as e:
            logger.error(f"[EMERGENCY-SAVE] Failed to save: {e}")
    
    async def close(self):
        """리소스 정리"""
        # SLM 배치 추론기 종료
        if hasattr(self, 'slm_batch_inferencer') and self.slm_batch_inferencer is not None:
            await self.slm_batch_inferencer.stop()
        
        # SLM 전용 ThreadPoolExecutor 종료
        if hasattr(self, '_slm_executor') and self._slm_executor is not None:
            self._slm_executor.shutdown(wait=False)
            self._slm_executor = None
        
        if self._metrics_cache_task is not None:
            self._metrics_cache_task.cancel()
            try:
                await self._metrics_cache_task
            except asyncio.CancelledError:
                pass
            self._metrics_cache_task = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._metrics_http_client is not None:
            await self._metrics_http_client.aclose()
            self._metrics_http_client = None
    
    def _init_csv_file(self):
        """CSV 파일 초기화 — 기존 데이터가 있으면 자동 백업 후 새로 생성"""
        csv_path = Path(self.metrics_csv_path)
        if csv_path.exists():
            try:
                with open(csv_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                if line_count > 1:
                    backup_path = csv_path.with_suffix(f".bak_{int(time.time())}.csv")
                    shutil.copy2(csv_path, backup_path)
                    logger.warning(
                        f"Existing CSV ({line_count - 1} records) backed up to: {backup_path}"
                    )
                else:
                    logger.info(f"Existing CSV is empty (header only), overwriting.")
            except Exception as e:
                logger.warning(f"Could not backup existing CSV: {e}")
        self._create_csv_with_header(csv_path)
        logger.info(f"Initialized metrics CSV file: {self.metrics_csv_path}")
    
    def _create_csv_with_header(self, csv_path: Path):
        """CSV 파일을 헤더와 함께 생성
        
        ⚠️ 중요: SLM adaptive의 경우 항상 SLM 필드를 포함해야 함
        (routing_algorithm 설정과 무관하게 CSV 구조가 일관되어야 함)
        """
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # CSV 헤더
            header = [
                'request_id',
                'timestamp',
                'target_server',
                'target_server_port',
            ]
            # 각 서버별 메트릭 컬럼 추가
            for server in self.backend_servers:
                header.extend([
                    f"{server['name']}_running",
                    f"{server['name']}_waiting"
                ])
            # 레이턴시 메트릭 추가
            header.extend([
                'ttft_seconds',          # Time To First Token (초)
                'e2e_latency_seconds',   # End-to-End Latency (초)
                'arrival_timestamp',          # 핸들러 도착 시각 (time.time() → ISO format)
                'proxy_overhead_seconds',     # 프록시 오버헤드: 핸들러 도착 → 백엔드 HTTP 요청 전송 직전 (초)
                'slm_inference_seconds',      # SLM 추론 시간 (SLM 알고리즘만, 나머지는 None)
            ])
            # SLM Adaptive Routing 분석용 필드는 항상 추가
            # (CSV 파일 생성 후 routing_algorithm이 변경될 수 있으므로)
            header.extend([
                'slm_mode_active',           # SLM 모드 활성화 여부 (True/False)
                'slm_prediction_used',       # SLM 예측 사용 여부 (True/False)
                'slm_latency_diff',           # SLM 예측 latency diff (algo 4) 또는 Fisher-Jenks split point (algo 5)
                'slm_predicted_selected_server_latency',  # 선택된 서버의 예측 latency (MAE 계산용)
            ])
            # 각 GPU 종류별 예측 레이턴시 동적 추가 (SLM은 GPU 종류별로 예측)
            for gpu_type in self.gpu_types:
                header.append(f"slm_predicted_{gpu_type.lower()}")
            header.append('recent_avg_e2e')  # 최근 평균 E2E latency (모드 전환 기준)
            writer.writerow(header)
    
    def reset_csv_file(self):
        """CSV 파일을 초기화 (헤더만 남기고 데이터 삭제)"""
        csv_path = Path(self.metrics_csv_path)
        self._create_csv_with_header(csv_path)
        logger.info(f"Reset metrics CSV file: {self.metrics_csv_path}")
    
    async def _fetch_metrics(self, server: Dict, quiet: bool = False) -> Optional[Dict]:
        """
        서버의 /metrics?format=json 엔드포인트에서 스케줄링 메트릭 수집
        ⭐ 공유 HTTP 클라이언트 사용 (매번 생성하지 않음)
        
        Args:
            quiet: True이면 에러 로그를 debug 레벨로 출력 (백그라운드 캐시용)
        """
        if not METRICS_CONFIG["enabled"]:
            return None
        
        _log = logger.debug if quiet else logger.warning
        metrics_url = (
            f"http://{server['host']}:{server['port']}/metrics?format=json")
        
        try:
            client = await self.get_metrics_http_client()
            response = await client.get(metrics_url)
            
            if response.status_code != 200:
                _log(f"Failed to fetch metrics from {server['name']}: {response.status_code}")
                return None
            
            data = response.json()
            if data.get('status') != 'success':
                raise ValueError(
                    f"metrics endpoint returned status={data.get('status')!r}")

            required_fields = (
                'engine_running_requests',
                'engine_waiting_requests',
                'inflight_prompt_token_lengths',
            )
            missing_fields = [
                field for field in required_fields if field not in data
            ]
            if missing_fields:
                raise ValueError(
                    f"metrics response missing fields: {', '.join(missing_fields)}")

            inflight_tokens = data['inflight_prompt_token_lengths']
            if not isinstance(inflight_tokens, list):
                raise ValueError(
                    "inflight_prompt_token_lengths must be a list")

            return {
                'num_requests_running': int(float(
                    data['engine_running_requests'])),
                'num_requests_waiting': int(float(
                    data['engine_waiting_requests'])),
                'inflight_prompt_token_lengths': inflight_tokens,
            }
                
        except Exception as e:
            _log(f"Error fetching metrics from {server['name']}: {str(e)}")
            return None
    
    async def _collect_all_metrics_direct(self, background: bool = False) -> Dict[str, Dict]:
        """
        모든 백엔드 서버의 메트릭을 비동기로 직접 수집 (HTTP 요청 발생)
        백그라운드 캐시 태스크 및 SLM/SQF의 실시간 수집에 사용
        
        Args:
            background: True이면 백그라운드 캐시에서 호출 (에러 로그를 debug로 출력)
        """
        # 실시간 수집도 반복 실패 시 로그 억제 (초당 168회 요청 시 로그 폭주 방지)
        _log = logger.debug
        
        # Prometheus 메트릭 수집
        tasks = []
        for server in self.backend_servers:
            tasks.append(self._fetch_metrics(server, quiet=True))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_metrics = {}
        fail_count = 0
        fail_servers = []
        for server, result in zip(self.backend_servers, results):
            if isinstance(result, Exception):
                _log(f"[METRICS] Exception collecting metrics from {server['name']}: {result}")
                all_metrics[server['name']] = {
                    'num_requests_running': -1,
                    'num_requests_waiting': -1
                }
                fail_count += 1
                fail_servers.append(server['name'])
            elif result is None:
                _log(f"[METRICS] Failed to collect metrics from {server['name']}")
                all_metrics[server['name']] = {
                    'num_requests_running': -1,
                    'num_requests_waiting': -1
                }
                fail_count += 1
                fail_servers.append(server['name'])
            else:
                inflight_tokens = result.pop(
                    'inflight_prompt_token_lengths', [])
                all_metrics[server['name']] = result
                if (self.routing_algorithm in
                        ("slm_adaptive", "fisher_jenks_sqf")
                        and server['name'] in self.inflight_tokens):
                    self.inflight_tokens[server['name']].clear()
                    self.inflight_tokens[server['name']].extend(
                        inflight_tokens)
        
        # 실패 요약 로그 (100회마다 1회만 출력하여 로그 폭주 방지)
        if fail_count > 0:
            if not hasattr(self, '_metrics_fail_log_count'):
                self._metrics_fail_log_count = 0
            self._metrics_fail_log_count += 1
            if self._metrics_fail_log_count <= 3 or self._metrics_fail_log_count % 100 == 0:
                logger.warning(f"[METRICS] {fail_count}/{len(self.backend_servers)} servers failed "
                             f"(count={self._metrics_fail_log_count}): {', '.join(fail_servers)}")
        
        return all_metrics
    
    async def _collect_all_metrics(self) -> Dict[str, Dict]:
        """
        메트릭 수집 (요청 경로에서 호출).
        
        ⭐ 항상 캐시된 데이터를 사용하여 즉시 반환 (HTTP 요청 없음)
        ⭐ 캐시가 오래되었어도 stale 데이터를 반환 (직접 수집 절대 안함)
        ⭐ 서버 시작 직후 캐시가 비어있을 때만 1회 직접 수집
        """
        cached = self.get_cached_metrics(allow_stale=True)
        if cached is not None:
            return cached
        
        # 최초 1회만 직접 수집 (서버 시작 직후, 캐시가 아예 없을 때)
        return await self._collect_all_metrics_direct()
    
    async def _log_metrics_to_csv(
        self, 
        request_id: int, 
        target_server: Dict, 
        all_metrics: Dict[str, Dict],
        ttft_seconds: Optional[float] = None,
        e2e_latency_seconds: Optional[float] = None,
        slm_info: Optional[Dict] = None,  # ⭐ SLM 관련 정보
        arrival_time: Optional[float] = None,  # ⭐ 핸들러 도착 시각 (time.time())
        proxy_overhead_seconds: Optional[float] = None,  # ⭐ 프록시 오버헤드 (arrival → 백엔드 요청 전송)
        slm_inference_seconds: Optional[float] = None  # ⭐ SLM 추론 시간
    ):
        """
        수집한 메트릭과 레이턴시를 CSV 파일에 저장
        
        Args:
            request_id: 요청 ID
            target_server: 대상 서버 정보
            all_metrics: 모든 서버의 메트릭
            ttft_seconds: Time To First Token (초)
            e2e_latency_seconds: End-to-End Latency (초)
            slm_info: SLM 관련 정보 (mode_active, prediction_used, latency_diff, recent_avg_e2e)
            arrival_time: 핸들러 도착 시각 (time.time() 값)
            proxy_overhead_seconds: 프록시 오버헤드 (핸들러 도착 → 백엔드 HTTP 요청 전송 직전)
            slm_inference_seconds: SLM 추론 시간 (초, SLM 알고리즘만 해당)
        """
        try:
            with open(self.metrics_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                row = [
                    request_id,
                    datetime.now().isoformat(),
                    target_server['name'],
                    target_server['port'],
                ]
                
                # 각 서버의 메트릭 추가
                for server in self.backend_servers:
                    server_metrics = all_metrics.get(server['name'], {
                        'num_requests_running': -1,
                        'num_requests_waiting': -1
                    })
                    row.extend([
                        server_metrics['num_requests_running'],
                        server_metrics['num_requests_waiting']
                    ])
                
                # 레이턴시 메트릭 추가
                row.extend([
                    round(ttft_seconds, 4) if ttft_seconds is not None else None,
                    round(e2e_latency_seconds, 4) if e2e_latency_seconds is not None else None
                ])
                
                # 도착 시각, 프록시 오버헤드, SLM 추론 시간 추가
                if arrival_time is not None:
                    arrival_dt = datetime.fromtimestamp(arrival_time)
                    row.append(arrival_dt.isoformat())
                else:
                    row.append(None)
                row.append(round(proxy_overhead_seconds, 4) if proxy_overhead_seconds is not None else None)
                row.append(round(slm_inference_seconds, 4) if slm_inference_seconds is not None else None)
                
                # SLM Adaptive Routing 정보 추가 (동적으로 서버 수에 맞춤)
                # 헤더와 데이터 컬럼 수를 일치시키기 위해 항상 추가
                if self.routing_algorithm in ("slm_adaptive", "fisher_jenks_sqf") and slm_info:
                    row.extend([
                        slm_info.get('mode_active', False),
                        slm_info.get('prediction_used', False),
                        round(slm_info.get('latency_diff'), 4) if slm_info.get('latency_diff') is not None else None,
                        round(slm_info.get('predicted_selected_server_latency'), 4) if slm_info.get('predicted_selected_server_latency') is not None else None,
                    ])
                    # 각 GPU 종류별 예측 레이턴시 동적 추가 (SLM은 GPU 종류별로 예측)
                    for gpu_type in self.gpu_types:
                        type_key = f"predicted_latency_{gpu_type.lower()}"
                        pred_val = slm_info.get(type_key)
                        row.append(round(pred_val, 4) if pred_val is not None else None)
                    row.append(round(slm_info.get('recent_avg_e2e'), 2) if slm_info.get('recent_avg_e2e') is not None else None)
                else:
                    # Non-SLM 알고리즘이거나 slm_info가 없는 경우 빈 값 추가
                    # 4개 고정 필드 (mode_active, prediction_used, latency_diff, predicted_selected_server_latency) + GPU 종류 수 + 1개 고정 필드 (recent_avg_e2e)
                    num_empty_fields = 4 + len(self.gpu_types) + 1
                    row.extend([None] * num_empty_fields)
                
                writer.writerow(row)
                
            logger.debug(f"Logged metrics for request {request_id} "
                        f"(TTFT: {ttft_seconds:.4f}s, E2E: {e2e_latency_seconds:.4f}s)" 
                        if ttft_seconds and e2e_latency_seconds else f"Logged metrics for request {request_id}")
            
        except Exception as e:
            logger.error(f"Error writing metrics to CSV: {str(e)}")
        
    async def get_next_server(self, request_data: Optional[Dict] = None) -> tuple[Dict, Optional[Dict], Optional[Dict]]:
        """
        설정된 라우팅 알고리즘에 따라 다음 서버 선택
        - round_robin: 균등 분배 (1:1)
        - weighted_round_robin: 가중치 기반 분배 (예: 1:18)
        - shortest_queue_first: 실시간 waiting 수가 가장 적은 서버 선택
        - slm_adaptive: 부하 상황에 따라 WRR ↔ SLM 동적 전환
        - fisher_jenks_sqf: SLM 예측 + Fisher-Jenks Natural Breaks로 후보 GPU 그룹 선별 + SQF
        
        Returns:
            tuple: (선택된 서버, SLM 정보, 메트릭 정보) - 메트릭은 라우팅 결정 시점의 서버 상태
        """
        if self.routing_algorithm == "fisher_jenks_sqf":
            server, slm_info, metrics = await self._select_server_fisher_jenks(request_data)
            return server, slm_info, metrics
        elif self.routing_algorithm == "slm_adaptive":
            server, slm_info, metrics = await self._select_server_slm_adaptive_with_prompt(request_data)
            return server, slm_info, metrics
        elif self.routing_algorithm == "shortest_queue_first":
            server, metrics = await self._select_shortest_queue_server()
            return server, None, metrics
        elif self.routing_algorithm == "weighted_round_robin":
            # Weighted Round Robin: 가중치 순환 리스트에서 선택
            server_idx = self.weighted_server_sequence[self.weighted_index]
            self.weighted_index = (self.weighted_index + 1) % len(self.weighted_server_sequence)
            all_metrics = await self._collect_all_metrics()
            return self.backend_servers[server_idx], None, all_metrics
        else:
            # Round Robin: 균등 분배
            server = self.backend_servers[self.current_server_index]
            self.current_server_index = (self.current_server_index + 1) % len(self.backend_servers)
            all_metrics = await self._collect_all_metrics()
            return server, None, all_metrics
    
    async def _select_server_slm_adaptive_with_prompt(self, request_data: Optional[Dict]) -> tuple[Dict, Dict, Optional[Dict]]:
        """
        SLM Adaptive with prompt data
        
        Returns:
            tuple: (선택된 서버, SLM 정보, 메트릭 정보)
        """
        # SLM 정보 초기화 (GPU 종류별 predicted_latency)
        slm_info = {
            'mode_active': self.slm_active,
            'prediction_used': False,
            'latency_diff': None,
            'recent_avg_e2e': None
        }
        
        # 모든 GPU 종류의 predicted_latency 필드 초기화
        for gpu_type in self.gpu_types:
            slm_info[f"predicted_latency_{gpu_type.lower()}"] = None
        
        # 최근 평균 E2E 계산
        if len(self.recent_e2e_latencies) > 0:
            slm_info['recent_avg_e2e'] = sum(self.recent_e2e_latencies) / len(self.recent_e2e_latencies)
        
        # 메트릭은 SLM 예측 시에만 수집 (나중에 수집)
        all_metrics = None
        
        try:
            # 모드 업데이트 (최근 E2E latency 기반)
            await self._update_slm_mode()
            slm_info['mode_active'] = self.slm_active  # 업데이트된 모드 반영
            
            # === Step 1: slm_active 체크 (부하 기반) ===
            if not self.slm_active or not request_data:
                # 부하가 낮거나 프롬프트 없음 → Base algorithm 사용
                if self.slm_base_algorithm == "weighted_round_robin":
                    server_idx = self.weighted_server_sequence[self.weighted_index]
                    self.weighted_index = (self.weighted_index + 1) % len(self.weighted_server_sequence)
                    return self.backend_servers[server_idx], slm_info, all_metrics
                elif self.slm_base_algorithm == "shortest_queue_first":
                    # SQF 사용 시 메트릭 수집
                    server, all_metrics = await self._select_shortest_queue_server()
                    return server, slm_info, all_metrics
                else:
                    server = self.backend_servers[self.current_server_index]
                    self.current_server_index = (self.current_server_index + 1) % len(self.backend_servers)
                    return server, slm_info, all_metrics
            
            # === Step 2: slm_active == True, 프롬프트 추출 ===
            prompt_text = ""
            messages = request_data.get("messages", [])
            if messages:
                # 마지막 user 메시지 사용
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        prompt_text = msg.get("content", "")
                        break
            
            # === SLM Adaptive: SLM 예측만 사용 ===
            prediction, all_metrics = await self._predict_with_slm(prompt_text)

            if prediction:
                selected_gpu_type = prediction['selected_gpu_type']
                selected_server = self._select_best_server_in_pool(selected_gpu_type, all_metrics)

                slm_info['prediction_used'] = True
                slm_info['latency_diff'] = prediction['latency_diff']
                slm_info['inference_time_ms'] = prediction.get('inference_time_ms')

                for key, val in prediction.items():
                    if key.startswith('predicted_latency_'):
                        slm_info[key] = val

                slm_info['predicted_selected_server_latency'] = prediction.get('predicted_selected_server_latency')

                self.slm_prediction_stats['total_predictions'] += 1

                stats_key = f"{selected_server['name']}_selected"
                if stats_key in self.slm_prediction_stats:
                    self.slm_prediction_stats[stats_key] += 1

                latency_strs = [f"{gt}: {prediction.get('predicted_latency_' + gt.lower(), 0):.2f}s" for gt in self.gpu_types]
                stats_parts = [f"{s['name']}={self.slm_prediction_stats.get(s['name'] + '_selected', 0)}"
                               for s in self.backend_servers]
                stats_str = ", ".join(stats_parts)

                logger.info(f"SLM[{self.request_count}]: Predicted type={selected_gpu_type} → {selected_server['name']} "
                           f"({', '.join(latency_strs)}) [Stats: {stats_str}]")
                return selected_server, slm_info, all_metrics
            
            # 예측 실패: fallback
            logger.warning("SLM prediction failed, using fallback")
            if self.slm_fallback == "weighted_round_robin":
                server_idx = self.weighted_server_sequence[self.weighted_index]
                self.weighted_index = (self.weighted_index + 1) % len(self.weighted_server_sequence)
                return self.backend_servers[server_idx], slm_info, all_metrics
            elif self.slm_fallback == "shortest_queue_first":
                server, all_metrics = await self._select_shortest_queue_server()
                return server, slm_info, all_metrics
            else:
                server = self.backend_servers[self.current_server_index]
                self.current_server_index = (self.current_server_index + 1) % len(self.backend_servers)
                return server, slm_info, all_metrics
        
        except Exception as e:
            logger.error(f"Error in SLM adaptive with prompt: {e}")
            # Fallback to WRR
            server_idx = self.weighted_server_sequence[self.weighted_index]
            self.weighted_index = (self.weighted_index + 1) % len(self.weighted_server_sequence)
            return self.backend_servers[server_idx], slm_info, all_metrics
    
    async def _select_shortest_queue_server(self) -> tuple[Dict, Optional[Dict]]:
        """
        Shortest Queue First: waiting 수가 가장 적은 서버 선택
        waiting 수가 같으면 랜덤 선택
        
        Returns:
            tuple: (선택된 서버, 메트릭 정보)
        """
        try:
            # 모든 서버의 메트릭 수집
            all_metrics = await self._collect_all_metrics()
            
            if not all_metrics:
                # 메트릭 수집 실패 시 fallback
                logger.warning("SQF: Failed to collect metrics, using fallback algorithm")
                server = await self._fallback_server_selection()
                return server, None
            
            # 각 서버의 큐 길이 계산
            server_queues = []
            for server in self.backend_servers:
                server_name = server['name']
                if server_name in all_metrics:
                    metrics = all_metrics[server_name]
                    
                    # 메트릭이 -1이면 수집 실패
                    if metrics.get('num_requests_running', -1) == -1 or metrics.get('num_requests_waiting', -1) == -1:
                        queue_length = float('inf')
                    elif self.sqf_metric == "waiting":
                        queue_length = metrics.get('num_requests_waiting', 0)
                    else:  # "total"
                        queue_length = metrics.get('num_requests_running', 0) + metrics.get('num_requests_waiting', 0)
                    
                    server_queues.append({
                        'server': server,
                        'queue_length': queue_length
                    })
                else:
                    # 메트릭이 없는 서버는 최대 큐 길이로 가정
                    server_queues.append({
                        'server': server,
                        'queue_length': float('inf')
                    })
            
            # 최소 큐 길이 찾기
            min_queue_length = min(sq['queue_length'] for sq in server_queues)
            
            # 최소 큐 길이를 가진 서버들 필터링
            shortest_servers = [
                sq['server'] for sq in server_queues 
                if sq['queue_length'] == min_queue_length
            ]
            
            # 동일한 큐 길이를 가진 서버가 여러 개면 랜덤 선택
            selected_server = random.choice(shortest_servers)
            
            # 디버그 정보 (처음 100개 요청만 출력)
            if self.request_count < 100:
                queue_info = ", ".join([f"{sq['server']['name']}={sq['queue_length']}" 
                                       for sq in server_queues])
                logger.info(f"SQF[{self.request_count}]: Queues=[{queue_info}], "
                           f"Selected={selected_server['name']}, "
                           f"Candidates={len(shortest_servers)}")
            
            return selected_server, all_metrics
            
        except Exception as e:
            logger.error(f"Error in shortest queue selection: {str(e)}")
            server = await self._fallback_server_selection()
            return server, None
    
    async def _fallback_server_selection(self) -> Dict:
        """메트릭 수집 실패 시 fallback 알고리즘 사용"""
        if self.sqf_fallback == "weighted_round_robin":
            server_idx = self.weighted_server_sequence[self.weighted_index]
            self.weighted_index = (self.weighted_index + 1) % len(self.weighted_server_sequence)
            selected = self.backend_servers[server_idx]
            if self.request_count < 100:
                logger.info(f"SQF[{self.request_count}]: Using WRR fallback → {selected['name']}")
            return selected
        else:  # round_robin (default)
            server = self.backend_servers[self.current_server_index]
            self.current_server_index = (self.current_server_index + 1) % len(self.backend_servers)
            if self.request_count < 100:
                logger.info(f"SQF[{self.request_count}]: Using RR fallback → {server['name']}")
            return server
    
    async def _load_slm_model(self):
        """SLM 모델 lazy loading"""
        if self.slm_model is not None:
            return
        
        async with self._slm_load_lock:
            # Double-check pattern
            if self.slm_model is not None:
                return
            
            try:
                logger.info(f"Loading SLM model from {self.slm_model_path}...")
                self.slm_tokenizer = AutoTokenizer.from_pretrained(self.slm_model_path)
                self.slm_model = AutoModelForSequenceClassification.from_pretrained(self.slm_model_path)
                self.slm_model.eval()
                
                # GPU 사용 가능하면 GPU로, 아니면 CPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.slm_model.to(device)
                self._slm_device = device  # 디바이스 캐싱 (매 추론마다 탐색하지 않음)
                logger.info(f"✅ SLM model loaded successfully (device: {device})")
                
                # 워밍업 추론 (첫 추론의 CUDA 커널 컴파일 지연 제거)
                try:
                    warmup_input = self.slm_tokenizer(
                        "warmup", max_length=32, padding=True,
                        truncation=True, return_tensors="pt"
                    )
                    warmup_input = {k: v.to(device) for k, v in warmup_input.items()}
                    with torch.inference_mode():
                        self.slm_model(**warmup_input)
                    logger.info(f"✅ SLM model warmup completed")
                except Exception as we:
                    logger.warning(f"SLM warmup failed (non-critical): {we}")
                
                # 배치 추론기 연결 (활성화된 경우에만 시작)
                self.slm_batch_inferencer.set_model(self.slm_model, self.slm_tokenizer, device)
                if self.slm_batch_enabled:
                    await self.slm_batch_inferencer.start()
                    logger.info(f"✅ SLM batch inferencer connected and started")
                else:
                    logger.info(f"✅ SLM model ready (individual inference via run_in_executor)")
            except Exception as e:
                logger.error(f"Failed to load SLM model: {e}")
                self.slm_model = None
                self.slm_tokenizer = None
    
    def _make_slm_prompt(self, prompt_text: str, pool_states: Dict[str, Dict]) -> str:
        """
        SLM 추론을 위한 프롬프트 생성 (논문 §IV.B 및 Eq.4).

        입력 구성:
            - 요청: 프롬프트 앞 150자 + 뒤 150자 + 전체 문자 수
            - 풀 단위 상태(Eq.4): 평균 running, 평균 waiting, |T_pool|,
              토큰 길이의 p99, p90, p75, p50, p25
        """

        def percentile(sorted_list, q):
            if not sorted_list:
                return 0
            n = len(sorted_list)
            idx = min(int(n * q), n - 1)
            return sorted_list[idx]

        type_descriptions = []
        for gpu_type in self.gpu_types:
            state = pool_states.get(gpu_type, {'running': 0, 'waiting': 0, 'inflight_tokens': []})
            tokens = sorted(state.get('inflight_tokens', []) or [])
            n_tokens = len(tokens)

            type_label = f"{gpu_type}_POOL"
            desc = (
                f"{type_label} status: running {state['running']:.0f}, waiting {state['waiting']:.0f}, "
                f"tokens {n_tokens} "
                f"[p99:{percentile(tokens, 0.99):.0f}, "
                f"p90:{percentile(tokens, 0.90):.0f}, "
                f"p75:{percentile(tokens, 0.75):.0f}, "
                f"p50:{percentile(tokens, 0.50):.0f}, "
                f"p25:{percentile(tokens, 0.25):.0f}]. "
            )
            type_descriptions.append(desc)

        first_chunk = prompt_text[:150].replace('\n', ' ')
        last_chunk = prompt_text[-150:].replace('\n', ' ')
        char_count = len(prompt_text)
        request_content = (
            f"Request: first150='{first_chunk}', last150='{last_chunk}', "
            f"total_chars={char_count}."
        )

        final_text = "".join(type_descriptions) + request_content
        return final_text
    
    async def _predict_with_slm(self, prompt_text: str) -> tuple[Optional[Dict], Optional[Dict]]:
        """
        SLM을 사용한 GPU 종류별 E2E latency 예측
        
        - GPU 종류별 메트릭 집계 (풀 대표값) → SLM 프롬프트 생성
        - SLM 모델은 GPU 종류별로 latency를 예측 (gpu_types 순서)
        - 가장 낮은 latency의 GPU 종류를 선택
        
        Returns:
            tuple: (예측 결과 dict, 전체 메트릭 dict)
            예측 결과 dict keys:
                - selected_gpu_type: 선택된 GPU 종류 (e.g., "RTX3090")
                - predicted_latency_{gpu_type}: GPU 종류별 예측 latency
                - latency_diff: 최대-최소 latency 차이
                - inference_time_ms: SLM 추론 시간
                - predicted_selected_server_latency: 선택된 종류의 예측 latency
        """
        _t_slm_start = time.perf_counter()
        
        try:
            # 모델 로드 (lazy loading)
            if self.slm_model is None:
                await self._load_slm_model()
            
            if self.slm_model is None:
                logger.warning("SLM model not available, using fallback")
                return None, None
            
            # [DIAG] 메트릭 수집 단계
            _t0 = time.perf_counter()
            all_metrics = await self._collect_all_metrics()
            _t_metrics = (time.perf_counter() - _t0) * 1000
            self._diag['slm_metrics_ms'].append(_t_metrics)
            
            if not all_metrics:
                logger.warning("Failed to collect metrics for SLM")
                return None, None
            
            # [DIAG] pool 집계 + 프롬프트 생성 단계
            _t0 = time.perf_counter()
            pool_states = self._aggregate_pool_states(all_metrics)
            slm_prompt = self._make_slm_prompt(prompt_text, pool_states)
            _t_prompt = (time.perf_counter() - _t0) * 1000
            self._diag['slm_prompt_gen_ms'].append(_t_prompt)
            
            # [DIAG] SLM 추론 단계
            _t0 = time.perf_counter()
            if self.slm_batch_enabled:
                logits, inference_time = await self.slm_batch_inferencer.predict(slm_prompt)
                # 배치 추론기 진단 데이터 수집
                self._diag['batch_queue_wait_ms'].append(
                    self.slm_batch_inferencer._stats.get('last_queue_wait_ms', 0))
                self._diag['batch_queue_depth'].append(
                    self.slm_batch_inferencer._stats.get('last_queue_depth', 0))
            else:
                model = self.slm_model
                tokenizer = self.slm_tokenizer
                device = self._slm_device
                
                def _slm_inference_sync():
                    full_start = time.perf_counter()
                    inputs = tokenizer(
                        slm_prompt,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        outputs = model(**inputs)
                        logits = outputs.logits
                    inference_time = (time.perf_counter() - full_start) * 1000
                    return logits, inference_time
                
                loop = asyncio.get_event_loop()
                logits, inference_time = await loop.run_in_executor(self._slm_executor, _slm_inference_sync)
            
            _t_inference = (time.perf_counter() - _t0) * 1000
            self._diag['slm_inference_ms'].append(_t_inference)
            
            # [DIAG] _predict_with_slm 전체 시간
            _t_slm_total = (time.perf_counter() - _t_slm_start) * 1000
            self._diag['slm_total_ms'].append(_t_slm_total)
            
            # 결과 해석: logits[0, i] = ln(gpu_type_i latency / 0.1초)
            # 역변환: exp(logits) → 0.1초 단위, 그 후 0.1 곱하기 → 초 단위
            predicted_latencies = {}
            result_dict = {}
            
            for idx, gpu_type in enumerate(self.gpu_types):
                latency = torch.exp(logits[0, idx]).item() * 0.1
                predicted_latencies[gpu_type] = latency
                result_dict[f"predicted_latency_{gpu_type.lower()}"] = latency
            
            # 가장 짧은 latency를 가진 GPU 종류 선택
            selected_gpu_type = min(predicted_latencies, key=predicted_latencies.get)
            selected_latency = predicted_latencies[selected_gpu_type]
            
            # latency 차이 계산 (최대값 - 최소값)
            sorted_latencies = sorted(predicted_latencies.values())
            latency_diff = sorted_latencies[-1] - sorted_latencies[0] if len(sorted_latencies) > 0 else 0
            
            # 선택된 GPU 종류의 예측 latency 저장 (MAE 계산용)
            result_dict['predicted_selected_server_latency'] = selected_latency
            
            # 로그 출력 (GPU 종류별)
            latency_strs = [f"{gt}: {predicted_latencies[gt]:.4f}s" for gt in self.gpu_types]
            logit_strs = [f"{logits[0, i].item():.4f}" for i in range(len(self.gpu_types))]
            
            pool_sizes = [f"{gt}({len(self.gpu_pools.get(gt, []))})" for gt in self.gpu_types]
            
            logger.info(f"[SLM PREDICTION] Selected type: {selected_gpu_type} | "
                       f"{' | '.join(latency_strs)} | "
                       f"Latency Diff: {latency_diff:.4f}s | "
                       f"Inference Time: {inference_time:.2f}ms | "
                       f"Raw logits: [{', '.join(logit_strs)}] | "
                       f"Pools: [{', '.join(pool_sizes)}]")
            
            result_dict.update({
                'selected_gpu_type': selected_gpu_type,
                'latency_diff': latency_diff,
                'inference_time_ms': inference_time
            })
            
            return result_dict, all_metrics
            
        except Exception as e:
            logger.error(f"SLM prediction error: {e}")
            return None, None
    
    async def _update_slm_mode(self):
        """최근 E2E latency 기반으로 SLM 모드 전환 여부 결정"""
        if len(self.recent_e2e_latencies) < self.slm_window_size:
            # 충분한 데이터가 쌓일 때까지 대기
            return
        
        avg_e2e = sum(self.recent_e2e_latencies) / len(self.recent_e2e_latencies)
        
        old_mode = self.slm_active
        
        if not self.slm_active and avg_e2e > self.slm_activation_threshold:
            # Base → SLM 전환
            self.slm_active = True
            self.slm_mode_switches += 1
            logger.warning(f"🔴 SLM MODE ACTIVATED: Avg E2E = {avg_e2e:.2f}s > {self.slm_activation_threshold}s "
                          f"(switch #{self.slm_mode_switches}, switching from {self.slm_base_algorithm.upper()})")
            
            # Fisher-Jenks: 모드 전환 시 diff 버퍼 및 EWMA 리셋
            if self.routing_algorithm == "fisher_jenks_sqf" and self.fj_window_reset_on_mode_change:
                self.fj_diff_buffer.clear()
                self.fj_current_split_point = self.fj_default_threshold
                self.fj_ewma_value = 0.0
                self.fj_ewma_initialized = False
                logger.info(f"[FJ] Diff buffer RESET on act transition (split_point → {self.fj_default_threshold})")
        
        elif self.slm_active and avg_e2e < self.slm_deactivation_threshold:
            # SLM → Base 전환
            self.slm_active = False
            self.slm_mode_switches += 1
            logger.info(f"✅ {self.slm_base_algorithm.upper()} MODE RESTORED: Avg E2E = {avg_e2e:.2f}s < {self.slm_deactivation_threshold}s "
                       f"(switch #{self.slm_mode_switches})")
            
            # Fisher-Jenks: 모드 전환 시 diff 버퍼 및 EWMA 리셋
            if self.routing_algorithm == "fisher_jenks_sqf" and self.fj_window_reset_on_mode_change:
                self.fj_diff_buffer.clear()
                self.fj_current_split_point = self.fj_default_threshold
                self.fj_ewma_value = 0.0
                self.fj_ewma_initialized = False
                logger.info(f"[FJ] Diff buffer RESET on deact transition (split_point → {self.fj_default_threshold})")
        
        # 주기적으로 상태 로깅
        if self.request_count % 100 == 0:
            base_algo_name = self.slm_base_algorithm.upper()
            mode_name = 'SLM' if self.slm_active else base_algo_name

            stats_parts = []
            for s in self.backend_servers:
                count = self.slm_prediction_stats.get(f"{s['name']}_selected", 0)
                stats_parts.append(f"{s['name']}={count}")

            logger.info(f"SLM Adaptive: mode={mode_name}, "
                       f"avg_e2e={avg_e2e:.2f}s, window={len(self.recent_e2e_latencies)}, "
                       f"predictions[{', '.join(stats_parts)}]")
    
    # ========== Fisher-Jenks SQF 관련 메서드 ==========
    
    @staticmethod
    def _compute_fisher_jenks_split(data: list) -> float:
        """
        2-Class Fisher-Jenks Natural Breaks Classification
        
        데이터를 두 그룹으로 나누는 최적의 split point를 찾음.
        GVF(Goodness of Variance Fit)를 최대화하는 분할점을 반환.
        
        Args:
            data: diff 값 리스트 (1등 제외, 양수 값들)
        
        Returns:
            float: 두 그룹을 나누는 최적 split point
        """
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        if n < 2:
            return sorted_data[0] if n == 1 else 0.0
        
        # 전체 분산 계산
        total_mean = sum(sorted_data) / n
        total_variance = sum((x - total_mean) ** 2 for x in sorted_data) / n
        
        if total_variance == 0:
            # 모든 값이 동일 → 그 값을 split point로 반환
            return sorted_data[0]
        
        best_gvf = -1.0
        best_split = sorted_data[n // 2]  # 기본값: 중앙값
        
        for i in range(1, n):
            # class1 = sorted_data[:i], class2 = sorted_data[i:]
            # 중복 분할점 건너뛰기 (같은 값에서 여러 번 나누는 것 방지)
            if sorted_data[i] == sorted_data[i - 1]:
                continue
            
            n1 = i
            n2 = n - i
            
            mean1 = sum(sorted_data[:i]) / n1
            mean2 = sum(sorted_data[i:]) / n2
            
            var1 = sum((x - mean1) ** 2 for x in sorted_data[:i]) / n1
            var2 = sum((x - mean2) ** 2 for x in sorted_data[i:]) / n2
            
            # 가중 within-class variance
            wcv = (n1 * var1 + n2 * var2) / n
            
            # GVF (Goodness of Variance Fit): 1에 가까울수록 좋은 분류
            gvf = 1.0 - (wcv / total_variance)
            
            if gvf > best_gvf:
                best_gvf = gvf
                # split point = 두 클래스 경계의 중간값
                best_split = (sorted_data[i - 1] + sorted_data[i]) / 2.0
        
        return best_split
    
    def _update_fj_split_point(self):
        """
        현재 diff 버퍼로 Fisher-Jenks split point 갱신
        
        - len(buffer) < fj_min_samples: 기본 threshold 사용 (갱신 안 함)
        - len(buffer) >= fj_min_samples: Fisher-Jenks 계산하여 갱신
        """
        buffer_len = len(self.fj_diff_buffer)
        
        if buffer_len < self.fj_min_samples:
            # 샘플 부족 → 기본 threshold 유지
            self.fj_stats['fj_fallback_used'] += 1
            return
        
        # Fisher-Jenks 계산
        data = list(self.fj_diff_buffer)
        new_split = self._compute_fisher_jenks_split(data)
        
        old_split = self.fj_current_split_point
        self.fj_current_split_point = new_split
        self.fj_stats['total_fj_computations'] += 1
        
        # split point 변화 추적 (최근 20개만)
        self.fj_split_history.append(new_split)
        if len(self.fj_split_history) > 20:
            self.fj_split_history.pop(0)
        
        if abs(old_split - new_split) > 0.1:
            logger.info(f"[FJ] Split point updated: {old_split:.4f}s → {new_split:.4f}s "
                       f"(buffer={buffer_len}/{self.fj_window_size}, "
                       f"computations={self.fj_stats['total_fj_computations']})")
    
    def _select_candidates_by_method(
        self,
        min_gpu_type: str,
        min_latency: float,
        current_diffs: dict,
        predicted_latencies: dict
    ) -> tuple:
        """
        fj_candidate_method에 따라 후보 GPU 종류 그룹과 사용된 threshold를 반환.
        
        Returns:
            tuple: (candidate_types: list, threshold_used: float)
        """
        method = self.fj_candidate_method
        
        if method == "fisher_jenks":
            self._update_fj_split_point()
            split_point = self.fj_current_split_point
            candidate_types = [min_gpu_type]
            for gtype, diff in current_diffs.items():
                if diff <= split_point:
                    candidate_types.append(gtype)
            return candidate_types, split_point
        
        elif method == "percent_of_min":
            threshold = min_latency * self.fj_percent_threshold
            candidate_types = [min_gpu_type]
            for gtype, diff in current_diffs.items():
                if diff <= threshold:
                    candidate_types.append(gtype)
            return candidate_types, threshold
        
        elif method == "top_n":
            sorted_types = sorted(predicted_latencies.keys(), key=lambda g: predicted_latencies[g])
            candidate_types = sorted_types[:self.fj_top_n]
            threshold = 0.0
            if len(candidate_types) > 1:
                threshold = predicted_latencies[candidate_types[-1]] - min_latency
            return candidate_types, threshold
        
        elif method == "ewma":
            all_diffs = list(current_diffs.values())
            for d in all_diffs:
                if not self.fj_ewma_initialized:
                    self.fj_ewma_value = d
                    self.fj_ewma_initialized = True
                else:
                    self.fj_ewma_value = self.fj_ewma_alpha * d + (1 - self.fj_ewma_alpha) * self.fj_ewma_value
            
            threshold = self.fj_ewma_value * self.fj_ewma_multiplier
            candidate_types = [min_gpu_type]
            for gtype, diff in current_diffs.items():
                if diff <= threshold:
                    candidate_types.append(gtype)
            return candidate_types, threshold
        
        elif method == "fixed_threshold":
            threshold = self.fj_fixed_threshold
            candidate_types = [min_gpu_type]
            for gtype, diff in current_diffs.items():
                if diff <= threshold:
                    candidate_types.append(gtype)
            return candidate_types, threshold
        
        else:
            logger.warning(f"[FJ] Unknown candidate method '{method}', falling back to fisher_jenks")
            self._update_fj_split_point()
            split_point = self.fj_current_split_point
            candidate_types = [min_gpu_type]
            for gtype, diff in current_diffs.items():
                if diff <= split_point:
                    candidate_types.append(gtype)
            return candidate_types, split_point
    
    async def _select_server_fisher_jenks(self, request_data: Optional[Dict] = None) -> tuple:
        """
        Fisher-Jenks SQF 라우팅 (GPU 풀 지원):
        1. SLM으로 각 GPU 종류의 E2E latency 예측 (풀 대표값 사용)
        2. 1등(최소 latency) GPU 종류와의 diff 계산, diff 버퍼에 추가
        3. Fisher-Jenks로 split point 갱신
        4. diff ≤ split point인 GPU 종류 + 1등 = 후보 GPU 종류 그룹
        5. 각 후보 GPU 종류의 풀에서 total(running+waiting) 최소 서버 선택
        6. 선택된 대표 서버들 간 SQF로 최종 선택
        
        Returns:
            tuple: (선택된 서버, SLM 정보, 메트릭 정보)
        """
        # SLM 정보 초기화 (GPU 종류별)
        slm_info = {
            'mode_active': self.slm_active,
            'prediction_used': False,
            'latency_diff': None,
            'recent_avg_e2e': None
        }
        for gpu_type in self.gpu_types:
            slm_info[f"predicted_latency_{gpu_type.lower()}"] = None
        
        if len(self.recent_e2e_latencies) > 0:
            slm_info['recent_avg_e2e'] = sum(self.recent_e2e_latencies) / len(self.recent_e2e_latencies)
        
        all_metrics = None
        
        try:
            # 모드 업데이트 (act/deact)
            await self._update_slm_mode()
            slm_info['mode_active'] = self.slm_active
            
            # === deact 상태: base algorithm 사용 ===
            if not self.slm_active or not request_data:
                if self.slm_base_algorithm == "weighted_round_robin":
                    server_idx = self.weighted_server_sequence[self.weighted_index]
                    self.weighted_index = (self.weighted_index + 1) % len(self.weighted_server_sequence)
                    return self.backend_servers[server_idx], slm_info, all_metrics
                elif self.slm_base_algorithm == "shortest_queue_first":
                    server, all_metrics = await self._select_shortest_queue_server()
                    return server, slm_info, all_metrics
                else:
                    server = self.backend_servers[self.current_server_index]
                    self.current_server_index = (self.current_server_index + 1) % len(self.backend_servers)
                    return server, slm_info, all_metrics
            
            # === act 상태: Fisher-Jenks SQF ===
            
            # Step 1: 프롬프트 추출
            prompt_text = ""
            messages = request_data.get("messages", [])
            if messages:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        prompt_text = msg.get("content", "")
                        break
            
            # Step 2: SLM 예측 (GPU 종류별 latency 예측)
            prediction, slm_metrics = await self._predict_with_slm(prompt_text)
            if slm_metrics:
                all_metrics = slm_metrics
            
            if not prediction:
                # SLM 예측 실패 → SQF fallback
                logger.warning("[FJ] SLM prediction FAILED, using SQF fallback")
                server, sqf_metrics = await self._select_shortest_queue_server()
                if sqf_metrics and not all_metrics:
                    all_metrics = sqf_metrics
                return server, slm_info, all_metrics
            
            # 예측 latency 정보 저장 (GPU 종류별)
            for key, val in prediction.items():
                if key.startswith('predicted_latency_'):
                    slm_info[key] = val
            slm_info['prediction_used'] = True
            slm_info['inference_time_ms'] = prediction.get('inference_time_ms')  # ⭐ SLM 추론 시간 전달
            self.slm_prediction_stats['total_predictions'] += 1
            
            # Step 3: 1등 GPU 종류 찾기 및 diff 계산
            predicted_latencies = {}
            for gpu_type in self.gpu_types:
                key = f"predicted_latency_{gpu_type.lower()}"
                if key in prediction:
                    predicted_latencies[gpu_type] = prediction[key]
            
            min_gpu_type = min(predicted_latencies, key=predicted_latencies.get)
            min_latency = predicted_latencies[min_gpu_type]
            
            # 1등 제외 나머지 GPU 종류의 diff 계산 및 버퍼 추가
            current_diffs = {}  # {gpu_type: diff}
            for gtype, lat in predicted_latencies.items():
                if gtype != min_gpu_type:
                    diff = lat - min_latency
                    current_diffs[gtype] = diff
                    self.fj_diff_buffer.append(diff)
            
            # Step 4-5: 후보 선정 방법에 따라 분기
            candidate_types, split_point = self._select_candidates_by_method(
                min_gpu_type, min_latency, current_diffs, predicted_latencies
            )
            
            slm_info['latency_diff'] = split_point
            slm_info['predicted_selected_server_latency'] = prediction.get('predicted_selected_server_latency')
            
            # 통계 업데이트 (GPU 종류별)
            for gtype in candidate_types:
                stats_key = f"{gtype}_in_candidate_group"
                if stats_key in self.fj_stats:
                    self.fj_stats[stats_key] += 1
            self.fj_stats['total_candidates_selected'] += len(candidate_types)
            
            # Step 6: 각 후보 GPU 종류의 풀에서 best 서버 선택 → SQF
            if all_metrics is None:
                all_metrics = await self._collect_all_metrics()
            
            # 각 후보 GPU 종류의 풀에서 total(running+waiting)이 가장 작은 서버 선택
            candidate_server_objs = []
            for gtype in candidate_types:
                best_in_pool = self._select_best_server_in_pool(gtype, all_metrics)
                candidate_server_objs.append(best_in_pool)
            
            # 대표 서버들 간 SQF (큐 길이 비교)
            server_queues = []
            for server in candidate_server_objs:
                server_name = server['name']
                if all_metrics and server_name in all_metrics:
                    metrics = all_metrics[server_name]
                    if metrics.get('num_requests_running', -1) == -1 or metrics.get('num_requests_waiting', -1) == -1:
                        queue_length = float('inf')
                    elif self.sqf_metric == "waiting":
                        queue_length = metrics.get('num_requests_waiting', 0)
                    else:
                        queue_length = metrics.get('num_requests_running', 0) + metrics.get('num_requests_waiting', 0)
                else:
                    queue_length = float('inf')
                server_queues.append({'server': server, 'queue_length': queue_length})
            
            # 최소 큐 길이 서버 선택
            min_queue = min(sq['queue_length'] for sq in server_queues)
            shortest_candidates = [sq['server'] for sq in server_queues if sq['queue_length'] == min_queue]
            selected_server = random.choice(shortest_candidates)
            
            # 선택된 서버가 속한 GPU 종류의 예측 latency 기록
            selected_gpu_type = selected_server.get('gpu_type', selected_server['name'])
            selected_key = f"predicted_latency_{selected_gpu_type.lower()}"
            if selected_key in prediction:
                slm_info['predicted_selected_server_latency'] = prediction[selected_key]
            
            # 서버 선택 통계 (물리 서버 단위)
            stats_key = f"{selected_server['name']}_selected"
            if stats_key in self.slm_prediction_stats:
                self.slm_prediction_stats[stats_key] += 1
            
            # 로그 출력
            latency_strs = [f"{gt}: {predicted_latencies[gt]:.2f}s" for gt in predicted_latencies]
            diff_strs = [f"{gt}: {d:.2f}s" for gt, d in current_diffs.items()]
            queue_strs = [f"{sq['server']['name']}={sq['queue_length']}" for sq in server_queues]
            
            logger.info(f"[FJ-SQF][{self.request_count}] "
                       f"method={self.fj_candidate_method} | "
                       f"Pred=[{', '.join(latency_strs)}] | "
                       f"1st_type={min_gpu_type}({min_latency:.2f}s) | "
                       f"Diffs=[{', '.join(diff_strs)}] | "
                       f"threshold={split_point:.4f}s | "
                       f"Candidate_types={candidate_types} | "
                       f"Pool_reps=[{', '.join(s['name'] for s in candidate_server_objs)}] | "
                       f"Queues=[{', '.join(queue_strs)}] | "
                       f"Selected={selected_server['name']} | "
                       f"buffer={len(self.fj_diff_buffer)}/{self.fj_window_size}")
            
            return selected_server, slm_info, all_metrics
        
        except Exception as e:
            logger.error(f"Error in Fisher-Jenks SQF: {e}")
            # Fallback to SQF
            try:
                server, sqf_metrics = await self._select_shortest_queue_server()
                return server, slm_info, sqf_metrics
            except:
                server_idx = self.weighted_server_sequence[self.weighted_index] if self.weighted_server_sequence else 0
                self.weighted_index = (self.weighted_index + 1) % max(len(self.weighted_server_sequence), 1)
                return self.backend_servers[server_idx], slm_info, all_metrics
    
    def get_server_url(self, server: Dict, endpoint: str = "/v1/chat/completions") -> str:
        """서버 URL 생성"""
        return f"http://{server['host']}:{server['port']}{endpoint}"
    
    async def forward_streaming_request(
        self, 
        request_data: Dict, 
        server: Dict,
        headers: Dict,
        request_id: int,
        slm_info: Optional[Dict] = None,  # ⭐ SLM 정보 추가
        all_metrics: Optional[Dict] = None,  # ⭐ 라우팅 시점의 메트릭
        endpoint: str = "/v1/chat/completions",
        arrival_time: Optional[float] = None  # ⭐ 핸들러 도착 시각 (time.time())
    ):
        """
        스트리밍 요청을 백엔드 서버로 전달하고 응답을 스트리밍
        TTFT와 E2E latency를 측정하여 CSV에 기록
        """
        # 활성 요청 수 증가
        await self.increment_active_requests()
        
        # 요청 시작 시간 (제너레이터 시작 시점)
        request_start_time = datetime.now()
        
        proxy_overhead = None
        slm_inference_time = None
        
        # SLM 추론 시간 추출 (slm_info에 기록된 경우)
        if slm_info and slm_info.get('inference_time_ms') is not None:
            slm_inference_time = slm_info['inference_time_ms'] / 1000.0  # ms → seconds
        
        ttft_seconds = None
        e2e_latency_seconds = None
        first_chunk_received = False
        
        try:
            # 메트릭 수집 (라우팅 시점의 메트릭이 없으면 새로 수집)
            if all_metrics is None:
                all_metrics = await self._collect_all_metrics()
            
            url = self.get_server_url(server, endpoint)
            
            logger.info(f"[Request {request_id}] Forwarding streaming request to {server['name']} ({url}) [endpoint: {endpoint}]")
            logger.debug(f"Request data: {json.dumps(request_data, ensure_ascii=False)[:200]}...")
            
            # 공유 HTTP 클라이언트 사용 (연결 풀 재사용)
            client = await self.get_http_client()
            
            # ★ 프록시 오버헤드 측정: 핸들러 도착 → 백엔드 HTTP 요청 전송 직전
            if arrival_time is not None:
                proxy_overhead = time.time() - arrival_time
            
            deadline = time.time() + self.total_request_timeout
            async with client.stream(
                "POST",
                url,
                json=request_data,
                headers=headers
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Backend server error: {response.status_code} - {error_text.decode()}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Backend server error: {error_text.decode()}"
                    )
                
                # 스트리밍 응답 전달 및 TTFT 측정
                async for chunk in response.aiter_bytes():
                    if chunk:
                        if not first_chunk_received:
                            ttft_seconds = (datetime.now() - request_start_time).total_seconds()
                            first_chunk_received = True
                            logger.debug(f"[Request {request_id}] TTFT: {ttft_seconds:.4f}s")
                        
                        yield chunk
                    
                    if time.time() > deadline:
                        raise TimeoutError(f"Total response time exceeded {self.total_request_timeout}s")
            
            # E2E latency 측정 (모든 chunk 완료)
            e2e_latency_seconds = (datetime.now() - request_start_time).total_seconds()
            
            # SLM Adaptive / Fisher-Jenks: E2E latency 추적
            if self.routing_algorithm in ("slm_adaptive", "fisher_jenks_sqf"):
                self.recent_e2e_latencies.append(e2e_latency_seconds)
            
            # 메트릭과 레이턴시를 CSV에 기록
            await self._log_metrics_to_csv(
                request_id, 
                server, 
                all_metrics,
                ttft_seconds=ttft_seconds,
                e2e_latency_seconds=e2e_latency_seconds,
                slm_info=slm_info,
                arrival_time=arrival_time,
                proxy_overhead_seconds=proxy_overhead,
                slm_inference_seconds=slm_inference_time
            )
            
            overhead_str = f", ProxyOverhead: {proxy_overhead:.4f}s" if proxy_overhead is not None else ""
            slm_str = f", SLM_Inference: {slm_inference_time:.4f}s" if slm_inference_time is not None else ""
            logger.info(f"[Request {request_id}] Completed - TTFT: {ttft_seconds:.4f}s, E2E: {e2e_latency_seconds:.4f}s{overhead_str}{slm_str}")
            
            # 성능 로깅
            if PERFORMANCE_CONFIG["log_slow_requests"] and e2e_latency_seconds > PERFORMANCE_CONFIG["slow_request_threshold"]:
                logger.warning(f"[Request {request_id}] Slow request: {e2e_latency_seconds:.2f}s")

        except TimeoutError:
            elapsed = (datetime.now() - request_start_time).total_seconds()
            logger.error(f"[Request {request_id}] Total timeout ({self.total_request_timeout}s) exceeded after {elapsed:.1f}s on {server['name']}")
            try:
                all_metrics = await self._collect_all_metrics()
                await self._log_metrics_to_csv(request_id, server, all_metrics, arrival_time=arrival_time, proxy_overhead_seconds=proxy_overhead, slm_inference_seconds=slm_inference_time)
            except:
                pass
            error_response = {
                "error": {
                    "message": f"Request timeout: total response time exceeded {self.total_request_timeout}s",
                    "type": "timeout_error",
                    "server": server['name']
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n".encode()
                            
        except httpx.RequestError as e:
            logger.error(f"[Request {request_id}] Request error to {server['name']}: {type(e).__name__}: {str(e) or repr(e)}")
            # 에러 발생 시에도 메트릭 기록 (레이턴시는 None)
            try:
                all_metrics = await self._collect_all_metrics()
                await self._log_metrics_to_csv(request_id, server, all_metrics, arrival_time=arrival_time, proxy_overhead_seconds=proxy_overhead, slm_inference_seconds=slm_inference_time)
            except:
                pass
            
            error_response = {
                "error": {
                    "message": f"Failed to connect to backend server: {type(e).__name__}: {str(e) or repr(e)}",
                    "type": "connection_error",
                    "server": server['name']
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n".encode()
        except Exception as e:
            logger.error(f"[Request {request_id}] Unexpected error to {server['name']}: {type(e).__name__}: {str(e) or repr(e)}")
            # 에러 발생 시에도 메트릭 기록
            try:
                all_metrics = await self._collect_all_metrics()
                await self._log_metrics_to_csv(request_id, server, all_metrics, arrival_time=arrival_time, proxy_overhead_seconds=proxy_overhead, slm_inference_seconds=slm_inference_time)
            except:
                pass
            
            error_response = {
                "error": {
                    "message": f"Unexpected error: {str(e)}",
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n".encode()
        finally:
            # 활성 요청 수 감소
            await self.decrement_active_requests()
    
    async def forward_non_streaming_request(
        self,
        request_data: Dict,
        server: Dict,
        headers: Dict,
        request_id: int,
        slm_info: Optional[Dict] = None,  # ⭐ SLM 정보 추가
        all_metrics: Optional[Dict] = None,  # ⭐ 라우팅 시점의 메트릭
        endpoint: str = "/v1/chat/completions",
        arrival_time: Optional[float] = None  # ⭐ 핸들러 도착 시각 (time.time())
    ) -> Dict:
        """
        비스트리밍 요청을 백엔드 서버로 전달
        TTFT와 E2E latency를 측정하여 CSV에 기록
        """
        # 활성 요청 수 증가
        await self.increment_active_requests()
        
        # 요청 시작 시간 (함수 시작 시점)
        request_start_time = datetime.now()
        
        proxy_overhead = None
        slm_inference_time = None
        
        # SLM 추론 시간 추출 (slm_info에 기록된 경우)
        if slm_info and slm_info.get('inference_time_ms') is not None:
            slm_inference_time = slm_info['inference_time_ms'] / 1000.0  # ms → seconds
        
        try:
            # 메트릭 수집 (라우팅 시점의 메트릭이 없으면 새로 수집)
            if all_metrics is None:
                all_metrics = await self._collect_all_metrics()
            
            url = self.get_server_url(server, endpoint)
            
            logger.info(f"[Request {request_id}] Forwarding non-streaming request to {server['name']} ({url}) [endpoint: {endpoint}]")
            logger.debug(f"Request data: {json.dumps(request_data, ensure_ascii=False)[:200]}...")
            
            # 공유 HTTP 클라이언트 사용 (연결 풀 재사용)
            client = await self.get_http_client()
            
            # ★ 프록시 오버헤드 측정: 핸들러 도착 → 백엔드 HTTP 요청 전송 직전
            if arrival_time is not None:
                proxy_overhead = time.time() - arrival_time
            
            async with asyncio.timeout(self.total_request_timeout):
                response = await client.post(
                    url,
                    json=request_data,
                    headers=headers
                )
            
            if response.status_code != 200:
                logger.error(f"Backend server error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Backend server error: {response.text}"
                )
            
            # E2E latency 측정 (응답 완료)
            e2e_latency_seconds = (datetime.now() - request_start_time).total_seconds()
            
            # 비스트리밍의 경우 TTFT = E2E (한 번에 응답 받음)
            ttft_seconds = e2e_latency_seconds
            
            # SLM Adaptive / Fisher-Jenks: E2E latency 추적
            if self.routing_algorithm in ("slm_adaptive", "fisher_jenks_sqf"):
                self.recent_e2e_latencies.append(e2e_latency_seconds)
            
            # 메트릭과 레이턴시를 CSV에 기록
            await self._log_metrics_to_csv(
                request_id, 
                server, 
                all_metrics,
                ttft_seconds=ttft_seconds,
                e2e_latency_seconds=e2e_latency_seconds,
                slm_info=slm_info,
                arrival_time=arrival_time,
                proxy_overhead_seconds=proxy_overhead,
                slm_inference_seconds=slm_inference_time
            )
            
            overhead_str = f", ProxyOverhead: {proxy_overhead:.4f}s" if proxy_overhead is not None else ""
            slm_str = f", SLM_Inference: {slm_inference_time:.4f}s" if slm_inference_time is not None else ""
            logger.info(f"[Request {request_id}] Completed - E2E: {e2e_latency_seconds:.4f}s{overhead_str}{slm_str}")
            
            # 성능 로깅
            if PERFORMANCE_CONFIG["log_slow_requests"] and e2e_latency_seconds > PERFORMANCE_CONFIG["slow_request_threshold"]:
                logger.warning(f"[Request {request_id}] Slow request: {e2e_latency_seconds:.2f}s")
            
            return response.json()

        except TimeoutError:
            elapsed = (datetime.now() - request_start_time).total_seconds()
            logger.error(f"[Request {request_id}] Total timeout ({self.total_request_timeout}s) exceeded after {elapsed:.1f}s on {server['name']}")
            try:
                all_metrics = await self._collect_all_metrics()
                await self._log_metrics_to_csv(request_id, server, all_metrics, arrival_time=arrival_time, proxy_overhead_seconds=proxy_overhead, slm_inference_seconds=slm_inference_time)
            except:
                pass
            raise HTTPException(
                status_code=504,
                detail=f"Request timeout: total response time exceeded {self.total_request_timeout}s"
            )
                
        except httpx.RequestError as e:
            logger.error(f"[Request {request_id}] Request error to {server['name']}: {type(e).__name__}: {str(e) or repr(e)}")
            # 에러 발생 시에도 메트릭 기록 (레이턴시는 None)
            try:
                all_metrics = await self._collect_all_metrics()
                await self._log_metrics_to_csv(request_id, server, all_metrics, arrival_time=arrival_time, proxy_overhead_seconds=proxy_overhead, slm_inference_seconds=slm_inference_time)
            except:
                pass
            
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to backend server: {type(e).__name__}: {str(e) or repr(e)}"
            )
        except Exception as e:
            logger.error(f"[Request {request_id}] Unexpected error to {server['name']}: {type(e).__name__}: {str(e) or repr(e)}")
            # 에러 발생 시에도 메트릭 기록
            try:
                all_metrics = await self._collect_all_metrics()
                await self._log_metrics_to_csv(request_id, server, all_metrics, arrival_time=arrival_time, proxy_overhead_seconds=proxy_overhead, slm_inference_seconds=slm_inference_time)
            except:
                pass
            
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
        finally:
            # 활성 요청 수 감소
            await self.decrement_active_requests()


# 프록시 서버 인스턴스 생성
proxy = ProxyServer(BACKEND_SERVERS)


# 앱 시작/종료 이벤트
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    logger.info("=" * 50)
    logger.info("vLLM Proxy Server Starting...")
    logger.info(f"Backend servers: {len(BACKEND_SERVERS)} servers")
    for server in BACKEND_SERVERS:
        logger.info(f"  - {server['name']}: {server['host']}:{server['port']}")
    logger.info(f"HTTP Client Config:")
    logger.info(f"  - Max connections: {HTTP_CLIENT_CONFIG['max_connections']}")
    logger.info(f"  - Max keepalive connections: {HTTP_CLIENT_CONFIG['max_keepalive_connections']}")
    logger.info(f"  - Read timeout: {HTTP_CLIENT_CONFIG['timeout_read']}s")
    logger.info(f"Uvicorn Config:")
    logger.info(f"  - Max concurrency: {UVICORN_CONFIG.get('limit_concurrency', 'unlimited')}")
    logger.info(f"  - Workers: {UVICORN_CONFIG.get('workers', 1)}")
    logger.info("=" * 50)
    
    # 백그라운드 메트릭 캐시 시작
    await proxy.start_metrics_cache_task()


@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 실행 (Ctrl+C / SIGTERM 포함)"""
    global _emergency_save_done
    logger.info("Shutting down vLLM Proxy Server...")
    
    # finalize 없이 중단된 경우 현재까지 기록된 결과를 긴급 저장
    if not _emergency_save_done and not proxy.shutting_down and proxy.request_count > 0:
        logger.warning("[SHUTDOWN] No finalize received — performing emergency save...")
        try:
            proxy.emergency_save()
            _emergency_save_done = True
        except Exception as e:
            logger.error(f"[SHUTDOWN] Emergency save failed: {e}")
    
    await proxy.close()
    logger.info("Proxy server closed successfully")


@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {
        "status": "running",
        "service": "vLLM Proxy Server",
        "backend_servers": BACKEND_SERVERS,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


@app.get("/diagnostics")
async def diagnostics():
    """
    병목 진단 엔드포인트
    
    각 단계별 소요시간 통계를 반환합니다.
    실험 후 이 결과를 에이전트에게 보여주면 문제 원인을 파악할 수 있습니다.
    
    단계별 의미:
    - handler_json_parse_ms: 요청 JSON 파싱 시간
    - handler_routing_ms: 라우팅 결정 전체 (get_next_server)
    - handler_total_ms: 전체 프록시 오버헤드 (arrival → 백엔드 포워딩 직전)
    - slm_metrics_ms: 백엔드 메트릭 수집 (캐시 사용 시 ~0ms)
    - slm_prompt_gen_ms: GPU pool 집계 + SLM 프롬프트 생성
    - slm_inference_ms: SLM 모델 추론 (토크나이징 + GPU 포워드)
    - slm_total_ms: _predict_with_slm 전체
    - batch_queue_wait_ms: 배치 큐 제출 → 결과 수신 (큐 대기 + 추론)
    - batch_queue_depth: 배치 큐에 제출 시점의 큐 깊이 (높으면 추론 지연)
    """
    try:
        result = proxy.get_diagnostics()
        result['timestamp'] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error(f"Error in diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnostics/reset")
async def reset_diagnostics_endpoint():
    """진단 데이터 초기화"""
    proxy.reset_diagnostics()
    return {"status": "reset", "timestamp": datetime.now().isoformat()}


# ===== QPS 병목 진단용 테스트 엔드포인트 =====
_qps_test_arrivals = []  # (arrival_time, request_id, body_size)
_qps_test_lock = asyncio.Lock()

@app.post("/v1/qps_test/echo")
async def qps_test_echo(request: Request):
    """
    QPS 병목 진단: 백엔드 전달 없이 요청 도착률만 측정
    클라이언트와 동일한 payload로 요청하면 네트워크/파싱 오버헤드 측정 가능
    
    curl -X POST http://proxy:8000/v1/qps_test/echo -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"test"}]}'
    """
    arrival_time = time.time()
    body = await request.body()
    body_parsed_time = time.time()
    
    async with _qps_test_lock:
        _qps_test_arrivals.append((arrival_time, body_parsed_time, len(body)))
    
    return {
        "status": "ok",
        "body_size": len(body),
        "body_parse_ms": round((body_parsed_time - arrival_time) * 1000, 2)
    }

@app.get("/v1/qps_test/results")
async def qps_test_results():
    """
    QPS 테스트 결과 조회
    
    반환: 도착률 통계, 초당 도착 분포, inter-arrival time 분포
    """
    async with _qps_test_lock:
        arrivals = list(_qps_test_arrivals)
    
    if len(arrivals) < 2:
        return {"message": "Not enough data", "count": len(arrivals)}
    
    total = len(arrivals)
    duration = arrivals[-1][0] - arrivals[0][0]
    
    # Inter-arrival times
    inter_arrivals_ms = [(arrivals[i][0] - arrivals[i-1][0]) * 1000 for i in range(1, len(arrivals))]
    ia_sorted = sorted(inter_arrivals_ms)
    
    # Body parse times
    parse_times_ms = [(a[1] - a[0]) * 1000 for a in arrivals]
    pt_sorted = sorted(parse_times_ms)
    
    # Per-second counts
    t0 = arrivals[0][0]
    per_sec = {}
    for a in arrivals:
        sec = int(a[0] - t0)
        per_sec[sec] = per_sec.get(sec, 0) + 1
    counts = list(per_sec.values())
    
    def p(arr, pct):
        idx = min(int(len(arr) * pct / 100), len(arr) - 1)
        return round(arr[idx], 2)
    
    return {
        "total_requests": total,
        "duration_sec": round(duration, 2),
        "overall_qps": round((total - 1) / duration, 2) if duration > 0 else 0,
        "inter_arrival_ms": {
            "avg": round(sum(inter_arrivals_ms) / len(inter_arrivals_ms), 2),
            "p50": p(ia_sorted, 50),
            "p90": p(ia_sorted, 90),
            "p99": p(ia_sorted, 99),
            "min": round(ia_sorted[0], 2),
            "max": round(ia_sorted[-1], 2),
        },
        "body_parse_ms": {
            "avg": round(sum(parse_times_ms) / len(parse_times_ms), 2),
            "p50": p(pt_sorted, 50),
            "p90": p(pt_sorted, 90),
            "p99": p(pt_sorted, 99),
            "max": round(pt_sorted[-1], 2),
        },
        "per_second_arrival": {
            "avg": round(sum(counts) / len(counts), 1),
            "min": min(counts),
            "max": max(counts),
            "distribution": {str(k): v for k, v in sorted(per_sec.items())},
        },
        "avg_body_size": round(sum(a[2] for a in arrivals) / total, 0),
    }

@app.post("/v1/qps_test/reset")
async def qps_test_reset():
    """QPS 테스트 데이터 초기화"""
    async with _qps_test_lock:
        _qps_test_arrivals.clear()
    return {"status": "reset", "message": "QPS test data cleared"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI 호환 Chat Completions API
    스트리밍 및 비스트리밍 모두 지원
    """
    # ===== 요청 도착 시각 기록 (병목 진단용, request.json() 이전에 기록) =====
    arrival_time = time.time()
    proxy.request_arrival_times.append(arrival_time)
    
    # N초마다 도착률 로그 출력
    if proxy._last_arrival_log_time is None:
        proxy._last_arrival_log_time = arrival_time
        proxy._last_arrival_log_count = 0
    
    proxy._last_arrival_log_count += 1
    elapsed_since_log = arrival_time - proxy._last_arrival_log_time
    if elapsed_since_log >= proxy._arrival_log_interval:
        # 최근 N초간 도착률 계산
        recent_rate = proxy._last_arrival_log_count / elapsed_since_log if elapsed_since_log > 0 else 0
        
        # 최근 1초간 도착률 (더 정밀한 순간 속도)
        one_sec_ago = arrival_time - 1.0
        recent_1s_count = sum(1 for t in proxy.request_arrival_times if t >= one_sec_ago)
        
        # 최근 10초간 도착률
        ten_sec_ago = arrival_time - 10.0
        recent_10s_count = sum(1 for t in proxy.request_arrival_times if t >= ten_sec_ago)
        recent_10s_rate = recent_10s_count / min(10.0, arrival_time - proxy.request_arrival_times[0]) if proxy.request_arrival_times else 0
        
        logger.info(
            f"[ARRIVAL-RATE] 최근 {elapsed_since_log:.1f}초: {recent_rate:.1f} req/s | "
            f"최근 1초: {recent_1s_count} req | "
            f"최근 10초 평균: {recent_10s_rate:.1f} req/s | "
            f"활성 요청: {proxy.active_requests} | "
            f"총 수신: {proxy.request_count + 1}"
        )
        
        proxy._last_arrival_log_time = arrival_time
        proxy._last_arrival_log_count = 0
    
    # 종료 대기 중이면 새 요청 거부
    if proxy.shutting_down:
        logger.warning("Server is shutting down. Rejecting new request.")
        raise HTTPException(
            status_code=503,
            detail="Server is shutting down. No new requests accepted."
        )
    
    try:
        # [DIAG] 요청 데이터 파싱
        _t0 = time.perf_counter()
        request_data = await request.json()
        _t_json_ms = (time.perf_counter() - _t0) * 1000
        proxy._diag['handler_json_parse_ms'].append(_t_json_ms)
        
        # 요청 카운트 증가
        proxy.request_count += 1
        current_request_id = proxy.request_count
        
        # 스트리밍 여부 확인
        is_streaming = request_data.get("stream", False)
        
        # [DIAG] 다음 서버 선택 (SLM의 경우 request_data 필요) + 메트릭 수집
        _t0 = time.perf_counter()
        server, slm_info, all_metrics = await proxy.get_next_server(request_data)
        _t_routing_ms = (time.perf_counter() - _t0) * 1000
        proxy._diag['handler_routing_ms'].append(_t_routing_ms)
        
        # [DIAG] 전체 프록시 오버헤드 (arrival → 서버 선택 완료)
        _t_total_ms = (time.time() - arrival_time) * 1000 if arrival_time else 0
        proxy._diag['handler_total_ms'].append(_t_total_ms)
        proxy._diag['total_diagnosed'] += 1
        
        # [DIAG] 배치 추론기 배치 크기 수집
        if hasattr(proxy, 'slm_batch_inferencer') and proxy.slm_batch_inferencer._stats.get('total_batches', 0) > 0:
            batch_stats = proxy.slm_batch_inferencer._stats
            if 'max_batch_seen' in batch_stats:
                proxy._diag['batch_size'].append(batch_stats.get('max_batch_seen', 0))
        
        # [DIAG] 주기적 요약 로그 출력 (N개 요청마다)
        _diag_count = proxy._diag['total_diagnosed']
        _summary_interval = proxy._diag['summary_interval']
        if _diag_count > 0 and _diag_count % _summary_interval == 0 and _diag_count != proxy._diag['last_summary_request_id']:
            proxy._diag['last_summary_request_id'] = _diag_count
            proxy._log_diag_summary()
        
        # 헤더 준비 (필요한 헤더만 전달)
        headers = {
            "Content-Type": "application/json",
        }
        
        # Authorization 헤더가 있으면 전달
        if "authorization" in request.headers:
            headers["Authorization"] = request.headers["authorization"]
        
        if is_streaming:
            # 스트리밍 응답
            return StreamingResponse(
                proxy.forward_streaming_request(request_data, server, headers, current_request_id, slm_info, all_metrics, arrival_time=arrival_time),
                media_type="text/event-stream"
            )
        else:
            # 비스트리밍 응답
            response_data = await proxy.forward_non_streaming_request(
                request_data, server, headers, current_request_id, slm_info, all_metrics, arrival_time=arrival_time
            )
            return JSONResponse(content=response_data)
            
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat_completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_algorithm")
async def set_algorithm(request: Request):
    """
    라우팅 알고리즘 및 설정 변경
    
    Request body:
        {
            "algorithm": "round_robin" | "weighted_round_robin" | "shortest_queue_first" | "slm_adaptive",
            
            // Weighted Round Robin 전용 옵션 (algorithm="weighted_round_robin"일 때만 적용)
            "weights": [1, 2, 2.5],                   // GPU별 weight 값 (RTX3090, RTX4090, RTX5090 순서)
            
            // SLM Adaptive 전용 옵션 (algorithm="slm_adaptive"일 때만 적용)
            "slm_activation_threshold": 15.0,         // SLM 활성화 E2E 임계값 (초)
            "slm_deactivation_threshold": 10.0        // SLM 비활성화 E2E 임계값 (초)
        }
    """
    try:
        data = await request.json()
        new_algorithm = data.get("algorithm")
        
        if not new_algorithm:
            raise HTTPException(status_code=400, detail="algorithm parameter is required")
        
        # 유효한 알고리즘 목록
        valid_algorithms = ["round_robin", "weighted_round_robin", "shortest_queue_first", "slm_adaptive", "fisher_jenks_sqf"]
        
        if new_algorithm not in valid_algorithms:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid algorithm. Must be one of: {', '.join(valid_algorithms)}"
            )
        
        old_algorithm = proxy.routing_algorithm
        old_settings = {}
        
        # 알고리즘 변경
        proxy.routing_algorithm = new_algorithm
        
        # 데이터셋 저장 (결과 저장 경로에 사용)
        if "dataset" in data:
            proxy.current_dataset = data["dataset"]
            logger.info(f"[Config] Dataset set to: {proxy.current_dataset}")
        
        # QPS 저장 (결과 저장 경로에 사용)
        if "qps" in data:
            proxy.current_qps = data["qps"]
            logger.info(f"[Config] QPS set to: {proxy.current_qps}")
        
        # Weighted Round Robin 설정 처리
        if new_algorithm == "weighted_round_robin":
            if "weights" in data:
                weights = data["weights"]
                
                # 1. weights가 리스트인지 확인
                if not isinstance(weights, list):
                    raise HTTPException(status_code=400, detail="weights must be a list")
                
                # 2. weights 개수가 서버 개수와 일치하는지 확인
                num_servers = len(proxy.backend_servers)
                if len(weights) != num_servers:
                    server_names = [s['name'] for s in proxy.backend_servers]
                    raise HTTPException(
                        status_code=400,
                        detail=f"weights length ({len(weights)}) must match number of servers ({num_servers}). "
                               f"Expected format: {server_names} -> [weight1, weight2, weight3]"
                    )
                
                # 3. 모든 weight가 양의 정수로 변환 가능한지 확인
                try:
                    weights_int = []
                    for i, w in enumerate(weights):
                        weight_val = int(w)
                        if weight_val <= 0:
                            server_name = proxy.backend_servers[i]['name']
                            raise HTTPException(
                                status_code=400, 
                                detail=f"Weight for {server_name} must be a positive integer (got {w})"
                            )
                        weights_int.append(weight_val)
                    weights = weights_int
                except (ValueError, TypeError) as e:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"All weights must be positive integers. Error: {str(e)}"
                    )
                
                # 4. 기존 설정 저장
                old_settings["weights"] = proxy.routing_weights.copy() if proxy.routing_weights else None
                
                # 5. 새로운 weight 설정
                proxy.routing_weights = weights
                
                # 6. Weighted sequence 재생성
                proxy._build_weighted_sequence()
                
                # 7. 로그 출력
                logger.info(f"[WRR Config] Weights changed: {old_settings.get('weights')} → {proxy.routing_weights}")
                logger.info(f"[WRR Config] Weight ratio: {':'.join(map(str, proxy.routing_weights))}")
                logger.info(f"[WRR Config] Sequence length: {len(proxy.weighted_server_sequence)}")
                
                # 서버별 weight 출력
                for i, server in enumerate(proxy.backend_servers):
                    logger.info(f"[WRR Config]   {server['name']}: weight={weights[i]}")
            elif "qps" in data:
                # weights 없이 qps만 전달 → RR 결과 기반 자동 weight 계산
                qps = data["qps"]
                dataset = data.get("dataset", None)  # "sharegpt" 또는 "lmsys-chat-1m"
                auto_weights = proxy._calculate_weights_from_rr(qps, dataset)
                if auto_weights:
                    old_settings["weights"] = proxy.routing_weights.copy() if proxy.routing_weights else None
                    proxy.routing_weights = auto_weights
                    proxy._build_weighted_sequence()
                    logger.info(f"[WRR Config] Auto weights from RR(QPS {qps}): {auto_weights}")
                    logger.info(f"[WRR Config] Weight ratio: {':'.join(map(str, auto_weights))}")
                    for i, server in enumerate(proxy.backend_servers):
                        logger.info(f"[WRR Config]   {server['name']}: weight={auto_weights[i]}")
                else:
                    logger.warning(f"[WRR Config] No RR data for QPS {qps}, using existing weights: {proxy.routing_weights}")
            else:
                # weights도 qps도 없으면 기존값 사용
                logger.info(f"[WRR Config] Using existing weights: {proxy.routing_weights}")
        
        # 알고리즘별 상태 초기화
        if new_algorithm == "slm_adaptive":
            # SLM 모드 초기화
            if not hasattr(proxy, 'slm_active'):
                proxy.slm_active = False
                proxy.recent_e2e_latencies = deque(maxlen=proxy.slm_window_size)
                proxy.slm_mode_switches = 0
                proxy.inflight_tokens = {}
                for server in proxy.backend_servers:
                    proxy.inflight_tokens[server['name']] = deque(maxlen=500)
                proxy.slm_model = None
                proxy.slm_tokenizer = None
                proxy._slm_load_lock = asyncio.Lock()
                proxy._slm_device = None
                proxy._slm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slm-inference")
                proxy.slm_batch_enabled = ROUTING_CONFIG.get("slm_batch_enabled", True)
                proxy.slm_batch_inferencer = SLMBatchInferencer(
                    max_batch_size=ROUTING_CONFIG.get("slm_batch_max_size", 16),
                    max_wait_ms=ROUTING_CONFIG.get("slm_batch_max_wait_ms", 50.0),
                    executor=proxy._slm_executor
                )
                proxy.slm_prediction_stats = {
                    'total_predictions': 0,
                }
                for server in proxy.backend_servers:
                    proxy.slm_prediction_stats[f"{server['name']}_selected"] = 0
            else:
                # 기존 SLM 상태 초기화
                proxy.slm_active = False
                proxy.recent_e2e_latencies.clear()
                proxy.slm_mode_switches = 0
                for server in proxy.backend_servers:
                    if server['name'] in proxy.inflight_tokens:
                        proxy.inflight_tokens[server['name']].clear()
                proxy.slm_prediction_stats = {
                    'total_predictions': 0,
                }
                for server in proxy.backend_servers:
                    proxy.slm_prediction_stats[f"{server['name']}_selected"] = 0

            if "slm_activation_threshold" in data:
                old_settings["slm_activation_threshold"] = proxy.slm_activation_threshold
                proxy.slm_activation_threshold = float(data["slm_activation_threshold"])
                logger.info(f"[SLM Config] Activation threshold: {old_settings['slm_activation_threshold']} → {proxy.slm_activation_threshold}")
            
            if "slm_deactivation_threshold" in data:
                old_settings["slm_deactivation_threshold"] = proxy.slm_deactivation_threshold
                proxy.slm_deactivation_threshold = float(data["slm_deactivation_threshold"])
                logger.info(f"[SLM Config] Deactivation threshold: {old_settings['slm_deactivation_threshold']} → {proxy.slm_deactivation_threshold}")
            
        # Fisher-Jenks SQF 알고리즘별 상태 초기화
        if new_algorithm == "fisher_jenks_sqf":
            # SLM 공통 상태 초기화 (slm_adaptive와 동일)
            if not hasattr(proxy, 'slm_active'):
                proxy.slm_model_path = ROUTING_CONFIG.get("slm_model_path")
                proxy.slm_window_size = ROUTING_CONFIG.get("slm_window_size", 20)
                proxy.slm_activation_threshold = ROUTING_CONFIG.get("slm_activation_threshold", 25.0)
                proxy.slm_deactivation_threshold = ROUTING_CONFIG.get("slm_deactivation_threshold", 15.0)
                proxy.slm_base_algorithm = ROUTING_CONFIG.get("slm_base_algorithm", "weighted_round_robin")
                proxy.slm_fallback = ROUTING_CONFIG.get("slm_fallback", "weighted_round_robin")
                proxy.slm_active = False
                proxy.recent_e2e_latencies = deque(maxlen=proxy.slm_window_size)
                proxy.slm_mode_switches = 0
                proxy.inflight_tokens = {}
                for server in proxy.backend_servers:
                    proxy.inflight_tokens[server['name']] = deque(maxlen=500)
                proxy.slm_model = None
                proxy.slm_tokenizer = None
                proxy._slm_load_lock = asyncio.Lock()
                proxy._slm_device = None
                proxy._slm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slm-inference")
                proxy.slm_batch_enabled = ROUTING_CONFIG.get("slm_batch_enabled", True)
                proxy.slm_batch_inferencer = SLMBatchInferencer(
                    max_batch_size=ROUTING_CONFIG.get("slm_batch_max_size", 16),
                    max_wait_ms=ROUTING_CONFIG.get("slm_batch_max_wait_ms", 50.0),
                    executor=proxy._slm_executor
                )
                proxy.slm_prediction_stats = {
                    'total_predictions': 0,
                }
                for server in proxy.backend_servers:
                    proxy.slm_prediction_stats[f"{server['name']}_selected"] = 0
            else:
                proxy.slm_active = False
                proxy.recent_e2e_latencies.clear()
                proxy.slm_mode_switches = 0
                for server in proxy.backend_servers:
                    if server['name'] in proxy.inflight_tokens:
                        proxy.inflight_tokens[server['name']].clear()
                proxy.slm_prediction_stats = {
                    'total_predictions': 0,
                }
                for server in proxy.backend_servers:
                    proxy.slm_prediction_stats[f"{server['name']}_selected"] = 0

            # 프리셋 적용: fj_preset이 주어지면 EXPERIMENT_PRESETS에서 파라미터 로드
            if "fj_preset" in data:
                preset_name = data["fj_preset"]
                if preset_name in EXPERIMENT_PRESETS:
                    preset = EXPERIMENT_PRESETS[preset_name]
                    for k, v in preset.items():
                        data.setdefault(k, v)
                    logger.info(f"[FJ Config] Preset '{preset_name}' applied: {preset}")
                else:
                    logger.warning(f"[FJ Config] Unknown preset '{preset_name}', available: {list(EXPERIMENT_PRESETS.keys())}")
            
            # Fisher-Jenks 전용 상태 초기화
            proxy.fj_window_size = ROUTING_CONFIG.get("fj_window_size", 50)
            proxy.fj_min_samples = ROUTING_CONFIG.get("fj_min_samples", 15)
            proxy.fj_default_threshold = ROUTING_CONFIG.get("fj_default_threshold", 0.0)
            proxy.fj_window_reset_on_mode_change = ROUTING_CONFIG.get("fj_window_reset_on_mode_change", True)
            proxy.fj_candidate_method = ROUTING_CONFIG.get("fj_candidate_method", "fisher_jenks")
            proxy.fj_percent_threshold = ROUTING_CONFIG.get("fj_percent_threshold", 0.07)
            proxy.fj_top_n = ROUTING_CONFIG.get("fj_top_n", 2)
            proxy.fj_ewma_alpha = ROUTING_CONFIG.get("fj_ewma_alpha", 0.2)
            proxy.fj_ewma_multiplier = ROUTING_CONFIG.get("fj_ewma_multiplier", 1.4)
            proxy.fj_fixed_threshold = ROUTING_CONFIG.get("fj_fixed_threshold", 10.5)
            proxy.fj_ewma_value = 0.0
            proxy.fj_ewma_initialized = False
            proxy.fj_diff_buffer = deque(maxlen=proxy.fj_window_size)
            proxy.fj_current_split_point = proxy.fj_default_threshold
            proxy.fj_split_history = []
            proxy.fj_stats = {
                'total_fj_computations': 0,
                'total_candidates_selected': 0,
                'fj_fallback_used': 0,
            }
            for gpu_type in proxy.gpu_types:
                proxy.fj_stats[f"{gpu_type}_in_candidate_group"] = 0
            
            # Fisher-Jenks 전용 설정 변경 (제공된 경우)
            if "fj_candidate_method" in data:
                proxy.fj_candidate_method = str(data["fj_candidate_method"])
                logger.info(f"[FJ Config] Candidate method: {proxy.fj_candidate_method}")
            if "fj_window_size" in data:
                proxy.fj_window_size = int(data["fj_window_size"])
                proxy.fj_diff_buffer = deque(maxlen=proxy.fj_window_size)
                logger.info(f"[FJ Config] Window size: {proxy.fj_window_size}")
            if "fj_min_samples" in data:
                proxy.fj_min_samples = int(data["fj_min_samples"])
                logger.info(f"[FJ Config] Min samples: {proxy.fj_min_samples}")
            if "fj_default_threshold" in data:
                proxy.fj_default_threshold = float(data["fj_default_threshold"])
                proxy.fj_current_split_point = proxy.fj_default_threshold
                logger.info(f"[FJ Config] Default threshold: {proxy.fj_default_threshold}")
            if "fj_percent_threshold" in data:
                proxy.fj_percent_threshold = float(data["fj_percent_threshold"])
                logger.info(f"[FJ Config] Percent threshold: {proxy.fj_percent_threshold}")
            if "fj_top_n" in data:
                proxy.fj_top_n = int(data["fj_top_n"])
                logger.info(f"[FJ Config] Top N: {proxy.fj_top_n}")
            if "fj_ewma_alpha" in data:
                proxy.fj_ewma_alpha = float(data["fj_ewma_alpha"])
                logger.info(f"[FJ Config] EWMA alpha: {proxy.fj_ewma_alpha}")
            if "fj_ewma_multiplier" in data:
                proxy.fj_ewma_multiplier = float(data["fj_ewma_multiplier"])
                logger.info(f"[FJ Config] EWMA multiplier: {proxy.fj_ewma_multiplier}")
            if "fj_fixed_threshold" in data:
                proxy.fj_fixed_threshold = float(data["fj_fixed_threshold"])
                logger.info(f"[FJ Config] Fixed threshold: {proxy.fj_fixed_threshold}")
            
            # SLM 관련 설정 변경 (act/deact threshold 등)
            if "slm_activation_threshold" in data:
                old_settings["slm_activation_threshold"] = proxy.slm_activation_threshold
                proxy.slm_activation_threshold = float(data["slm_activation_threshold"])
                logger.info(f"[FJ Config] Activation threshold: {old_settings['slm_activation_threshold']} → {proxy.slm_activation_threshold}")
            if "slm_deactivation_threshold" in data:
                old_settings["slm_deactivation_threshold"] = proxy.slm_deactivation_threshold
                proxy.slm_deactivation_threshold = float(data["slm_deactivation_threshold"])
                logger.info(f"[FJ Config] Deactivation threshold: {old_settings['slm_deactivation_threshold']} → {proxy.slm_deactivation_threshold}")
            
            logger.info(f"[FJ] Fisher-Jenks SQF initialized: method={proxy.fj_candidate_method}, "
                       f"window={proxy.fj_window_size}, min_samples={proxy.fj_min_samples}, "
                       f"default_threshold={proxy.fj_default_threshold}")
        
        # 인덱스 초기화
        proxy.current_server_index = 0
        proxy.weighted_index = 0
        
        # 메트릭 실패 카운터 리셋
        proxy._metrics_fail_log_count = 0
        
        # CSV 파일 초기화 + 요청 카운터 리셋 (새 실험 시작)
        previous_count = proxy.request_count
        proxy.reset_csv_file()
        proxy.request_count = 0
        logger.info(f"[Algorithm Change] CSV reset & request_count reset: {previous_count} → 0")
        
        # SLM 배치 추론기 통계 리셋
        if hasattr(proxy, 'slm_batch_inferencer') and proxy.slm_batch_inferencer is not None:
            proxy.slm_batch_inferencer._stats = {
                'total_batches': 0,
                'total_items': 0,
                'max_batch_seen': 0,
            }
            logger.info(f"[Algorithm Change] SLM batch inferencer stats reset")
        
        # SLM이 아닌 알고리즘으로 전환 시 SLM 상태 비활성화
        if new_algorithm not in ("slm_adaptive", "fisher_jenks_sqf"):
            if hasattr(proxy, 'slm_active') and proxy.slm_active:
                proxy.slm_active = False
                logger.info(f"[Algorithm Change] SLM mode deactivated (switching to non-SLM algorithm)")
        
        # 백그라운드 메트릭 캐시 태스크 관리 (알고리즘 전환 시 주기 조정 + 재시작)
        # 기존 태스크 중지
        if proxy._metrics_cache_task is not None:
            proxy._metrics_cache_task.cancel()
            try:
                await proxy._metrics_cache_task
            except asyncio.CancelledError:
                pass
            proxy._metrics_cache_task = None
        
        # 새 알고리즘에 맞는 캐시 주기 설정
        if new_algorithm in ("shortest_queue_first", "slm_adaptive", "fisher_jenks_sqf"):
            proxy._metrics_cache_interval = 0.2  # 200ms (라우팅용)
        else:
            proxy._metrics_cache_interval = 0.5  # 500ms (로깅용)
        
        # 캐시 태스크 재시작
        await proxy.start_metrics_cache_task()
        
        logger.info(f"[Algorithm Change] {old_algorithm} → {new_algorithm}")
        logger.info(f"[Algorithm Change] Verified: proxy.routing_algorithm = {proxy.routing_algorithm}")
        
        response = {
            "status": "success",
            "message": f"Algorithm changed from {old_algorithm} to {new_algorithm}",
            "old_algorithm": old_algorithm,
            "new_algorithm": new_algorithm,
            "timestamp": datetime.now().isoformat()
        }
        
        # WRR 설정이 변경된 경우 응답에 포함
        if new_algorithm == "weighted_round_robin":
            response["weights"] = proxy.routing_weights
            if "weights" in old_settings:
                response["old_weights"] = old_settings["weights"]
        
        # SLM 설정이 변경된 경우 응답에 포함
        if new_algorithm == "slm_adaptive" and old_settings:
            response["old_settings"] = old_settings
            response["new_settings"] = {
                "slm_activation_threshold": proxy.slm_activation_threshold,
                "slm_deactivation_threshold": proxy.slm_deactivation_threshold
            }
        
        return response
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing algorithm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """프록시 서버 통계"""
    # 도착률 계산
    now = time.time()
    arrival_rate_1s = sum(1 for t in proxy.request_arrival_times if t >= now - 1.0)
    arrival_rate_5s = sum(1 for t in proxy.request_arrival_times if t >= now - 5.0)
    arrival_rate_10s = sum(1 for t in proxy.request_arrival_times if t >= now - 10.0)
    arrival_rate_30s = sum(1 for t in proxy.request_arrival_times if t >= now - 30.0)
    
    stats = {
        "total_requests": proxy.request_count,
        "active_requests": proxy.active_requests,
        "arrival_rate": {
            "last_1s": arrival_rate_1s,
            "last_5s_avg": round(arrival_rate_5s / 5.0, 2),
            "last_10s_avg": round(arrival_rate_10s / 10.0, 2),
            "last_30s_avg": round(arrival_rate_30s / 30.0, 2),
        },
        "shutting_down": proxy.shutting_down,
        "routing_algorithm": proxy.routing_algorithm,
        "backend_servers": BACKEND_SERVERS,
        "current_server_index": proxy.current_server_index,
        "metrics_csv_path": proxy.metrics_csv_path,
        "timestamp": datetime.now().isoformat()
    }
    
    if proxy.routing_algorithm == "weighted_round_robin":
        stats["routing_weights"] = proxy.routing_weights
        stats["weight_ratio"] = ":".join(map(str, proxy.routing_weights))
    elif proxy.routing_algorithm == "shortest_queue_first":
        stats["sqf_metric"] = proxy.sqf_metric
        stats["sqf_fallback"] = proxy.sqf_fallback
    elif proxy.routing_algorithm == "slm_adaptive":
        base_algo_name = proxy.slm_base_algorithm.upper()
        stats["slm_active"] = proxy.slm_active
        stats["slm_mode"] = "SLM" if proxy.slm_active else base_algo_name
        stats["slm_base_algorithm"] = proxy.slm_base_algorithm
        stats["slm_fallback"] = proxy.slm_fallback
        stats["slm_mode_switches"] = proxy.slm_mode_switches
        stats["slm_window_size"] = proxy.slm_window_size
        stats["slm_activation_threshold"] = proxy.slm_activation_threshold
        stats["slm_deactivation_threshold"] = proxy.slm_deactivation_threshold
        stats["slm_prediction_stats"] = proxy.slm_prediction_stats
        if hasattr(proxy, 'slm_batch_inferencer') and proxy.slm_batch_inferencer is not None:
            stats["slm_batch_stats"] = proxy.slm_batch_inferencer.get_stats()
        if len(proxy.recent_e2e_latencies) > 0:
            stats["recent_avg_e2e"] = sum(proxy.recent_e2e_latencies) / len(proxy.recent_e2e_latencies)
            stats["recent_e2e_window"] = len(proxy.recent_e2e_latencies)
    elif proxy.routing_algorithm == "fisher_jenks_sqf":
        base_algo_name = proxy.slm_base_algorithm.upper()
        stats["slm_active"] = proxy.slm_active
        stats["slm_mode"] = "FJ-SQF" if proxy.slm_active else base_algo_name
        stats["slm_base_algorithm"] = proxy.slm_base_algorithm
        stats["slm_mode_switches"] = proxy.slm_mode_switches
        stats["slm_window_size"] = proxy.slm_window_size
        stats["slm_activation_threshold"] = proxy.slm_activation_threshold
        stats["slm_deactivation_threshold"] = proxy.slm_deactivation_threshold
        stats["slm_prediction_stats"] = proxy.slm_prediction_stats
        # Fisher-Jenks 전용 통계
        stats["fj_candidate_method"] = proxy.fj_candidate_method
        stats["fj_window_size"] = proxy.fj_window_size
        stats["fj_min_samples"] = proxy.fj_min_samples
        stats["fj_default_threshold"] = proxy.fj_default_threshold
        stats["fj_current_split_point"] = proxy.fj_current_split_point
        stats["fj_buffer_size"] = len(proxy.fj_diff_buffer)
        stats["fj_split_history"] = proxy.fj_split_history[-10:]
        stats["fj_stats"] = proxy.fj_stats
        if proxy.fj_candidate_method == "percent_of_min":
            stats["fj_percent_threshold"] = proxy.fj_percent_threshold
        elif proxy.fj_candidate_method == "top_n":
            stats["fj_top_n"] = proxy.fj_top_n
        elif proxy.fj_candidate_method == "ewma":
            stats["fj_ewma_alpha"] = proxy.fj_ewma_alpha
            stats["fj_ewma_multiplier"] = proxy.fj_ewma_multiplier
            stats["fj_ewma_value"] = proxy.fj_ewma_value
        elif proxy.fj_candidate_method == "fixed_threshold":
            stats["fj_fixed_threshold"] = proxy.fj_fixed_threshold
        if len(proxy.recent_e2e_latencies) > 0:
            stats["recent_avg_e2e"] = sum(proxy.recent_e2e_latencies) / len(proxy.recent_e2e_latencies)
            stats["recent_e2e_window"] = len(proxy.recent_e2e_latencies)
    
    return stats


@app.get("/metrics/current")
async def get_current_metrics():
    """현재 모든 백엔드 서버의 메트릭 조회"""
    try:
        all_metrics = await proxy._collect_all_metrics()
        return {
            "timestamp": datetime.now().isoformat(),
            "servers": all_metrics
        }
    except Exception as e:
        logger.error(f"Error fetching current metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/history")
async def get_metrics_history(limit: int = 100):
    """CSV 파일에서 메트릭 히스토리 조회"""
    try:
        csv_path = Path(proxy.metrics_csv_path)
        if not csv_path.exists():
            return {"message": "No metrics history available", "data": []}
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # 최근 limit개만 반환
        recent_rows = rows[-limit:] if len(rows) > limit else rows
        
        return {
            "total_records": len(rows),
            "returned_records": len(recent_rows),
            "data": recent_rows
        }
    except Exception as e:
        logger.error(f"Error reading metrics history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/finalize")
async def finalize_experiment(request: Request):
    """
    클라이언트로부터 실험 완료 신호를 받아 진행 중인 요청을 완료한 후 메트릭 로그를 저장
    """
    global _emergency_save_done
    _emergency_save_done = True  # finalize 완료 → 시그널 핸들러에서 중복 저장 방지
    try:
        # 요청 본문 파싱 (클라이언트 정보 수신)
        try:
            client_info = await request.json()
        except:
            client_info = {}
        
        client_id = client_info.get("client_id", "unknown")
        experiment_name = client_info.get("experiment_name", "experiment")
        total_requests = client_info.get("total_requests", 0)
        timeout = client_info.get("timeout", 300.0)  # 기본 5분
        qps = client_info.get("qps", None)  # QPS 값 (옵션)
        
        logger.info("=" * 70)
        logger.info(f"[Finalize] Received completion signal from client: {client_id}")
        logger.info(f"  Experiment: {experiment_name}")
        logger.info(f"  Total requests sent: {total_requests}")
        logger.info(f"  QPS: {qps if qps else 'N/A'}")
        logger.info(f"  Timeout: {timeout}s")
        
        # 종료 플래그 설정 (새 요청 거부)
        proxy.shutting_down = True
        logger.info(f"[Finalize] Shutting down flag set. No new requests will be accepted.")
        
        # 현재 활성 요청 수 확인
        current_active = proxy.active_requests
        logger.info(f"[Finalize] Current active requests: {current_active}")
        
        if current_active > 0:
            logger.info(f"[Finalize] Waiting for {current_active} active requests to complete...")
            logger.info(f"[Finalize] Maximum wait time: {timeout}s")
            
            # 모든 활성 요청이 완료될 때까지 대기
            completed = await proxy.wait_for_all_requests(timeout=timeout)
            
            if completed:
                logger.info(f"[Finalize] All active requests completed successfully!")
            else:
                logger.warning(f"[Finalize] Timeout! {proxy.active_requests} requests still active after {timeout}s")
        else:
            logger.info(f"[Finalize] No active requests. Proceeding immediately.")
        
        # 현재 메트릭 CSV 파일 경로
        source_csv = Path(proxy.metrics_csv_path)
        
        if not source_csv.exists():
            logger.warning(f"[Finalize] Metrics CSV file not found: {source_csv}")
            proxy.shutting_down = False  # 플래그 해제
            return {
                "status": "warning",
                "message": "No metrics file to save",
                "active_requests_at_timeout": proxy.active_requests,
                "timestamp": datetime.now().isoformat()
            }
        
        algorithm_name = proxy.routing_algorithm
        logger.info(f"[Finalize] Current algorithm: {algorithm_name}")

        threshold_suffix = ""
        if algorithm_name == "slm_adaptive":
            act_threshold = getattr(proxy, 'slm_activation_threshold', 15.0)
            deact_threshold = getattr(proxy, 'slm_deactivation_threshold', 10.0)
            threshold_suffix = f"_act{act_threshold}_deact{deact_threshold}"
            logger.info(f"[Finalize] Threshold suffix: {threshold_suffix}")
        
        elif algorithm_name == "fisher_jenks_sqf":
            act_threshold = getattr(proxy, 'slm_activation_threshold', 15.0)
            deact_threshold = getattr(proxy, 'slm_deactivation_threshold', 10.0)
            method = getattr(proxy, 'fj_candidate_method', 'fisher_jenks')
            
            if method == "fisher_jenks":
                fj_win = getattr(proxy, 'fj_window_size', 50)
                fj_min = getattr(proxy, 'fj_min_samples', 15)
                fj_def = getattr(proxy, 'fj_default_threshold', 0.0)
                threshold_suffix = f"_act{act_threshold}_deact{deact_threshold}_fjw{fj_win}_fjm{fj_min}_fjd{fj_def}"
            elif method == "percent_of_min":
                pct = getattr(proxy, 'fj_percent_threshold', 0.09)
                threshold_suffix = f"_act{act_threshold}_deact{deact_threshold}_pctmin{pct}"
            elif method == "top_n":
                n = getattr(proxy, 'fj_top_n', 2)
                threshold_suffix = f"_act{act_threshold}_deact{deact_threshold}_topn{n}"
            elif method == "ewma":
                alpha = getattr(proxy, 'fj_ewma_alpha', 0.2)
                mult = getattr(proxy, 'fj_ewma_multiplier', 1.4)
                threshold_suffix = f"_act{act_threshold}_deact{deact_threshold}_ewma{alpha}_m{mult}"
            elif method == "fixed_threshold":
                ft = getattr(proxy, 'fj_fixed_threshold', 10.8)
                threshold_suffix = f"_act{act_threshold}_deact{deact_threshold}_fixed{ft}"
            else:
                threshold_suffix = f"_act{act_threshold}_deact{deact_threshold}_{method}"
            
            logger.info(f"[Finalize] Fisher-Jenks suffix: {threshold_suffix} (method={method})")
        
        elif algorithm_name == "weighted_round_robin":
            # WRR의 경우 weight 비율 추가
            weights = getattr(proxy, 'routing_weights', None)
            if weights:
                # weight를 문자열로 변환 (예: [1, 2, 3] -> "1_2_3")
                weight_str = "_".join(str(int(w)) for w in weights)
                threshold_suffix = f"_w{weight_str}"
                logger.info(f"[Finalize] WRR weights suffix: {threshold_suffix}")
        
        # 결과 저장 디렉토리 설정
        num_gpus = len(proxy.backend_servers)
        if algorithm_name == "fisher_jenks_sqf":
            base_results_dir = Path(__file__).parent / "fj_ablation_study"
        else:
            base_results_dir = Path(__file__).parent / f"regressor_results_{num_gpus}gpu"
        
        # 데이터셋 하위 폴더 (finalize 요청에서 받거나, set_algorithm 시 저장된 값 사용)
        dataset = client_info.get("dataset", None) or getattr(proxy, 'current_dataset', None)
        if dataset:
            base_results_dir = base_results_dir / f"used_{dataset}"
            logger.info(f"[Finalize] Dataset: {dataset} → results dir: {base_results_dir}")
        
        # total_requests를 파일명에 추가
        req_suffix = f"_req{total_requests}" if total_requests > 0 else ""
        logger.info(f"[Finalize] Request suffix for filename: {req_suffix}")
        
        if qps:
            # QPS와 알고리즘명으로 폴더 생성
            results_dir = base_results_dir / f"QPS{qps}_{algorithm_name}{threshold_suffix}"
            base_filename = f"results_QPS{qps}_{algorithm_name}{threshold_suffix}{req_suffix}"
        else:
            # QPS 없으면 알고리즘명만으로 폴더 생성
            results_dir = base_results_dir / f"{algorithm_name}{threshold_suffix}"
            base_filename = f"results_{algorithm_name}{threshold_suffix}{req_suffix}"
        
        # 폴더가 없으면 생성
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 충돌 확인 및 번호 붙이기 (맨 뒤에 넘버링)
        extension = source_csv.suffix
        new_filename = f"{base_filename}{extension}"
        dest_csv = results_dir / new_filename
        
        # 이미 파일이 존재하면 맨 뒤에 번호 붙이기 (..._req1000_2.csv, ..._req1000_3.csv)
        counter = 2
        while dest_csv.exists():
            # 맨 뒤에 _{counter} 형태로 추가
            new_filename = f"{base_filename}_{counter}{extension}"
            dest_csv = results_dir / new_filename
            counter += 1
        
        # base_filename도 업데이트 (summary, load_analysis 파일명에 사용)
        if counter > 2:
            base_filename = f"{base_filename}_{counter-1}"
            logger.info(f"[Finalize] File conflict detected. Using numbered filename: {base_filename}")
        
        # 파일 복사 (원본은 유지)
        shutil.copy2(source_csv, dest_csv)
        
        # 저장된 레코드 수 계산
        with open(dest_csv, 'r') as f:
            reader = csv.reader(f)
            record_count = sum(1 for row in reader) - 1  # 헤더 제외
        
        logger.info(f"[Finalize] Metrics saved to: {dest_csv}")
        logger.info(f"[Finalize] Total records: {record_count}")
        
        # ===== Summary 자동 생성 =====
        summary_filename = f"{base_filename}_summary.csv"
        summary_path = results_dir / summary_filename
        try:
            logger.info(f"[Finalize] Generating summary...")
            generate_summary_csv(
                str(dest_csv),
                str(summary_path),
                proxy.backend_servers,
                proxy.gpu_types
            )
            logger.info(f"[Finalize] Summary saved to: {summary_path}")
        except Exception as e:
            logger.error(f"[Finalize] Error generating summary: {e}")
        
        # ===== Diagnostics CSV 저장 =====
        diagnostics_filename = f"{base_filename}_diagnostics.csv"
        diagnostics_path = results_dir / diagnostics_filename
        try:
            logger.info(f"[Finalize] Saving diagnostics...")
            proxy._log_diag_summary()
            proxy.save_diagnostics_csv(str(diagnostics_path))
            logger.info(f"[Finalize] Diagnostics saved to: {diagnostics_path}")
        except Exception as e:
            logger.error(f"[Finalize] Error saving diagnostics: {e}")
        
        # 원본 CSV 파일 초기화 (다음 실험을 위해)
        proxy.reset_csv_file()
        logger.info(f"[Finalize] Original CSV file reset for next experiment")
        
        # 진단 데이터 초기화 (다음 실험을 위해)
        proxy.reset_diagnostics()
        logger.info(f"[Finalize] Diagnostics reset for next experiment")
        
        # 요청 카운터 초기화 (다음 실험은 request_id 1부터 시작)
        previous_count = proxy.request_count
        proxy.request_count = 0
        logger.info(f"[Finalize] Request counter reset: {previous_count} -> 0")
        
        logger.info(f"[Finalize] Finalization complete!")
        logger.info("=" * 70)
        
        # 종료 플래그 해제 (다음 실험을 위해)
        proxy.shutting_down = False
        
        return {
            "status": "success",
            "message": "All requests completed, metrics saved, analysis completed, and CSV reset for next experiment",
            "algorithm": algorithm_name,
            "qps": qps,
            "saved_files": {
                "metrics": new_filename,
                "summary": summary_filename,
                "diagnostics": diagnostics_filename
            },
            "save_directory": str(results_dir),
            "record_count": record_count,
            "client_id": client_id,
            "experiment_name": experiment_name,
            "total_requests_sent": total_requests,
            "total_requests_processed": previous_count,
            "active_requests_at_completion": proxy.active_requests,
            "completed_gracefully": proxy.active_requests == 0,
            "csv_reset": True,
            "diagnostics_reset": True,
            "next_request_id_starts_from": 1,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[Finalize] Error during finalization: {str(e)}")
        proxy.shutting_down = False  # 에러 발생 시 플래그 해제
        raise HTTPException(status_code=500, detail=f"Failed to finalize: {str(e)}")


_emergency_save_done = False

def _do_emergency_save_sync():
    """
    signal/atexit에서 호출되는 동기 emergency_save 래퍼.
    중복 실행 방지 플래그 사용.
    """
    global _emergency_save_done
    if _emergency_save_done:
        return
    _emergency_save_done = True

    try:
        if proxy.shutting_down:
            return
        if proxy.request_count <= 0:
            return
        logger.warning("[SIGNAL-SAVE] Performing emergency save before exit...")
        proxy.emergency_save()
    except Exception as e:
        try:
            logger.error(f"[SIGNAL-SAVE] Emergency save failed: {e}")
        except Exception:
            pass


def _signal_handler(signum, frame):
    """첫 번째 SIGINT/SIGTERM을 잡아서 emergency_save 후 종료."""
    sig_name = signal.Signals(signum).name
    try:
        logger.warning(f"[SIGNAL] Received {sig_name}. Running emergency save...")
    except Exception:
        pass

    _do_emergency_save_sync()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def main():
    """서버 실행"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_do_emergency_save_sync)

    uvicorn.run(
        app,
        host=UVICORN_CONFIG["host"],
        port=UVICORN_CONFIG["port"],
        workers=UVICORN_CONFIG["workers"],
        log_level=UVICORN_CONFIG["log_level"],
        limit_concurrency=UVICORN_CONFIG["limit_concurrency"],
        backlog=UVICORN_CONFIG["backlog"],
        timeout_keep_alive=UVICORN_CONFIG["timeout_keep_alive"],
    )


if __name__ == "__main__":
    main()
