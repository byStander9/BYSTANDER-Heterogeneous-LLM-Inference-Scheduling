#!/usr/bin/env python3
"""
=== Proxy Server로 QPS 제어 요청 전송 도구 ===

이 스크립트는 ShareGPT/LMSYS 대화 데이터를 프록시 서버로 스트리밍 방식으로 전송합니다.
QPS(초당 요청 수)를 설정하여 부하를 조절할 수 있습니다.

주요 기능:
- QPS 기반 요청 전송 속도 제어 (Poisson 프로세스 기반 유동적 간격)
  * 요청 간격이 지수 분포를 따라 유동적으로 변화
  * 평균 QPS는 설정한 값으로 유지
  * 실제 트래픽 패턴을 더 현실적으로 시뮬레이션
- 동적 QPS 범위 지원 (예: 12-18)
  * 지정된 범위 내에서 주기적으로 QPS 변경
  * 가변적인 트래픽 부하 시뮬레이션
- 동시 요청 수 제한
- 스트리밍 응답 처리 (SSE)
- 실시간 메트릭 수집 및 저장
- 중간 저장 기능 (Ctrl+C로 중단 가능)

사용법:
    # 기본 실행 (QPS=10, 총 1000개 요청)
    python proxy_request_qps.py
    
    # QPS 20으로 2000개 요청 전송
    python proxy_request_qps.py --qps 20 --total 2000
    
    # 동적 QPS (범위 지정): 12~18 사이에서 변화
    python proxy_request_qps.py --qps 12-18 --total 2000
    
    # 프록시 서버 주소 지정
    python proxy_request_qps.py --proxy-host YOUR_PROXY_HOST --proxy-port 20013
    
    # 동시 요청 수 제한 (최대 100개 동시 실행)
    python proxy_request_qps.py --qps 50 --max-concurrent 100
    
    # 특정 인덱스부터 시작
    python proxy_request_qps.py --start-index 1000 --total 500
    
    # 결과를 CSV로 저장
    python proxy_request_qps.py --output results.csv

주요 옵션:
    --qps               초당 요청 수 (기본값: 10) 또는 범위 (예: 12-18)
    --qps-change-interval  동적 QPS 변경 주기 (초, 기본값: 30)
    --max-concurrent    최대 동시 요청 수 (기본값: 50)
    --total             총 전송할 요청 수 (기본값: 1000)
    --start-index       시작 인덱스 (기본값: 0)
    --proxy-host        프록시 서버 호스트 (기본값: 127.0.0.1)
    --proxy-port        프록시 서버 포트 (기본값: 20013)
    --algorithm         라우팅 알고리즘 (1:RR, 2:WRR, 3:SQF, 4:SLM, 5:FJ_SQF)
    --fj-window-size    FJ_SQF diff 슬라이딩 윈도우 크기 (기본값: 50)
    --fj-min-samples    FJ_SQF 최소 샘플 수 (기본값: 15)
    --fj-default-threshold  FJ_SQF 기본 split point (기본값: 0.0s)
    --dataset           데이터셋 파일 경로 또는 sharegpt/lmsys 별칭
    --sharegpt          --dataset의 하위 호환 별칭
    --output            결과 저장 파일 (.xlsx 또는 .csv)
    --model             모델 이름
    --temperature       생성 온도 (기본값: 1.0)
    --max-turns         최대 대화 턴 수 (기본값: 3)

예제:
    # 낮은 QPS로 테스트
    python proxy_request_qps.py --qps 5 --total 100 --output test.xlsx
    
    # WRR 알고리즘으로 실험
    python proxy_request_qps.py --algorithm 2 --qps 10 --total 1000 --output results_wrr.xlsx
    
    # SLM Adaptive 모드
    python proxy_request_qps.py --algorithm 4 --qps 10 --total 1000 --output results_slm.xlsx

    # 특정 구간만 테스트
    python proxy_request_qps.py --start-index 5000 --total 1000 --qps 20
"""

import asyncio
import httpx
import json
import csv
import time
import argparse
import os
import signal
import re
import random
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import pandas as pd
from datetime import datetime

# 전역 변수
shutdown_requested = False
actual_http_sent_count = 0  # 실제 HTTP 요청이 서버로 전송된 수 (세마포어 획득 후)

# 알고리즘 매핑 (숫자 → 알고리즘 이름)
ALGORITHM_MAP = {
    1: "round_robin",
    2: "weighted_round_robin",
    3: "shortest_queue_first",
    4: "slm_adaptive",
    5: "fisher_jenks_sqf"
}

CLIENT_DIR = Path(__file__).resolve().parent
DATASET_ALIASES = {
    "sharegpt": "sharegpt_shuffled.json",
    "lmsys": "lmsys_english_shuffled.json",
    "lmsys-chat-1m": "lmsys_english_shuffled.json",
}

DEFAULT_CONFIG = {
    "dataset": os.getenv("BYSTANDER_DATASET", "sharegpt"),
    "base_url": os.getenv("BYSTANDER_BASE_URL"),
    "proxy_host": os.getenv("PROXY_HOST", "127.0.0.1"),
    "proxy_port": int(os.getenv("PROXY_PORT", "8012")),
    "qps": 10,
    "max_concurrent": 500,
    "start_index": 0,
    "total": 1000,
    "model": os.getenv("BYSTANDER_MODEL",
                       "Meta-Llama-3.1-8B-Instruct-AWQ-INT4"),
    "temperature": 1.0,
    "max_tokens": None,
    "max_turns": 3,
    "output": os.getenv("BYSTANDER_OUTPUT",
                        "results/proxy_experiment_results.xlsx"),
    "client_timeout": 600.0,
}

def parse_qps(qps_str: str) -> tuple:
    """
    QPS 문자열 파싱
    
    Args:
        qps_str: QPS 문자열 (예: "10" 또는 "12-18")
    
    Returns:
        (is_range, min_qps, max_qps): 
            - is_range: 범위 여부
            - min_qps: 최소 QPS (고정 QPS인 경우 해당 값)
            - max_qps: 최대 QPS (고정 QPS인 경우 None)
    """
    qps_str = str(qps_str).strip()
    
    if '-' in qps_str:
        # 범위 형식: "12-18"
        parts = qps_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"잘못된 QPS 범위 형식: {qps_str}. '최소-최대' 형식을 사용하세요.")
        
        min_qps = float(parts[0].strip())
        max_qps = float(parts[1].strip())
        
        if min_qps <= 0 or max_qps <= 0:
            raise ValueError(f"QPS는 양수여야 합니다: {qps_str}")
        if min_qps >= max_qps:
            raise ValueError(f"최소 QPS가 최대 QPS보다 작아야 합니다: {qps_str}")
        
        return (True, min_qps, max_qps)
    else:
        # 고정 QPS: "10"
        qps = float(qps_str)
        if qps <= 0:
            raise ValueError(f"QPS는 양수여야 합니다: {qps_str}")
        return (False, qps, None)


def qps_argument(value: str) -> str:
    try:
        parse_qps(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 정수여야 합니다.")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수여야 합니다.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 숫자여야 합니다.")
    return parsed


def build_chat_completions_url(args) -> str:
    if args.base_url:
        return f"{args.base_url.rstrip('/')}/v1/chat/completions"
    return f"http://{args.proxy_host}:{args.proxy_port}/v1/chat/completions"

def get_next_qps(min_qps: float, max_qps: float) -> float:
    """
    범위 내에서 다음 QPS 값 생성 (균등 분포)
    
    Args:
        min_qps: 최소 QPS
        max_qps: 최대 QPS
    
    Returns:
        범위 내의 무작위 QPS 값
    """
    return random.uniform(min_qps, max_qps)

@dataclass
class RequestResult:
    request_id: int
    http_status: int
    start_time: str
    end_time: str
    latency_e2e_ms: float
    latency_ttft_ms: float
    tokens_generated: int
    prompt_tokens: int
    prompt: str
    error: Optional[str]

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n중단 신호를 받았습니다. 진행 중인 요청들을 완료하고 종료합니다...")
    shutdown_requested = True

def clean_text(text: str) -> str:
    """XML/CSV 호환되지 않는 문자 제거"""
    if isinstance(text, str):
        # ASCII 제어 문자 제거 (탭, 개행, 캐리지 리턴은 허용)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # 유니코드 Surrogates 및 비문자 제거
        text = re.sub(r'[\uD800-\uDFFF\uFFFE\uFFFF]', '', text)
        return text
    return text

def resolve_dataset_path(file_path: str,
                         dataset_dir: Optional[str] = None) -> Path:
    """Resolve an explicit path or a ShareGPT/LMSYS dataset alias."""
    requested = Path(file_path).expanduser()
    dataset_name = DATASET_ALIASES.get(file_path.lower(), file_path)
    candidates = [requested]

    search_dirs = []
    if dataset_dir:
        search_dirs.append(Path(dataset_dir).expanduser())
    if env_dir := os.getenv("BYSTANDER_DATASET_DIR"):
        search_dirs.append(Path(env_dir).expanduser())
    search_dirs.extend([Path.cwd(), CLIENT_DIR])

    candidates.extend(directory / dataset_name for directory in search_dirs)

    checked = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_file():
            return resolved

    checked_paths = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"데이터셋 파일을 찾을 수 없습니다: {file_path}\n"
        f"확인한 경로:\n  - {checked_paths}\n"
        "직접 경로를 지정하거나 BYSTANDER_DATASET_DIR을 설정하세요.")


def iter_dataset_records(file_path: Path):
    """Yield JSON-array or JSONL records without loading the file as text."""
    with file_path.open("r", encoding="utf-8") as source:
        first_char = ""
        while char := source.read(1):
            if not char.isspace():
                first_char = char
                break

    if first_char == "[":
        try:
            import ijson
        except ImportError:
            with file_path.open("r", encoding="utf-8") as source:
                yield from json.load(source)
        else:
            with file_path.open("rb") as source:
                yield from ijson.items(source, "item")
        return

    with file_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                yield None
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"경고: JSONL {line_number}행을 건너뜁니다: {exc}")
                yield None


def load_dataset_auto(file_path: str,
                      start_index: int = 0,
                      limit: int = None,
                      dataset_dir: Optional[str] = None
                      ) -> List[Dict[str, Any]]:
    """데이터셋 로드 (ShareGPT / LMSYS 형식 자동 감지, JSON/JSONL 자동 감지)
    
    지원 형식:
    - ShareGPT: {"conversations": [{"from": "human/gpt", "value": "..."}]}
    - LMSYS:    {"conversation": [{"role": "user/assistant", "content": "..."}]}
    """
    resolved_path = resolve_dataset_path(file_path, dataset_dir=dataset_dir)
    print(f"데이터셋 로드 중: {resolved_path}")

    items = []
    for index, item in enumerate(iter_dataset_records(resolved_path)):
        if index < start_index:
            continue
        if not isinstance(item, dict):
            continue
        if item.get("conversations") or item.get("conversation"):
            items.append(item)
            if limit is not None and len(items) >= limit:
                break
    
    # 데이터셋 형식 감지 및 출력
    if items:
        if items[0].get("conversations"):
            dataset_type = "ShareGPT"
        elif items[0].get("conversation"):
            dataset_type = "LMSYS"
        else:
            dataset_type = "Unknown"
        print(f"데이터셋 형식: {dataset_type}")
    
    print(f"데이터셋 로드 완료: {len(items)}개 대화")
    return items

def convert_to_openai_messages(conversations: List[Dict[str, str]], max_turns: int = 10) -> List[Dict[str, str]]:
    """대화를 OpenAI 형식으로 변환 (ShareGPT / LMSYS 형식 자동 감지)
    
    지원 형식:
    - ShareGPT: {"from": "human/gpt", "value": "..."}
    - LMSYS:    {"role": "user/assistant", "content": "..."}
    """
    messages = []
    for conv in conversations:
        # ShareGPT 형식 (from/value) 또는 LMSYS 형식 (role/content) 모두 지원
        role = conv.get("from", conv.get("role", "")).lower()
        content = conv.get("value", conv.get("content", ""))
        
        if role in ("human", "user"):
            messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})
    
    # 마지막이 user 메시지로 끝나도록 조정
    if messages and messages[-1]["role"] != "user":
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages = messages[:i + 1]
                break
    
    # 최대 턴 수 제한
    if len(messages) > max_turns:
        messages = messages[-max_turns:]
    
    return messages

async def send_set_algorithm(
    proxy_host: str,
    proxy_port: int,
    algorithm: str,
    dataset: str,
    weights: Optional[List[int]] = None,
    qps: Optional[float] = None,
    slm_activation_threshold: Optional[float] = None,
    slm_deactivation_threshold: Optional[float] = None,
    fj_window_size: Optional[int] = None,
    fj_min_samples: Optional[int] = None,
    fj_default_threshold: Optional[float] = None,
    fj_preset: Optional[str] = None,
    timeout: float = 10.0
) -> bool:
    """프록시 서버의 라우팅 알고리즘 설정"""
    url = f"http://{proxy_host}:{proxy_port}/set_algorithm"

    payload = {
        "algorithm": algorithm,
        "dataset": dataset
    }

    if algorithm == "weighted_round_robin":
        if weights is not None and len(weights) > 0:
            payload["weights"] = weights
        elif qps is not None:
            payload["qps"] = float(qps)
    else:
        if weights is not None and len(weights) > 0:
            payload["weights"] = weights

    if slm_activation_threshold is not None:
        payload["slm_activation_threshold"] = slm_activation_threshold
    if slm_deactivation_threshold is not None:
        payload["slm_deactivation_threshold"] = slm_deactivation_threshold

    # Fisher-Jenks SQF 설정 (fisher_jenks_sqf일 때)
    if fj_window_size is not None:
        payload["fj_window_size"] = fj_window_size
    if fj_min_samples is not None:
        payload["fj_min_samples"] = fj_min_samples
    if fj_default_threshold is not None:
        payload["fj_default_threshold"] = fj_default_threshold
    if fj_preset is not None:
        payload["fj_preset"] = fj_preset
    
    try:
        print(f"\n{'='*60}")
        print(f"[알고리즘 설정] 프록시 서버 라우팅 알고리즘 변경...")
        print(f"  URL: {url}")
        print(f"  Algorithm: {algorithm}")
        if algorithm == "weighted_round_robin":
            if weights is not None and len(weights) > 0:
                print(f"  Weights: {weights} (수동 지정)")
            elif qps is not None:
                print(f"  QPS: {qps} (서버 자동 weights 결정)")
        if slm_activation_threshold is not None:
            print(f"  SLM Activation Threshold: {slm_activation_threshold}s")
        if slm_deactivation_threshold is not None:
            print(f"  SLM Deactivation Threshold: {slm_deactivation_threshold}s")
        if fj_window_size is not None:
            print(f"  FJ Window Size: {fj_window_size}")
        if fj_min_samples is not None:
            print(f"  FJ Min Samples: {fj_min_samples}")
        if fj_default_threshold is not None:
            print(f"  FJ Default Threshold: {fj_default_threshold}s")
        if fj_preset is not None:
            print(f"  FJ Preset: {fj_preset}")
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✓ 알고리즘 설정 성공!")
                print(f"  이전 알고리즘: {result.get('previous_algorithm', 'N/A')}")
                print(f"  새 알고리즘: {result.get('new_algorithm', 'N/A')}")
                print(f"{'='*60}\n")
                return True
            else:
                print(f"\n✗ 알고리즘 설정 실패: HTTP {response.status_code}")
                print(f"  {response.text[:200]}")
                print(f"{'='*60}\n")
                return False
    except Exception as e:
        print(f"\n✗ 알고리즘 설정 오류: {str(e)}")
        print(f"  프록시 서버가 실행 중인지 확인하세요.")
        print(f"{'='*60}\n")
        return False


async def send_finalize_signal(
    proxy_host: str,
    proxy_port: int,
    client_id: str,
    experiment_name: str,
    total_requests: int,
    qps,  # float 또는 str (동적 QPS의 경우 "min-max" 형식)
    timeout: float = 10.0
) -> bool:
    """
    프록시 서버에 실험 완료 신호를 전송하여 메트릭 로그 저장을 요청
    """
    finalize_url = f"http://{proxy_host}:{proxy_port}/finalize"
    
    payload = {
        "client_id": client_id,
        "experiment_name": experiment_name,
        "total_requests": total_requests,
        "qps": qps
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"[완료 신호 전송] 프록시 서버에 실험 완료를 알립니다...")
        print(f"  URL: {finalize_url}")
        print(f"  Client ID: {client_id}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Total Requests: {total_requests}")
        # QPS 출력 (고정 또는 동적)
        if isinstance(qps, str):
            print(f"  QPS: {qps} (동적 범위)")
        else:
            print(f"  QPS: {qps:.2f}")
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(finalize_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✓ 프록시 서버 응답:")
                print(f"  상태: {result.get('status', 'unknown')}")
                print(f"  메시지: {result.get('message', 'N/A')}")
                if 'saved_file' in result:
                    print(f"  저장된 파일: {result['saved_file']}")
                    print(f"  레코드 수: {result.get('record_count', 'N/A')}")
                print(f"{'='*60}\n")
                return True
            else:
                print(f"\n✗ 프록시 서버 응답 오류: HTTP {response.status_code}")
                print(f"  {response.text[:200]}")
                print(f"{'='*60}\n")
                return False
                
    except Exception as e:
        print(f"\n✗ 완료 신호 전송 실패: {str(e)}")
        print(f"  프록시 서버가 실행 중인지 확인하세요.")
        print(f"{'='*60}\n")
        return False


def restart_k8s_daemonsets(
    master_host: str,
    namespace: str,
    daemonset_names: List[str],
    ssh_user: str = "root",
    ssh_port: int = 22,
    wait_time: int = 30
) -> bool:
    """
    쿠버네티스 데몬셋 재시작
    
    Args:
        master_host: 쿠버네티스 마스터 노드 IP/호스트명
        namespace: 네임스페이스
        daemonset_names: 재시작할 데몬셋 이름 리스트
        ssh_user: SSH 사용자 (기본값: root)
        ssh_port: SSH 포트 (기본값: 22)
        wait_time: 재시작 후 대기 시간 (초)
    
    Returns:
        성공 여부
    """
    try:
        print(f"\n{'='*60}")
        print(f"[K8s 데몬셋 재시작] 시작...")
        print(f"  Master Host: {master_host}")
        print(f"  SSH Port: {ssh_port}")
        print(f"  Namespace: {namespace}")
        print(f"  Daemonsets: {', '.join(daemonset_names)}")
        print(f"  SSH User: {ssh_user}")
        print(f"{'='*60}\n")
        
        for ds_name in daemonset_names:
            print(f"[{ds_name}] 재시작 중...")
            
            # kubectl rollout restart 명령 실행 (포트 포함)
            cmd = f"ssh -p {ssh_port} {ssh_user}@{master_host} 'kubectl rollout restart daemonset/{ds_name} -n {namespace}'"
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"  ✓ {ds_name} 재시작 명령 성공")
                print(f"    {result.stdout.strip()}")
            else:
                print(f"  ✗ {ds_name} 재시작 실패")
                print(f"    오류: {result.stderr.strip()}")
                return False
        
        # 재시작 완료 대기
        print(f"\n[대기] 데몬셋 재시작 완료까지 {wait_time}초 대기...")
        time.sleep(wait_time)
        
        # 파드 상태 확인
        print(f"\n[상태 확인] 데몬셋 파드 상태...")
        for ds_name in daemonset_names:
            cmd = f"ssh -p {ssh_port} {ssh_user}@{master_host} 'kubectl get pods -n {namespace} -l app={ds_name} -o wide'"
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"\n[{ds_name}] 파드 상태:")
                print(result.stdout.strip())
            else:
                print(f"  ⚠ {ds_name} 상태 확인 실패 (계속 진행)")
        
        print(f"\n{'='*60}")
        print(f"✓ 데몬셋 재시작 완료!")
        print(f"{'='*60}\n")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n✗ 데몬셋 재시작 타임아웃")
        print(f"{'='*60}\n")
        return False
    except Exception as e:
        print(f"\n✗ 데몬셋 재시작 오류: {str(e)}")
        print(f"{'='*60}\n")
        return False


async def send_streaming_request(
    client: httpx.AsyncClient,
    proxy_url: str,
    request_id: int,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    semaphore: asyncio.Semaphore,
    timeout_seconds: float = 600.0,
) -> RequestResult:
    """프록시 서버로 스트리밍 요청 전송"""
    global shutdown_requested, actual_http_sent_count
    
    if shutdown_requested:
        return RequestResult(
            request_id=request_id,
            http_status=0,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            latency_e2e_ms=0,
            latency_ttft_ms=0,
            tokens_generated=0,
            prompt_tokens=0,
            prompt=json.dumps(messages, ensure_ascii=False),
            error="Request cancelled"
        )
    
    start_time_wall = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_wall))
    start_time = time.perf_counter()
    ttft_time = None
    tokens_generated = 0
    prompt_tokens = 0
    http_status = 0
    error = None
    
    request_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "top_p": 0.95,
        "presence_penalty": 0.1,
    }
    if max_tokens is not None:
        request_payload["max_tokens"] = max_tokens
    
    try:
        async with semaphore:
            actual_http_sent_count += 1  # 세마포어 획득 = 실제 HTTP 전송 시작
            async with client.stream(
                "POST",
                proxy_url,
                json=request_payload,
                timeout=httpx.Timeout(timeout_seconds,
                                      connect=10.0,
                                      read=timeout_seconds,
                                      write=10.0)
            ) as response:
                http_status = response.status_code
                
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            event = json.loads(data_str)
                            choices = event.get("choices", [])
                            
                            if choices:
                                choice = choices[0]
                                delta = choice.get("delta", {})
                                
                                # 첫 토큰 시간 측정 (TTFT)
                                if ttft_time is None and delta.get("content"):
                                    ttft_time = time.perf_counter()
                                
                                # 토큰 수 계산
                                if delta.get("content"):
                                    tokens_generated += 1
                            
                            # 사용량 정보 수집
                            if "usage" in event:
                                usage = event["usage"]
                                tokens_generated = usage.get("completion_tokens", tokens_generated)
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
                else:
                    error = f"HTTP {response.status_code}"
                    try:
                        error_body = await response.aread()
                        error += f" - {error_body.decode('utf-8', errors='ignore')[:200]}"
                    except:
                        pass
    
    except Exception as e:
        error = str(e)
    
    end_time_wall = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time_wall))
    end_time = time.perf_counter()
    
    # 지연시간 계산
    latency_e2e_ms = (end_time - start_time) * 1000
    latency_ttft_ms = (ttft_time - start_time) * 1000 if ttft_time else latency_e2e_ms
    
    return RequestResult(
        request_id=request_id,
        http_status=http_status,
        start_time=start_time_str,
        end_time=end_time_str,
        latency_e2e_ms=round(latency_e2e_ms, 2),
        latency_ttft_ms=round(latency_ttft_ms, 2),
        tokens_generated=tokens_generated,
        prompt_tokens=prompt_tokens,
        prompt=json.dumps(messages, ensure_ascii=False),
        error=error
    )

async def run_experiment(args):
    """QPS 제어 실험 실행"""
    global shutdown_requested, actual_http_sent_count
    shutdown_requested = False
    actual_http_sent_count = 0  # 실험 시작 시 리셋

    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)
    
    # 실험 시작 시간 기록
    experiment_start_time = time.time()
    experiment_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 데이터셋 로드 및 타입 감지 (먼저 수행)
    dataset_arg = getattr(args, "dataset", getattr(args, "sharegpt", None))
    dataset = load_dataset_auto(dataset_arg,
                                start_index=args.start_index,
                                limit=args.total,
                                dataset_dir=getattr(args, "dataset_dir", None))
    if not dataset:
        raise SystemExit("데이터셋이 비어있습니다.")
    
    # 데이터셋 타입 감지
    dataset_type = "sharegpt"  # 기본값
    if dataset:
        if dataset[0].get("conversations"):
            dataset_type = "sharegpt"
        elif dataset[0].get("conversation"):
            dataset_type = "lmsys-chat-1m"

    if getattr(args, "dry_run", False):
        sample = dataset[0].get("conversations") or dataset[0].get(
            "conversation", [])
        messages = convert_to_openai_messages(sample,
                                              max_turns=args.max_turns)
        print("\n드라이런 검증 완료")
        print(f"  데이터셋 형식: {dataset_type}")
        print(f"  선택된 대화 수: {len(dataset)}")
        print(f"  첫 대화 메시지 수: {len(messages)}")
        print(f"  QPS: {args.qps}")
        print("  프록시에는 요청을 보내지 않았습니다.")
        return
    
    # 알고리즘 매핑 (숫자 → 알고리즘 이름)
    if args.algorithm:
        algorithm_name = ALGORITHM_MAP.get(args.algorithm)
        if not algorithm_name:
            print(f"✗ 오류: 잘못된 알고리즘 번호 {args.algorithm}")
            print(f"  사용 가능한 알고리즘:")
            for num, name in ALGORITHM_MAP.items():
                print(f"    {num}: {name}")
            raise SystemExit(1)
        
        # 프록시 서버 알고리즘 설정
        # WRR일 때 weights / qps 전달 결정
        weights_value = None
        qps_value = None
        if args.algorithm == 2:  # WRR
            if args.weights is not None and len(args.weights) > 0:
                # 수동 weights 지정 → weights를 전달
                weights_value = args.weights
            else:
                # weights 미지정 → qps만 전달하여 서버 자동 결정
                is_qps_range_tmp, min_qps_tmp, max_qps_tmp = parse_qps(args.qps)
                qps_value = min_qps_tmp
        
        if args.algorithm == 4:  # SLM Adaptive
            slm_activation_value = getattr(args, 'slm_activation_threshold', 15.0)
            slm_deactivation_value = getattr(args, 'slm_deactivation_threshold', 10.0)
        else:
            slm_activation_value = None
            slm_deactivation_value = None

        success = await send_set_algorithm(
            proxy_host=args.proxy_host,
            proxy_port=args.proxy_port,
            algorithm=algorithm_name,
            dataset=dataset_type,
            weights=weights_value,
            qps=qps_value,
            slm_activation_threshold=slm_activation_value,
            slm_deactivation_threshold=slm_deactivation_value,
            fj_window_size=getattr(args, 'fj_window_size', None),
            fj_min_samples=getattr(args, 'fj_min_samples', None),
            fj_default_threshold=getattr(args, 'fj_default_threshold', None),
            fj_preset=getattr(args, 'fj_preset', None),
        )
        
        if not success:
            print("✗ 알고리즘 설정에 실패했습니다. 실험을 중단합니다.")
            raise SystemExit(1)
    
    # QPS 파싱
    is_qps_range, min_qps, max_qps = parse_qps(args.qps)
    
    if is_qps_range:
        current_qps = get_next_qps(min_qps, max_qps)
        print(f"\n{'='*60}")
        print(f"동적 QPS 모드 활성화")
        print(f"  범위: {min_qps} ~ {max_qps} requests/sec")
        print(f"  변경 주기: {args.qps_change_interval}초")
        print(f"  초기 QPS: {current_qps:.2f} requests/sec")
        print(f"{'='*60}\n")
    else:
        current_qps = min_qps
    
    print(f"\n{'='*60}")
    print(f"실험 시작 시간: {experiment_start_str}")
    if args.base_url:
        print(f"OpenAI 호환 서버: {args.base_url}")
    else:
        print(f"프록시 서버: {args.proxy_host}:{args.proxy_port}")
    if args.algorithm:
        print(f"라우팅 알고리즘: {ALGORITHM_MAP[args.algorithm]} (#{args.algorithm})")
    
    if is_qps_range:
        print(f"QPS 설정: {min_qps}-{max_qps} requests/sec (동적 범위)")
        print(f"  - 현재 QPS: {current_qps:.2f} requests/sec")
        print(f"  - 변경 주기: {args.qps_change_interval}초")
        print(f"  - 요청 간격: 지수 분포 기반 유동적 조절")
    else:
        print(f"QPS 설정: {current_qps} requests/sec (고정)")
        print(f"  - 요청 간격: 지수 분포 기반 유동적 조절")
        print(f"  - 평균 간격: {1.0/current_qps:.3f}초")
    
    print(f"최대 동시 요청: {args.max_concurrent}")
    print(f"총 요청 수: {args.total}")
    print(f"시작 인덱스: {args.start_index}")
    print(f"{'='*60}\n")
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    
    # 프록시 URL 생성
    proxy_url = build_chat_completions_url(args)
    
    actual_total = len(dataset)
    print(f"전송할 요청 수: {actual_total}개\n")
    
    results = []
    active_tasks = set()
    next_request_id = 0
    
    # QPS 제어를 위한 변수 (경과 시간 기반 누적 카운트 방식)
    # 이벤트 루프 지연에 무관하게 정확한 QPS를 보장
    expected_sent_float = 0.0  # 보냈어야 할 누적 개수 (소수점 포함)
    last_qps_check_time = 0    # 마지막 QPS 체크 시간
    
    # 동적 QPS 변경을 위한 변수
    last_qps_change_time = 0
    qps_change_count = 0
    
    try:
        # HTTP 클라이언트 설정
        limits = httpx.Limits(
            max_connections=args.max_concurrent * 2,
            max_keepalive_connections=args.max_concurrent
        )
        timeout = httpx.Timeout(args.client_timeout)
        semaphore = asyncio.Semaphore(args.max_concurrent)
        
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            print("요청 전송을 시작합니다...\n")
            # 실제 요청 전송 시점 기준으로 덮어쓰기 (정확한 qps측정을 위해)
            experiment_start_time = time.time()
            last_qps_change_time = experiment_start_time
            last_qps_check_time = experiment_start_time  # QPS 누적 카운트 시작 시간
            # 진행 상황 추적
            sent_count = 0
            completed_count = 0
            last_progress_time = time.time()
            
            while (next_request_id < actual_total or active_tasks) and not shutdown_requested:
                current_time = time.time()
                
                # 동적 QPS 변경 (범위가 지정된 경우)
                if is_qps_range and (current_time - last_qps_change_time >= args.qps_change_interval):
                    old_qps = current_qps
                    current_qps = get_next_qps(min_qps, max_qps)
                    qps_change_count += 1
                    last_qps_change_time = current_time
                    
                    print(f"\n[QPS 변경 #{qps_change_count}] {old_qps:.2f} → {current_qps:.2f} requests/sec "
                          f"(경과: {current_time - experiment_start_time:.1f}초)\n")
                current_time = time.time()
                
                # 완료된 태스크 정리
                done_tasks = set()
                for t in active_tasks:
                    if t.done():
                        done_tasks.add(t)
                        try:
                            result = t.result()
                            if isinstance(result, RequestResult):
                                results.append(result)
                                completed_count += 1
                                
                                # 1000개 단위로 중간 저장
                                if len(results) > 0 and len(results) % 1000 == 0:
                                    print(f"\n[중간 저장] {len(results)}개 요청 완료. 결과 저장 중...")
                                    save_results(results, args.output, experiment_start_str, 
                                               time.strftime("%Y-%m-%d %H:%M:%S"), 
                                               time.time() - experiment_start_time)
                                    print(f"저장 완료. 계속 진행합니다...\n")
                        except Exception as e:
                            print(f"태스크 결과 처리 오류: {e}")
                
                active_tasks -= done_tasks
                
                # 진행 상황 출력 (5초마다)
                if current_time - last_progress_time >= 5.0:
                    elapsed = current_time - experiment_start_time
                    scheduled_qps = sent_count / elapsed if elapsed > 0 else 0
                    actual_qps = actual_http_sent_count / elapsed if elapsed > 0 else 0
                    print(f"[진행 상황] 예약: {sent_count}/{actual_total}, "
                          f"실전송: {actual_http_sent_count}/{actual_total}, "
                          f"완료: {completed_count}/{actual_total}, "
                          f"실행 중: {len(active_tasks)}, "
                          f"예약QPS: {scheduled_qps:.2f}, "
                          f"실전송QPS: {actual_qps:.2f}")
                    last_progress_time = current_time
                
                # ============================================================
                # QPS 제어: 경과 시간 기반 누적 카운트 방식
                # 이벤트 루프 지연에 무관하게 정확한 QPS 보장
                # ============================================================
                
                # 경과 시간에 비례하여 보내야 할 누적 개수 갱신
                time_delta = current_time - last_qps_check_time
                last_qps_check_time = current_time
                expected_sent_float += time_delta * current_qps
                
                # 보내야 할 개수 계산 (누적 기대값 - 실제 전송 수)
                to_send = int(expected_sent_float) - sent_count
                
                # 실험 마지막 5% 요청은 QPS 무시하고 빠르게 전송
                if next_request_id >= actual_total * 0.95 and len(active_tasks) < args.max_concurrent * 0.5:
                    to_send = max(to_send, actual_total - next_request_id)
                
                # burst 전송 (밀린 만큼 즉시 전송)
                sent_in_this_loop = 0
                max_batch = 100  # 한 루프당 최대 전송 수
                
                while (to_send > 0
                       and next_request_id < actual_total
                       and len(active_tasks) < args.max_concurrent
                       and sent_in_this_loop < max_batch
                       and not shutdown_requested):
                    
                    # 데이터 준비 (ShareGPT: conversations, LMSYS: conversation)
                    data_item = dataset[next_request_id]
                    convs = data_item.get("conversations") or data_item.get("conversation", [])
                    messages = convert_to_openai_messages(
                        convs, 
                        max_turns=args.max_turns
                    )
                    
                    # 요청 태스크 생성
                    task = asyncio.create_task(
                        send_streaming_request(
                            client=client,
                            proxy_url=proxy_url,
                            request_id=args.start_index + next_request_id,
                            messages=messages,
                            model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            semaphore=semaphore,
                            timeout_seconds=getattr(args, "client_timeout",
                                                    600.0),
                        )
                    )
                    active_tasks.add(task)
                    next_request_id += 1
                    sent_count += 1
                    sent_in_this_loop += 1
                    to_send -= 1
                
                # 대기: 다음 전송 시점까지
                if next_request_id < actual_total:
                    if to_send <= 0 and current_qps > 0:
                        # 다음 1개가 필요한 시점까지 대기
                        surplus = expected_sent_float - sent_count  # 0~1 사이
                        remaining = (1.0 - surplus) / current_qps
                        await asyncio.sleep(min(max(remaining, 0), 0.05))
                    else:
                        # 아직 밀린 요청이 있으면 즉시 다음 루프
                        await asyncio.sleep(0)
                else:
                    # 모든 요청 전송 완료, 완료 대기 중
                    await asyncio.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n키보드 인터럽트를 받았습니다.")
    finally:
        if shutdown_requested:
            print("실험이 중단되었습니다.")
    
    # 실험 종료 시간 기록
    experiment_end_time = time.time()
    experiment_end_str = time.strftime("%Y-%m-%d %H:%M:%S")
    experiment_duration = experiment_end_time - experiment_start_time
    
    print(f"\n{'='*60}")
    print(f"실험 종료 시간: {experiment_end_str}")
    print(f"총 실험 시간: {experiment_duration:.2f}초 ({experiment_duration/60:.2f}분)")
    print(f"전송된 요청: {sent_count}개 (목표: {actual_total}개)")
    print(f"완료된 요청: {completed_count}개")
    if sent_count != actual_total:
        print(f"⚠️  전송되지 않은 요청: {actual_total - sent_count}개")
    if completed_count != sent_count:
        print(f"⚠️  실패한 요청: {sent_count - completed_count}개")
    if experiment_duration > 0:
        actual_qps = sent_count / experiment_duration
        print(f"실제 평균 QPS: {actual_qps:.2f}")
    print(f"{'='*60}\n")
    
    # 최종 결과 저장
    if results:
        save_results(results, args.output, experiment_start_str, experiment_end_str, experiment_duration)
        print_summary(results)
        
        # 프록시 서버에 완료 신호 전송
        # 실험 이름 생성 (출력 파일명 기반)
        experiment_name = Path(args.output).stem
        client_id = f"client_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # QPS 정보 준비 (범위인 경우 "min-max", 고정인 경우 숫자)
        qps_info = f"{min_qps}-{max_qps}" if is_qps_range else str(min_qps)
        
        if not args.base_url:
            await send_finalize_signal(
                proxy_host=args.proxy_host,
                proxy_port=args.proxy_port,
                client_id=client_id,
                experiment_name=experiment_name,
                total_requests=len(results),
                qps=qps_info
            )
        
        # 쿠버네티스 데몬셋 재시작 (옵션)
        if hasattr(args, 'k8s_restart') and args.k8s_restart and args.k8s_daemonsets:
            print("\n실험 완료 후 데몬셋 재시작을 시작합니다...")
            restart_success = restart_k8s_daemonsets(
                master_host=args.k8s_master,
                namespace=args.k8s_namespace,
                daemonset_names=args.k8s_daemonsets,
                ssh_user=args.k8s_ssh_user,
                ssh_port=args.k8s_ssh_port,
                wait_time=args.k8s_wait_time
            )
            
            if not restart_success:
                print("⚠ 데몬셋 재시작에 실패했지만 실험은 완료되었습니다.")
    else:
        print("저장할 결과가 없습니다.")

def save_results(results: List[RequestResult], output_file: str, 
                start_time: str, end_time: str, duration: float):
    """결과를 파일로 저장 (Excel 또는 CSV)"""
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)
    if output_file.lower().endswith('.csv'):
        save_results_to_csv(results, output_file, start_time, end_time, duration)
    else:
        save_results_to_excel(results, output_file, start_time, end_time, duration)

def save_results_to_excel(results: List[RequestResult], output_file: str, 
                          start_time: str, end_time: str, duration: float):
    """결과를 Excel 파일로 저장"""
    # 데이터를 DataFrame으로 변환
    data = []
    for result in results:
        row = {
            "request_id": result.request_id,
            "http_status": result.http_status,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "latency_e2e_ms": result.latency_e2e_ms,
            "latency_ttft_ms": result.latency_ttft_ms,
            "tokens_generated": result.tokens_generated,
            "prompt_tokens": result.prompt_tokens,
            "prompt": result.prompt,
            "error": result.error,
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # XML 호환되지 않는 문자 제거
    if hasattr(df, 'map'):
        df = df.map(clean_text)
    else:
        df = df.applymap(clean_text)
    
    # Excel 파일 경로
    excel_file = output_file if output_file.lower().endswith('.xlsx') else output_file + '.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # 메인 데이터 시트
        df.to_excel(writer, sheet_name='실험결과', index=False)
        
        # 실험 정보 시트
        successful_requests = [r for r in results if r.http_status == 200 and not r.error]
        latencies = sorted([r.latency_e2e_ms for r in successful_requests])
        ttft_latencies = sorted([r.latency_ttft_ms for r in successful_requests])
        
        def percentile(values, p):
            if not values:
                return ""
            n = len(values)
            idx = min(n - 1, max(0, int(n * p)))
            return f"{values[idx]:.1f}"
        
        avg_latency = f"{sum(latencies)/len(latencies):.1f}" if latencies else ""
        avg_ttft = f"{sum(ttft_latencies)/len(ttft_latencies):.1f}" if ttft_latencies else ""
        total_tokens = sum(r.tokens_generated for r in successful_requests)
        actual_qps = f"{len(results) / duration:.2f}" if duration > 0 else ""
        
        experiment_info = pd.DataFrame({
            '항목': [
                '실험 시작 시간', '실험 종료 시간', '총 실험 시간(초)', '총 실험 시간(분)',
                '총 요청 수', '성공한 요청 수', '실패한 요청 수',
                '실제 평균 QPS', '총 생성 토큰',
                '평균 E2E 지연(ms)', 'E2E P50(ms)', 'E2E P90(ms)', 'E2E P95(ms)', 'E2E P99(ms)',
                '평균 TTFT(ms)', 'TTFT P50(ms)', 'TTFT P90(ms)', 'TTFT P95(ms)', 'TTFT P99(ms)'
            ],
            '값': [
                start_time, end_time, f"{duration:.2f}", f"{duration/60:.2f}",
                len(results), len(successful_requests), 
                len([r for r in results if r.http_status != 200 or r.error]),
                actual_qps, total_tokens,
                avg_latency, percentile(latencies, 0.5), percentile(latencies, 0.9),
                percentile(latencies, 0.95), percentile(latencies, 0.99),
                avg_ttft, percentile(ttft_latencies, 0.5), percentile(ttft_latencies, 0.9),
                percentile(ttft_latencies, 0.95), percentile(ttft_latencies, 0.99)
            ]
        })
        experiment_info.to_excel(writer, sheet_name='실험정보', index=False)
    
    print(f"결과가 {excel_file}에 저장되었습니다.")

def save_results_to_csv(results: List[RequestResult], output_file: str,
                       start_time: str, end_time: str, duration: float):
    """결과를 CSV 파일로 저장"""
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        
        # 실험 정보 헤더
        writer.writerow(["=== 실험 정보 ==="])
        writer.writerow(["실험 시작 시간", start_time])
        writer.writerow(["실험 종료 시간", end_time])
        writer.writerow(["총 실험 시간(초)", f"{duration:.2f}"])
        writer.writerow(["총 실험 시간(분)", f"{duration/60:.2f}"])
        writer.writerow(["총 요청 수", len(results)])
        
        successful_requests = [r for r in results if r.http_status == 200 and not r.error]
        writer.writerow(["성공한 요청 수", len(successful_requests)])
        writer.writerow(["실패한 요청 수", len([r for r in results if r.http_status != 200 or r.error])])
        
        if duration > 0:
            writer.writerow(["실제 평균 QPS", f"{len(results) / duration:.2f}"])
        
        # 통계 정보
        latencies = sorted([r.latency_e2e_ms for r in successful_requests])
        if latencies:
            writer.writerow(["평균 E2E 지연(ms)", f"{sum(latencies)/len(latencies):.1f}"])
        
        writer.writerow([])
        
        # 데이터 헤더
        headers = [
            "request_id", "http_status", "start_time", "end_time",
            "latency_e2e_ms", "latency_ttft_ms", "tokens_generated", 
            "prompt_tokens", "prompt", "error"
        ]
        writer.writerow(headers)
        
        # 데이터 작성
        for result in results:
            row = [
                result.request_id, result.http_status, result.start_time, result.end_time,
                result.latency_e2e_ms, result.latency_ttft_ms, result.tokens_generated,
                result.prompt_tokens, clean_text(result.prompt),
                clean_text(result.error) if result.error else ""
            ]
            writer.writerow(row)
    
    print(f"결과가 {output_file}에 저장되었습니다.")

def print_summary(results: List[RequestResult]):
    """실험 결과 요약 출력"""
    successful_requests = [r for r in results if r.http_status == 200 and not r.error]
    latencies = sorted([r.latency_e2e_ms for r in successful_requests])
    ttft_latencies = sorted([r.latency_ttft_ms for r in successful_requests])
    
    print(f"\n{'='*60}")
    print("=== 실험 결과 요약 ===")
    print(f"{'='*60}")
    print(f"총 요청 수: {len(results)}")
    print(f"성공한 요청: {len(successful_requests)}")
    print(f"실패한 요청: {len(results) - len(successful_requests)}")
    
    if latencies:
        n = len(latencies)
        def pct(p):
            idx = min(n - 1, max(0, int(n * p)))
            return latencies[idx]
        
        print(f"\n[E2E 지연시간]")
        print(f"  평균: {sum(latencies)/n:.1f}ms")
        print(f"  P50:  {pct(0.50):.1f}ms")
        print(f"  P90:  {pct(0.90):.1f}ms")
        print(f"  P95:  {pct(0.95):.1f}ms")
        print(f"  P99:  {pct(0.99):.1f}ms")
    
    if ttft_latencies:
        n = len(ttft_latencies)
        def pct_ttft(p):
            idx = min(n - 1, max(0, int(n * p)))
            return ttft_latencies[idx]
        
        print(f"\n[TTFT (첫 토큰까지 시간)]")
        print(f"  평균: {sum(ttft_latencies)/n:.1f}ms")
        print(f"  P50:  {pct_ttft(0.50):.1f}ms")
        print(f"  P90:  {pct_ttft(0.90):.1f}ms")
        print(f"  P95:  {pct_ttft(0.95):.1f}ms")
        print(f"  P99:  {pct_ttft(0.99):.1f}ms")
    
    if successful_requests:
        total_tokens = sum(r.tokens_generated for r in successful_requests)
        print(f"\n[토큰 통계]")
        print(f"  총 생성 토큰: {total_tokens}")
        print(f"  평균 생성 토큰: {total_tokens/len(successful_requests):.1f}")
    
    print(f"{'='*60}\n")

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Proxy Server로 QPS 제어 요청 전송",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 실행 (QPS=10, 1000개 요청)
  python proxy_request_qps.py
  
  # QPS 20으로 2000개 요청
  python proxy_request_qps.py --qps 20 --total 2000
  
  # 동적 QPS (12~18 사이에서 30초마다 변경)
  python proxy_request_qps.py --qps 12-18 --total 2000
  
  # 동적 QPS (변경 주기 60초)
  python proxy_request_qps.py --qps 12-18 --qps-change-interval 60 --total 2000
  
  # 알고리즘 선택 (1:RR, 2:WRR, 3:SQF, 4:SLM, 5:FJ_SQF)
  python proxy_request_qps.py --algorithm 2 --qps 10 --total 1000
  
  # Fisher-Jenks SQF 알고리즘으로 실험
  python proxy_request_qps.py --algorithm 5 --qps 10 --total 1000 --fj-window-size 50 --fj-min-samples 15
  
  # SLM Adaptive
  python proxy_request_qps.py --algorithm 4 --qps 10 --total 1000
        """
    )
    
    # 프록시 서버 설정
    parser.add_argument("--base-url", default=DEFAULT_CONFIG["base_url"],
                       help="OpenAI 호환 API base URL (지정 시 proxy host/port와 /finalize를 사용하지 않음)")
    parser.add_argument("--proxy-host", default=DEFAULT_CONFIG["proxy_host"],
                       help=f"프록시 서버 호스트 (기본값: {DEFAULT_CONFIG['proxy_host']})")
    parser.add_argument("--proxy-port", type=positive_int, default=DEFAULT_CONFIG["proxy_port"],
                       help=f"프록시 서버 포트 (기본값: {DEFAULT_CONFIG['proxy_port']})")
    parser.add_argument("--algorithm", type=int, choices=[1, 2, 3, 4, 5],
                       help="라우팅 알고리즘 (1:RR, 2:WRR, 3:SQF, 4:SLM, 5:FJ_SQF)")
    
    # WRR weights 설정
    parser.add_argument("--weights", type=int, nargs='+', default=None,
                       help="WRR weights 리스트 (예: --weights 1 1 1 2 2 3 3)")
    
    # SLM 모드 활성화 임계값 설정
    parser.add_argument("--slm-activation-threshold", type=float, default=15.0,
                       help="SLM 모드 활성화 E2E latency threshold in seconds (기본값: 15.0)")
    parser.add_argument("--slm-deactivation-threshold", type=float, default=10.0,
                       help="SLM 모드 비활성화 E2E latency threshold in seconds (기본값: 10.0)")

    # Fisher-Jenks SQF 설정
    parser.add_argument("--fj-window-size", type=positive_int, default=50,
                       help="FJ_SQF diff 슬라이딩 윈도우 크기 (기본값: 50)")
    parser.add_argument("--fj-min-samples", type=positive_int, default=15,
                       help="FJ_SQF 계산 최소 샘플 수 (기본값: 15)")
    parser.add_argument("--fj-default-threshold", type=float, default=0.0,
                       help="FJ_SQF 샘플 부족 시 기본 split point in seconds (기본값: 0.0)")
    parser.add_argument("--fj-preset", type=str, default=None,
                       help="FJ 프리셋 이름 (서버 config.py EXPERIMENT_PRESETS에 정의된 프리셋)")
    
    # QPS 제어 설정
    parser.add_argument("--qps", type=qps_argument, default=str(DEFAULT_CONFIG["qps"]),
                       help=f"초당 요청 수 (고정: 숫자, 동적: 최소-최대) (기본값: {DEFAULT_CONFIG['qps']}, 예: '10' 또는 '12-18')")
    parser.add_argument("--qps-change-interval", type=positive_float, default=30.0,
                       help="동적 QPS 변경 주기 (초) (기본값: 30.0)")
    parser.add_argument("--max-concurrent", type=positive_int, default=DEFAULT_CONFIG["max_concurrent"],
                       help=f"최대 동시 요청 수 (기본값: {DEFAULT_CONFIG['max_concurrent']})")
    
    # 데이터셋 설정
    parser.add_argument("--dataset", "--sharegpt", dest="dataset",
                       default=DEFAULT_CONFIG["dataset"],
                       help="데이터셋 경로 또는 별칭(sharegpt/lmsys). --sharegpt도 호환 지원")
    parser.add_argument("--dataset-dir",
                       help="데이터셋 검색 디렉터리 (또는 BYSTANDER_DATASET_DIR 사용)")
    parser.add_argument("--start-index", type=non_negative_int, default=DEFAULT_CONFIG["start_index"],
                       help=f"시작 인덱스 (기본값: {DEFAULT_CONFIG['start_index']})")
    parser.add_argument("--total", type=positive_int, default=DEFAULT_CONFIG["total"],
                       help=f"전송할 총 요청 수 (기본값: {DEFAULT_CONFIG['total']})")
    
    # 모델 설정
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"],
                       help="모델 이름")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"],
                       help=f"생성 온도 (기본값: {DEFAULT_CONFIG['temperature']})")
    parser.add_argument("--max-tokens", type=positive_int,
                       default=DEFAULT_CONFIG["max_tokens"],
                       help="요청당 최대 생성 토큰 수 (미지정 시 서버 기본값)")
    parser.add_argument("--max-turns", type=positive_int, default=DEFAULT_CONFIG["max_turns"],
                       help=f"최대 대화 턴 수 (기본값: {DEFAULT_CONFIG['max_turns']})")
    
    # 출력 설정
    parser.add_argument("--output", default=DEFAULT_CONFIG["output"],
                       help="결과 출력 파일 (.xlsx 또는 .csv)")
    
    # 타임아웃 설정
    parser.add_argument("--client-timeout", type=positive_float, default=DEFAULT_CONFIG["client_timeout"],
                       help=f"HTTP 클라이언트 타임아웃(초) (기본값: {DEFAULT_CONFIG['client_timeout']})")

    parser.add_argument("--seed", type=int,
                       help="동적 QPS 난수 시드 (미지정 시 기존의 비결정적 동작 유지)")
    parser.add_argument("--dry-run", action="store_true",
                       help="데이터셋과 옵션만 검증하고 프록시 요청 없이 종료")
    
    # 쿠버네티스 데몬셋 재시작 설정
    parser.add_argument("--k8s-restart", action="store_true",
                       help="실험 완료 후 K8s 데몬셋 재시작")
    parser.add_argument("--k8s-master", type=str,
                       help="쿠버네티스 마스터 노드 IP/호스트명")
    parser.add_argument("--k8s-ssh-port", type=int, default=22,
                       help="SSH 포트 (기본값: 22)")
    parser.add_argument("--k8s-namespace", type=str, default="default",
                       help="쿠버네티스 네임스페이스 (기본값: default)")
    parser.add_argument("--k8s-daemonsets", type=str, nargs="+",
                       help="재시작할 데몬셋 이름들 (공백으로 구분)")
    parser.add_argument("--k8s-ssh-user", type=str, default="root",
                       help="SSH 사용자 (기본값: root)")
    parser.add_argument("--k8s-wait-time", type=int, default=30,
                       help="데몬셋 재시작 후 대기 시간(초) (기본값: 30)")
    
    return parser


def main():
    args = build_argument_parser().parse_args()
    
    # 실험 실행
    asyncio.run(run_experiment(args))

if __name__ == "__main__":
    main()
