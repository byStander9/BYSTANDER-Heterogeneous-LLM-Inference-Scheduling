#!/bin/bash
# 반복 실험 실행 스크립트
# 각 실험 사이에 K8s 데몬셋을 자동으로 재시작합니다.

# ============================================================================
# 가상환경 활성화
# ============================================================================

# 스크립트 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 가상환경 경로 (.uvvenv 또는 venv)
VENV_PATH="$SCRIPT_DIR/../.uvvenv"
if [ ! -d "$VENV_PATH" ]; then
    VENV_PATH="$SCRIPT_DIR/.uvvenv"
fi
if [ ! -d "$VENV_PATH" ]; then
    VENV_PATH="$SCRIPT_DIR/../venv"
fi
if [ ! -d "$VENV_PATH" ]; then
    VENV_PATH="$SCRIPT_DIR/venv"
fi

# 가상환경 활성화
if [ -d "$VENV_PATH" ]; then
    echo "가상환경 활성화: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    echo "Python 경로: $(which python)"
    echo ""
else
    echo "⚠ 가상환경을 찾을 수 없습니다. python3 명령을 사용합니다."
    echo ""
fi

# Python 명령 설정 (가상환경이 있으면 python, 없으면 python3)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "✗ 오류: python 또는 python3를 찾을 수 없습니다."
    exit 1
fi

echo "사용할 Python: $PYTHON_CMD"
echo "Python 버전: $($PYTHON_CMD --version)"
echo ""

# ============================================================================
# 설정
# ============================================================================

# 프록시 서버 설정
PROXY_HOST="${PROXY_HOST:-YOUR_PROXY_HOST}"   # 프록시 서버 IP (환경변수로 주입 권장)
PROXY_PORT="${PROXY_PORT:-8012}"

# 쿠버네티스 설정
K8S_ENABLED=true                         # true: K8s 재시작 활성화, false: 비활성화
K8S_MASTER="${K8S_MASTER:-YOUR_K8S_MASTER_IP}"   # 마스터 노드 IP (예: 192.168.1.100)
K8S_SSH_PORT="${K8S_SSH_PORT:-22}"        # SSH 포트 (포트포워딩된 경우 변경, 예: 2222)
K8S_NAMESPACE="vllm-inference"                 # 네임스페이스
K8S_DAEMONSETS=("vllm-engine")  # 데몬셋 이름들
K8S_SSH_USER="${K8S_SSH_USER:-ubuntu}"    # SSH 사용자
K8S_WAIT_TIME=120                        # 재시작 후 대기 시간 (초)

# Vast AI 설정 (인스턴스 재시작용)
VASTAI_ENABLED=false                    # true: Vast AI 인스턴스 재시작 활성화
VASTAI_API_KEY="${VASTAI_API_KEY:-}"    # Vast AI API Key (환경변수로 주입 - 절대 하드코딩 금지)
VASTAI_INSTANCE_IDS=(${VASTAI_INSTANCE_IDS:-})  # Vast AI 인스턴스 ID 목록 (공백 구분 환경변수)
VASTAI_WAIT_TIME=120                   # 재시작 후 대기 시간 (초) - K8s와 동일하게 설정

# 실험 설정
# 데이터셋 기본값 (실험 세트별 EXPERIMENT_SET_N_DATASET 미설정 시 사용)
#   ShareGPT:  ./sharegpt_shuffled.json
#   LMSYS:     ./lmsys_english_shuffled.json  (English만 필터링된 데이터)
DATASET_DEFAULT="./sharegpt_shuffled.json"
START_INDEX=0
MAX_CONCURRENT=10000

# 실험 반복 횟수는 각 실험 세트별로 별도 설정 (아래 EXPERIMENT_SET_N_REPEAT_COUNT)

# ============================================================================
# 실험 세트 정의
# ============================================================================
# N개의 실험 세트를 순차적으로 실행합니다.
# EXPERIMENT_SET_COUNT를 늘리고, 같은 양식으로 세트를 추가하면 자동 인식됩니다.
EXPERIMENT_SET_COUNT=4  # 총 실험 세트 수

# 실험 세트 1 (LMSYS) - 기존 실험 (완료됨)
EXPERIMENT_SET_1_ENABLED=false
EXPERIMENT_SET_1_REPEAT_COUNT=10
EXPERIMENT_SET_1_TOTAL_REQUESTS=4000
EXPERIMENT_SET_1_ALGORITHMS=(1 3 4 5)
EXPERIMENT_SET_1_ALGORITHM_NAMES=("RR" "SQF" "SLM" "FJ_SQF")
EXPERIMENT_SET_1_QPS_LIST=(25-55)
EXPERIMENT_SET_1_DATASET="./lmsys_english_shuffled.json"
EXPERIMENT_SET_1_WRR_MANUAL_WEIGHTS=false
EXPERIMENT_SET_1_WRR_WEIGHTS_LIST=()

# 실험 세트 2 (LMSYS) - 기존 실험 (완료됨)
EXPERIMENT_SET_2_ENABLED=false
EXPERIMENT_SET_2_REPEAT_COUNT=1
EXPERIMENT_SET_2_TOTAL_REQUESTS=4000
EXPERIMENT_SET_2_ALGORITHMS=(5)
EXPERIMENT_SET_2_ALGORITHM_NAMES=("FJ_SQF")
EXPERIMENT_SET_2_QPS_LIST=(22-38 24 30 36)
EXPERIMENT_SET_2_DATASET="./sharegpt_shuffled.json"
EXPERIMENT_SET_2_WRR_MANUAL_WEIGHTS=false
EXPERIMENT_SET_2_WRR_WEIGHTS_LIST=()

# 실험 세트 3 (ShareGPT FJ 대안 비교 실험)
EXPERIMENT_SET_3_ENABLED=false
EXPERIMENT_SET_3_REPEAT_COUNT=1
EXPERIMENT_SET_3_TOTAL_REQUESTS=4000
EXPERIMENT_SET_3_ALGORITHMS=(5)
EXPERIMENT_SET_3_ALGORITHM_NAMES=("FJ_SQF")
EXPERIMENT_SET_3_QPS_LIST=(22-38 24 30 36)
EXPERIMENT_SET_3_DATASET="./sharegpt_shuffled.json"
EXPERIMENT_SET_3_WRR_MANUAL_WEIGHTS=false
EXPERIMENT_SET_3_WRR_WEIGHTS_LIST=()
EXPERIMENT_SET_3_FJ_PRESETS=("sharegpt_pctmin_mid" "sharegpt_topn_2" "sharegpt_fixed_mid")

# 실험 세트 4 (LMSYS FJ 대안 비교 실험)
EXPERIMENT_SET_4_ENABLED=true
EXPERIMENT_SET_4_REPEAT_COUNT=1
EXPERIMENT_SET_4_TOTAL_REQUESTS=4000
EXPERIMENT_SET_4_ALGORITHMS=(5)
EXPERIMENT_SET_4_ALGORITHM_NAMES=("FJ_SQF")
EXPERIMENT_SET_4_QPS_LIST=(45-75 50 60 70)
EXPERIMENT_SET_4_DATASET="./lmsys_english_shuffled.json"
EXPERIMENT_SET_4_WRR_MANUAL_WEIGHTS=false
EXPERIMENT_SET_4_WRR_WEIGHTS_LIST=()
EXPERIMENT_SET_4_FJ_PRESETS=("lmsys_pctmin_mid" "lmsys_topn_2")

# 실험 세트 추가 예시:
# EXPERIMENT_SET_N_ENABLED=true
# EXPERIMENT_SET_N_REPEAT_COUNT=5
# EXPERIMENT_SET_N_TOTAL_REQUESTS=4000
# EXPERIMENT_SET_N_ALGORITHMS=(1 5)
# EXPERIMENT_SET_N_ALGORITHM_NAMES=("RR" "FJ_SQF")
# EXPERIMENT_SET_N_QPS_LIST=(10-20)
# EXPERIMENT_SET_N_DATASET="./sharegpt_shuffled.json"
# EXPERIMENT_SET_N_WRR_MANUAL_WEIGHTS=false
# EXPERIMENT_SET_N_WRR_WEIGHTS_LIST=()

# 알고리즘 목록 (호환성 유지용 - 실제로는 위의 세트별 설정 사용)
ALGORITHMS=(1 2 3 4 5)
ALGORITHM_NAMES=("RR" "WRR" "SQF" "SLM" "FJ_SQF")

# WRR Weights 기본값 (실험 세트별 미설정 시 사용)
WRR_MANUAL_WEIGHTS_DEFAULT=false
WRR_WEIGHTS_LIST_DEFAULT=()

# 기본값 (실험 세트별 미설정 시 사용)
TOTAL_REQUESTS_DEFAULT=6000  # 총 요청 수 기본값

# 실험 변수 리스트 (호환성 유지용)
QPS_LIST=(12 15 10-20)         # QPS 리스트 (고정 QPS: "12", 동적 QPS: "12-18")

# 동적 QPS 설정 (QPS_LIST에 범위를 지정한 경우)
QPS_CHANGE_INTERVAL=10      # 동적 QPS 변경 주기 (초)

# SLM Threshold 리스트 (여러 값 설정 가능)
SLM_ACTIVATION_THRESHOLD_LIST=(15.0)          # Activation threshold 리스트 (초)
SLM_DEACTIVATION_THRESHOLD_LIST=(10.0)        # Deactivation threshold 리스트 (초)

# Fisher-Jenks SQF 설정 (알고리즘 5번일 때)
FJ_WINDOW_SIZE_LIST=(50)                   # diff 슬라이딩 윈도우 크기 리스트
FJ_MIN_SAMPLES_LIST=(15)                   # FJ 계산 최소 샘플 수 리스트
FJ_DEFAULT_THRESHOLD_LIST=(0.0)            # 샘플 부족 시 기본 split point 리스트 (초)

# 결과 저장 디렉토리
OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# 함수 정의
# ============================================================================

# 타임스탬프 생성
timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# Vast AI 단일 인스턴스 재시작
restart_vastai_single_instance() {
    local instance_id=$1
    local instance_idx=$2
    
    echo "🔄 [Vast AI #${instance_idx}] 인스턴스 ${instance_id} 재시작 중..."
    
    # 인스턴스 상태 확인
    if ! vastai show instance "$instance_id" > /dev/null 2>&1; then
        echo "✗ [Vast AI #${instance_idx}] 인스턴스 ID ${instance_id}를 찾을 수 없습니다."
        return 1
    fi
    
    echo "✓ [Vast AI #${instance_idx}] 인스턴스 발견: ${instance_id}"
    
    # 인스턴스 재시작
    if vastai reboot instance "$instance_id"; then
        echo "✓ [Vast AI #${instance_idx}] 인스턴스 ${instance_id} 재시작 요청 성공"
        return 0
    else
        echo "✗ [Vast AI #${instance_idx}] 인스턴스 ${instance_id} 재시작 실패"
        return 1
    fi
}

# Vast AI 전체 인스턴스 재시작 (CLI 사용, 모든 인스턴스 병렬 재시작)
restart_vastai_instance() {
    if [ "$VASTAI_ENABLED" != "true" ]; then
        return 0
    fi
    
    echo "🔄 [Vast AI] 전체 인스턴스 재시작 중 (${#VASTAI_INSTANCE_IDS[@]}대)..."
    
    # API Key 확인
    if [ -z "$VASTAI_API_KEY" ]; then
        echo "⚠️  [Vast AI] VASTAI_API_KEY가 설정되지 않았습니다. 재시작을 건너뜁니다."
        return 1
    fi
    
    # Instance IDs 확인
    if [ ${#VASTAI_INSTANCE_IDS[@]} -eq 0 ]; then
        echo "⚠️  [Vast AI] VASTAI_INSTANCE_IDS가 비어있습니다. 재시작을 건너뜁니다."
        return 1
    fi
    
    # API Key 설정
    vastai set api-key "$VASTAI_API_KEY" > /dev/null 2>&1
    
    # 모든 인스턴스 병렬 재시작 요청
    local reboot_pids=()
    local idx=1
    for instance_id in "${VASTAI_INSTANCE_IDS[@]}"; do
        restart_vastai_single_instance "$instance_id" "$idx" &
        reboot_pids+=($!)
        idx=$((idx + 1))
    done
    
    # 모든 재시작 요청 완료 대기
    local all_success=true
    local i=0
    for pid in "${reboot_pids[@]}"; do
        if ! wait $pid; then
            all_success=false
        fi
        i=$((i + 1))
    done
    
    if [ "$all_success" != true ]; then
        echo "⚠️  [Vast AI] 일부 인스턴스 재시작 요청 실패"
    fi
    
    # 대기 시간 (모든 인스턴스가 동시에 재시작되므로 한 번만 대기)
    echo "⏳ [Vast AI] 전체 인스턴스 재시작 및 vLLM 서비스 시작 대기 중 (${VASTAI_WAIT_TIME}초)..."
    sleep "$VASTAI_WAIT_TIME"
    
    # 재시작 후 상태 확인 (모든 인스턴스)
    echo "🔍 [Vast AI] 인스턴스 재시작 완료 확인 중..."
    local max_retries=5
    local retry=0
    local all_running=false
    
    while [ $retry -lt $max_retries ]; do
        all_running=true
        for instance_id in "${VASTAI_INSTANCE_IDS[@]}"; do
            if ! vastai show instance "$instance_id" 2>/dev/null | grep -q "running"; then
                all_running=false
                break
            fi
        done
        
        if [ "$all_running" = true ]; then
            echo "✓ [Vast AI] 모든 인스턴스(${#VASTAI_INSTANCE_IDS[@]}대)가 정상적으로 실행 중입니다."
            return 0
        fi
        
        retry=$((retry + 1))
        echo "⏳ [Vast AI] 재시작 확인 중... (시도 ${retry}/${max_retries})"
        sleep 10
    done
    
    echo "⚠️  [Vast AI] 일부 인스턴스 상태를 확인할 수 없지만 계속 진행합니다."
    return 0
}

# K8s DaemonSet 재시작
restart_k8s_daemonsets() {
    echo "🔄 [K8s] DaemonSet 재시작 중..."
    
    # SSH 연결 테스트
    if ! ssh -p "$K8S_SSH_PORT" -o ConnectTimeout=10 -o BatchMode=yes \
        "${K8S_SSH_USER}@${K8S_MASTER}" "echo 'SSH 연결 성공'" > /dev/null 2>&1; then
        echo "✗ [K8s] SSH 연결 실패: ${K8S_SSH_USER}@${K8S_MASTER}:${K8S_SSH_PORT}"
        return 1
    fi
    
    echo "✓ [K8s] SSH 연결 성공"
    
    # 각 DaemonSet 재시작
    for daemonset in "${K8S_DAEMONSETS[@]}"; do
        echo "🔄 [K8s] DaemonSet '${daemonset}' 재시작 중..."
        
        if ssh -p "$K8S_SSH_PORT" "${K8S_SSH_USER}@${K8S_MASTER}" \
            "kubectl rollout restart daemonset/${daemonset} -n ${K8S_NAMESPACE}" > /dev/null 2>&1; then
            echo "✓ [K8s] DaemonSet '${daemonset}' 재시작 요청 성공"
        else
            echo "✗ [K8s] DaemonSet '${daemonset}' 재시작 실패"
            return 1
        fi
    done
    
    echo "⏳ [K8s] 파드 재시작 대기 중 (${K8S_WAIT_TIME}초)..."
    sleep "$K8S_WAIT_TIME"
    echo "✓ [K8s] 파드 재시작 완료"
    
    return 0
}

# Vast AI와 K8s 병렬 재시작
restart_all_parallel() {
    local vastai_enabled="$VASTAI_ENABLED"
    local k8s_enabled="$K8S_ENABLED"
    
    if [ "$vastai_enabled" != "true" ] && [ "$k8s_enabled" != "true" ]; then
        echo "⚠️  재시작 옵션이 모두 비활성화되어 있습니다."
        return 0
    fi
    
    echo ""
    echo "========================================================================"
    echo "🔄 인프라 재시작 (병렬 실행)"
    echo "========================================================================"
    
    local pids=()
    local results=()
    
    # Vast AI 재시작 (백그라운드)
    if [ "$vastai_enabled" = "true" ]; then
        restart_vastai_instance &
        pids+=($!)
        results+=("vastai")
    fi
    
    # K8s 재시작 (백그라운드)
    if [ "$k8s_enabled" = "true" ]; then
        restart_k8s_daemonsets &
        pids+=($!)
        results+=("k8s")
    fi
    
    # 모든 백그라운드 프로세스 완료 대기
    local all_success=true
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local name=${results[$i]}
        
        if wait $pid; then
            echo "✓ [${name}] 재시작 완료"
        else
            echo "✗ [${name}] 재시작 실패"
            all_success=false
        fi
    done
    
    if [ "$all_success" = true ]; then
        echo ""
        echo "✓ 모든 인프라 재시작 완료"
        echo "========================================================================"
        return 0
    else
        echo ""
        echo "⚠️  일부 재시작 실패 (실험 계속 진행)"
        echo "========================================================================"
        return 1
    fi
}

# 실험 실행
run_experiment() {
    local algo=$1
    local algo_name=$2
    local total_requests=$3
    local qps=$4
    local slm_activation=$5
    local slm_deactivation=$6
    local repeat_num=$7
    local wrr_weights="${8}"
    local fj_window_size="${9}"
    local fj_min_samples="${10}"
    local fj_default_threshold="${11}"
    local experiment_dataset="${12}"
    local fj_preset="${13}"

    local timestamp=$(timestamp)

    if [ "$algo" -eq 4 ]; then
        local act_str=$(echo "$slm_activation" | sed 's/\./_/g')
        local deact_str=$(echo "$slm_deactivation" | sed 's/\./_/g')
        local output_file="$OUTPUT_DIR/results_${algo_name}_T${total_requests}_QPS${qps}_A${act_str}_D${deact_str}_R${repeat_num}_${timestamp}.xlsx"
    elif [ "$algo" -eq 2 ]; then
        # WRR인 경우 (weights 없이 QPS 정보만 사용)
        local output_file="$OUTPUT_DIR/results_${algo_name}_T${total_requests}_QPS${qps}_R${repeat_num}_${timestamp}.xlsx"
    elif [ "$algo" -eq 5 ]; then
        if [ -n "$fj_preset" ]; then
            # FJ 프리셋 모드: 프리셋 이름을 파일명에 포함
            local output_file="$OUTPUT_DIR/results_${algo_name}_${fj_preset}_T${total_requests}_QPS${qps}_R${repeat_num}_${timestamp}.xlsx"
        else
            # FJ_SQF 기존 모드: 파라미터 정보를 파일명에 포함
            local fj_default_str=$(echo "$fj_default_threshold" | sed 's/\./_/g')
            local output_file="$OUTPUT_DIR/results_${algo_name}_T${total_requests}_QPS${qps}_W${fj_window_size}_M${fj_min_samples}_D${fj_default_str}_R${repeat_num}_${timestamp}.xlsx"
        fi
    else
        local output_file="$OUTPUT_DIR/results_${algo_name}_T${total_requests}_QPS${qps}_R${repeat_num}_${timestamp}.xlsx"
    fi
    
    echo ""
    echo "========================================================================"
    echo "실험 시작: ${algo_name}"
    echo "  Dataset: ${experiment_dataset}"
    echo "  Total Requests: ${total_requests}"
    echo "  QPS: ${qps}"
    echo "  Repeat: ${repeat_num}"
    if [ "$algo" -eq 2 ]; then
        if [ -n "$wrr_weights" ]; then
            echo "  WRR: 수동 weights 지정 → [${wrr_weights}]"
        else
            echo "  WRR: 프록시 서버 자동 weights 결정"
        fi
    fi
    if [ "$algo" -eq 4 ]; then
        echo "  Activation Threshold: ${slm_activation}s"
        echo "  Deactivation Threshold: ${slm_deactivation}s"
    fi
    if [ "$algo" -eq 5 ]; then
        if [ -n "$fj_preset" ]; then
            echo "  FJ Preset: ${fj_preset}"
        else
            echo "  FJ Window Size: ${fj_window_size}"
            echo "  FJ Min Samples: ${fj_min_samples}"
            echo "  FJ Default Threshold: ${fj_default_threshold}s"
        fi
    fi
    echo "  출력 파일: $output_file"
    echo "========================================================================"
    echo ""
    
    # 실험 명령 구성
    local cmd="$PYTHON_CMD proxy_request_qps.py \
        --proxy-host $PROXY_HOST \
        --proxy-port $PROXY_PORT \
        --algorithm $algo \
        --qps $qps \
        --total $total_requests \
        --start-index $START_INDEX \
        --max-concurrent $MAX_CONCURRENT \
        --sharegpt $experiment_dataset \
        --output $output_file"
    
    # WRR 알고리즘(2번)일 때 weights 처리 (수동 지정된 경우에만)
    if [ "$algo" -eq 2 ] && [ -n "$wrr_weights" ]; then
        cmd="$cmd --weights $wrr_weights"
    fi
    
    # SLM 옵션 추가 (알고리즘 4번일 때)
    if [ "$algo" -eq 4 ]; then
        cmd="$cmd --slm-activation-threshold $slm_activation"
        cmd="$cmd --slm-deactivation-threshold $slm_deactivation"
    fi
    
    # FJ_SQF 옵션 추가 (알고리즘 5번일 때)
    if [ "$algo" -eq 5 ]; then
        if [ -n "$fj_preset" ]; then
            cmd="$cmd --fj-preset $fj_preset"
        else
            cmd="$cmd --fj-window-size $fj_window_size"
            cmd="$cmd --fj-min-samples $fj_min_samples"
            cmd="$cmd --fj-default-threshold $fj_default_threshold"
        fi
        cmd="$cmd --slm-activation-threshold $slm_activation"
        cmd="$cmd --slm-deactivation-threshold $slm_deactivation"
    fi
    
    # 동적 QPS 옵션 추가 (QPS에 '-'가 포함된 경우)
    if [[ "$qps" == *-* ]]; then
        cmd="$cmd --qps-change-interval $QPS_CHANGE_INTERVAL"
    fi
    
    # K8s 재시작은 Bash 스크립트의 restart_all_parallel()에서 처리
    # Python 스크립트에는 --k8s-restart를 전달하지 않음 (중복 재시작 방지)
    
    echo "실행 명령:"
    echo "$cmd"
    echo ""
    
    # 실험 실행
    eval $cmd
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ 실험 완료: ${algo_name}"
        echo "  Total Requests: ${total_requests}"
        echo "  QPS: ${qps}"
        echo "  결과 파일: $output_file"
    else
        echo ""
        echo "✗ 실험 실패: ${algo_name}"
        echo "  Total Requests: ${total_requests}"
        echo "  QPS: ${qps}"
        echo "  종료 코드: $exit_code"
        return $exit_code
    fi
    
    echo ""
    echo "다음 실험 준비 중..."
    echo ""
    
    return 0
}

# ============================================================================
# 메인 실험 루프
# ============================================================================

echo "========================================================================"
echo "반복 실험 시작"
echo "========================================================================"

# 실험 수 계산 함수
calculate_experiments() {
    local qps_count=$1
    shift
    local preset_count=$1
    shift
    local algo_list=("$@")
    
    local slm_count=0
    local fj_sqf_count=0
    local non_special_count=0

    for algo in "${algo_list[@]}"; do
        if [ "$algo" -eq 4 ]; then
            slm_count=$((qps_count * ${#SLM_ACTIVATION_THRESHOLD_LIST[@]}))
        elif [ "$algo" -eq 5 ]; then
            if [ "$preset_count" -gt 0 ]; then
                fj_sqf_count=$((qps_count * preset_count))
            else
                fj_sqf_count=$((qps_count * ${#FJ_WINDOW_SIZE_LIST[@]} * ${#FJ_MIN_SAMPLES_LIST[@]} * ${#FJ_DEFAULT_THRESHOLD_LIST[@]} * ${#SLM_ACTIVATION_THRESHOLD_LIST[@]}))
            fi
        else
            non_special_count=$((non_special_count + 1))
        fi
    done

    local total=$(($non_special_count * qps_count + slm_count + fj_sqf_count))
    echo $total
}

# 각 실험 세트별 실험 수 계산 및 출력
total_experiments=0
echo ""
for set_idx in $(seq 1 $EXPERIMENT_SET_COUNT); do
    # 변수명 동적 참조
    enabled_var="EXPERIMENT_SET_${set_idx}_ENABLED"
    repeat_var="EXPERIMENT_SET_${set_idx}_REPEAT_COUNT"
    algos_var="EXPERIMENT_SET_${set_idx}_ALGORITHMS[@]"
    algo_names_var="EXPERIMENT_SET_${set_idx}_ALGORITHM_NAMES[@]"
    qps_var="EXPERIMENT_SET_${set_idx}_QPS_LIST[@]"
    
    if [ "${!enabled_var}" = true ]; then
        # QPS 리스트 길이 계산
        qps_count_var="EXPERIMENT_SET_${set_idx}_QPS_LIST[@]"
        local_qps_list=("${!qps_count_var}")
        qps_count=${#local_qps_list[@]}
        
        # FJ 프리셋 리스트 길이 계산
        fj_presets_count_var="EXPERIMENT_SET_${set_idx}_FJ_PRESETS[@]"
        local_fj_presets=()
        if declare -p "EXPERIMENT_SET_${set_idx}_FJ_PRESETS" &>/dev/null; then
            local_fj_presets=("${!fj_presets_count_var}")
        fi
        fj_preset_count=${#local_fj_presets[@]}
        
        local_algos=("${!algos_var}")
        set_experiments=$(calculate_experiments $qps_count $fj_preset_count "${local_algos[@]}")
        set_repeat=${!repeat_var}
        set_total=$((set_experiments * set_repeat))
        total_experiments=$((total_experiments + set_total))
        
        local_names=("${!algo_names_var}")
        summary_dataset_var="EXPERIMENT_SET_${set_idx}_DATASET"
        summary_wrr_var="EXPERIMENT_SET_${set_idx}_WRR_MANUAL_WEIGHTS"
        summary_total_req_var="EXPERIMENT_SET_${set_idx}_TOTAL_REQUESTS"
        echo "실험 세트 ${set_idx}:"
        echo "  데이터셋: ${!summary_dataset_var:-$DATASET_DEFAULT}"
        echo "  요청 수: ${!summary_total_req_var:-$TOTAL_REQUESTS_DEFAULT}"
        echo "  알고리즘: ${local_names[*]}"
        echo "  QPS: ${local_qps_list[*]}"
        echo "  WRR 수동 weights: ${!summary_wrr_var:-$WRR_MANUAL_WEIGHTS_DEFAULT}"
        if [ "$fj_preset_count" -gt 0 ]; then
            echo "  FJ Presets (${fj_preset_count}개): ${local_fj_presets[*]}"
        fi
        echo "  반복 횟수: $set_repeat"
        echo "  실험 수 (1회당): $set_experiments"
        echo "  실험 수 (총): $set_total"
    fi
done
echo ""
echo "전체 실험 수: $total_experiments"
echo "K8s Master: $K8S_MASTER"
echo "K8s Namespace: $K8S_NAMESPACE"
echo "K8s Daemonsets: ${K8S_DAEMONSETS[@]}"
echo "========================================================================"
echo ""

experiment_count=0
success_count=0
failure_count=0

# 실험 세트 실행 함수
run_experiment_set() {
    local set_name=$1
    shift
    local set_repeat_count=$1
    shift
    local set_total_requests=$1
    shift
    local -n algos=$1
    shift
    local -n algo_names=$1
    shift
    local -n qps_list=$1
    shift
    local set_dataset=$1
    shift
    local set_wrr_manual=$1
    shift
    local -n set_wrr_weights_list=$1
    shift
    local -n set_fj_presets=$1
    
    echo ""
    echo "========================================================================"
    echo "[$set_name] 시작 (반복: ${set_repeat_count}회)"
    echo "  데이터셋: ${set_dataset}"
    echo "  요청 수: ${set_total_requests}"
    echo "  알고리즘: ${algo_names[@]}"
    echo "  QPS: ${qps_list[@]}"
    echo "  WRR 수동 weights: ${set_wrr_manual}"
    if [ "$set_wrr_manual" = "true" ] && [ ${#set_wrr_weights_list[@]} -gt 0 ]; then
        echo "  WRR Weights: ${set_wrr_weights_list[*]}"
    fi
    if [ ${#set_fj_presets[@]} -gt 0 ]; then
        echo "  FJ Presets (${#set_fj_presets[@]}개): ${set_fj_presets[*]}"
    fi
    echo "========================================================================"
    echo ""
    
    # 세트별 반복 루프
    for repeat in $(seq 1 $set_repeat_count); do
        echo ""
        echo "------------------------------------------------------------------------"
        echo "[$set_name] 반복 사이클 [$repeat / $set_repeat_count]"
        echo "------------------------------------------------------------------------"
        
    # 각 알고리즘에 대해
    local total_requests=$set_total_requests
        for i in "${!algos[@]}"; do
            algo=${algos[$i]}
            algo_name=${algo_names[$i]}
            
            # WRR 알고리즘(2번)인 경우 각 QPS에 대해 실행
            if [ "$algo" -eq 2 ]; then
                for qps_idx in "${!qps_list[@]}"; do
                    qps=${qps_list[$qps_idx]}
                    # 세트별 WRR_MANUAL_WEIGHTS 설정으로 weights 가져오기
                    local wrr_w=""
                    if [ "$set_wrr_manual" = "true" ] && [ ${#set_wrr_weights_list[@]} -gt 0 ]; then
                        wrr_w="${set_wrr_weights_list[$qps_idx]:-${set_wrr_weights_list[-1]}}"
                    fi
                    experiment_count=$((experiment_count + 1))
                    echo ""
                    echo "========================================================================"
                    echo "실험 [$experiment_count / $total_experiments]"
                    echo "========================================================================"
                    
                    run_experiment $algo $algo_name $total_requests $qps "0" "0" $repeat "$wrr_w" "" "" "" "$set_dataset"
                    
                    if [ $? -eq 0 ]; then
                        success_count=$((success_count + 1))
                    else
                        failure_count=$((failure_count + 1))
                        echo "⚠ 실험 실패. 계속 진행합니다..."
                    fi
                    
                    # 인프라 병렬 재시작 (Vast AI + K8s)
                    restart_all_parallel
                    
                    # 실험 간 대기
                    echo "다음 실험까지 10초 대기..."
                    sleep 10
                done
            # SLM 알고리즘(4번)인 경우 threshold 쌍으로 실행
            elif [ "$algo" -eq 4 ]; then
                for qps in "${qps_list[@]}"; do
                    for idx in "${!SLM_ACTIVATION_THRESHOLD_LIST[@]}"; do
                        act=${SLM_ACTIVATION_THRESHOLD_LIST[$idx]}
                        deact=${SLM_DEACTIVATION_THRESHOLD_LIST[$idx]:-"10.0"}

                        experiment_count=$((experiment_count + 1))
                        echo ""
                        echo "========================================================================"
                        echo "실험 [$experiment_count / $total_experiments]"
                        echo "========================================================================"

                        run_experiment 4 "SLM_Adaptive" $total_requests $qps $act $deact $repeat "" "" "" "" "$set_dataset"

                        if [ $? -eq 0 ]; then
                            success_count=$((success_count + 1))
                        else
                            failure_count=$((failure_count + 1))
                            echo "⚠ 실험 실패. 계속 진행합니다..."
                        fi

                        restart_all_parallel
                        echo "다음 실험까지 10초 대기..."
                        sleep 10
                    done
                done
            # FJ_SQF 알고리즘(5번)인 경우
            elif [ "$algo" -eq 5 ]; then
                if [ ${#set_fj_presets[@]} -gt 0 ]; then
                    # FJ 프리셋 모드: 각 QPS × 각 프리셋 조합 실행
                    for qps in "${qps_list[@]}"; do
                        for preset in "${set_fj_presets[@]}"; do
                            local act=${SLM_ACTIVATION_THRESHOLD_LIST[0]:-"15.0"}
                            local deact=${SLM_DEACTIVATION_THRESHOLD_LIST[0]:-"10.0"}
                            
                            experiment_count=$((experiment_count + 1))
                            echo ""
                            echo "========================================================================"
                            echo "실험 [$experiment_count / $total_experiments]"
                            echo "========================================================================"
                            
                            run_experiment 5 "FJ_SQF" $total_requests $qps $act $deact $repeat "" "" "" "" "$set_dataset" "$preset"
                            
                            if [ $? -eq 0 ]; then
                                success_count=$((success_count + 1))
                            else
                                failure_count=$((failure_count + 1))
                                echo "⚠ 실험 실패. 계속 진행합니다..."
                            fi
                            
                            # 인프라 병렬 재시작 (Vast AI + K8s)
                            restart_all_parallel
                            
                            # 실험 간 대기
                            echo "다음 실험까지 10초 대기..."
                            sleep 10
                        done
                    done
                else
                    # FJ 기존 모드: 파라미터 그리드 조합 실행
                    for qps in "${qps_list[@]}"; do
                        for fj_window in "${FJ_WINDOW_SIZE_LIST[@]}"; do
                            for fj_min in "${FJ_MIN_SAMPLES_LIST[@]}"; do
                                for fj_default in "${FJ_DEFAULT_THRESHOLD_LIST[@]}"; do
                                    for idx in "${!SLM_ACTIVATION_THRESHOLD_LIST[@]}"; do
                                        act=${SLM_ACTIVATION_THRESHOLD_LIST[$idx]}
                                        deact=${SLM_DEACTIVATION_THRESHOLD_LIST[$idx]:-"10.0"}
                                        
                                        experiment_count=$((experiment_count + 1))
                                        echo ""
                                        echo "========================================================================"
                                        echo "실험 [$experiment_count / $total_experiments]"
                                        echo "========================================================================"
                                        
                                        run_experiment 5 "FJ_SQF" $total_requests $qps $act $deact $repeat "" "$fj_window" "$fj_min" "$fj_default" "$set_dataset"
                                        
                                        if [ $? -eq 0 ]; then
                                            success_count=$((success_count + 1))
                                        else
                                            failure_count=$((failure_count + 1))
                                            echo "⚠ 실험 실패. 계속 진행합니다..."
                                        fi
                                        
                                        # 인프라 병렬 재시작 (Vast AI + K8s)
                                        restart_all_parallel
                                        
                                        # 실험 간 대기
                                        echo "다음 실험까지 10초 대기..."
                                        sleep 10
                                    done
                                done
                            done
                        done
                    done
                fi
            else
                # 다른 알고리즘 (RR, SQF)은 각 QPS에 대해 실행
                for qps in "${qps_list[@]}"; do
                    experiment_count=$((experiment_count + 1))
                    echo ""
                    echo "========================================================================"
                    echo "실험 [$experiment_count / $total_experiments]"
                    echo "========================================================================"
                    
                    run_experiment $algo $algo_name $total_requests $qps "0" "0" $repeat "" "" "" "" "$set_dataset"
                    
                    if [ $? -eq 0 ]; then
                        success_count=$((success_count + 1))
                    else
                        failure_count=$((failure_count + 1))
                        echo "⚠ 실험 실패. 계속 진행합니다..."
                    fi
                    
                    # 인프라 병렬 재시작 (Vast AI + K8s)
                    restart_all_parallel
                    
                    # 실험 간 대기
                    echo "다음 실험까지 10초 대기..."
                    sleep 10
                done
            fi
        done
    
        echo ""
        echo "------------------------------------------------------------------------"
        echo "[$set_name] 반복 사이클 [$repeat / $set_repeat_count] 완료"
        echo "------------------------------------------------------------------------"
    done
}

# 모든 실험 세트 순차 실행
for set_idx in $(seq 1 $EXPERIMENT_SET_COUNT); do
    enabled_var="EXPERIMENT_SET_${set_idx}_ENABLED"
    repeat_var="EXPERIMENT_SET_${set_idx}_REPEAT_COUNT"
    dataset_var="EXPERIMENT_SET_${set_idx}_DATASET"
    wrr_manual_var="EXPERIMENT_SET_${set_idx}_WRR_MANUAL_WEIGHTS"
    
    if [ "${!enabled_var}" = true ]; then
        # 세트별 TOTAL_REQUESTS (미설정 시 기본값 사용)
        total_req_var="EXPERIMENT_SET_${set_idx}_TOTAL_REQUESTS"
        local_total_requests="${!total_req_var:-$TOTAL_REQUESTS_DEFAULT}"
        # 세트별 DATASET (미설정 시 기본값 사용)
        local_dataset="${!dataset_var:-$DATASET_DEFAULT}"
        # 세트별 WRR_MANUAL_WEIGHTS (미설정 시 기본값 사용)
        local_wrr_manual="${!wrr_manual_var:-$WRR_MANUAL_WEIGHTS_DEFAULT}"
        
        # WRR_WEIGHTS_LIST 변수 존재 확인 (미정의 시 기본값 배열 사용)
        wrr_list_var="EXPERIMENT_SET_${set_idx}_WRR_WEIGHTS_LIST"
        if ! declare -p "$wrr_list_var" &>/dev/null; then
            declare -a "$wrr_list_var=()"
        fi
        
        # FJ_PRESETS 변수 존재 확인 (미정의 시 기본값 배열 사용)
        fj_presets_var="EXPERIMENT_SET_${set_idx}_FJ_PRESETS"
        if ! declare -p "$fj_presets_var" &>/dev/null; then
            declare -a "$fj_presets_var=()"
        fi
        
        run_experiment_set "실험 세트 ${set_idx}" "${!repeat_var}" \
            "$local_total_requests" \
            "EXPERIMENT_SET_${set_idx}_ALGORITHMS" \
            "EXPERIMENT_SET_${set_idx}_ALGORITHM_NAMES" \
            "EXPERIMENT_SET_${set_idx}_QPS_LIST" \
            "$local_dataset" \
            "$local_wrr_manual" \
            "$wrr_list_var" \
            "$fj_presets_var"
    fi
done

# ============================================================================
# 결과 요약
# ============================================================================

echo ""
echo "========================================================================"
echo "모든 실험 완료!"
echo "========================================================================"
echo "총 실험 수: $experiment_count"
echo "성공: $success_count"
echo "실패: $failure_count"
echo "결과 디렉토리: $OUTPUT_DIR"
echo "========================================================================"
echo ""

# 결과 파일 목록
echo "생성된 결과 파일:"
ls -lh "$OUTPUT_DIR"/*.xlsx 2>/dev/null | tail -n $experiment_count

exit 0
