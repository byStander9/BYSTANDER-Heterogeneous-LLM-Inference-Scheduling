#!/bin/bash
# 단일 실험 + 데몬셋 재시작 예제 스크립트

# ============================================================================
# 설정 (여기를 수정하세요)
# ============================================================================

# 프록시 서버
PROXY_HOST="${PROXY_HOST:-YOUR_PROXY_HOST}"
PROXY_PORT="8012"

# 쿠버네티스 마스터 노드
K8S_MASTER="YOUR_K8S_MASTER_IP"  # 예: "192.168.1.100"

# SSH 포트 (포트포워딩된 경우 변경)
K8S_SSH_PORT=22  # 예: 2222

# 쿠버네티스 네임스페이스
K8S_NAMESPACE="default"

# 재시작할 데몬셋 이름들 (실제 데몬셋 이름으로 변경)
K8S_DAEMONSETS="vllm-rtx3090 vllm-rtx5090"

# SSH 사용자
K8S_SSH_USER="root"

# 실험 설정
ALGORITHM=2          # 1:RR, 2:WRR, 3:SQF, 4:SLM
QPS=10
TOTAL=1000
OUTPUT="results/test_experiment.xlsx"

# ============================================================================
# 실험 실행
# ============================================================================

python proxy_request_qps.py \
    --proxy-host $PROXY_HOST \
    --proxy-port $PROXY_PORT \
    --algorithm $ALGORITHM \
    --qps $QPS \
    --total $TOTAL \
    --output $OUTPUT \
    --k8s-restart \
    --k8s-master $K8S_MASTER \
    --k8s-ssh-port $K8S_SSH_PORT \
    --k8s-namespace $K8S_NAMESPACE \
    --k8s-daemonsets $K8S_DAEMONSETS \
    --k8s-ssh-user $K8S_SSH_USER \
    --k8s-wait-time 30

echo ""
echo "실험 완료! 결과: $OUTPUT"
