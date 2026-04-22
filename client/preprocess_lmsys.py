#!/usr/bin/env python3
"""
LMSYS-Chat-1M 데이터셋 전처리 스크립트

HuggingFace datasets (arrow) 형식의 lmsys-chat-1m 데이터를
JSON 파일로 변환합니다.

기능:
- English 언어 필터링
- 고정 seed로 셔플 (실험 재현성 보장)
- 유효한 대화만 포함 (user 메시지가 있는 것)
- JSON 배열 형식으로 저장

사용법:
    python preprocess_lmsys.py
    python preprocess_lmsys.py --input ./lmsys-chat-1m/processed --output ./lmsys_english_shuffled.json
    python preprocess_lmsys.py --seed 42 --language English
"""

import json
import random
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="LMSYS-Chat-1M 데이터셋 전처리")
    parser.add_argument("--input", default="./lmsys-chat-1m/processed",
                       help="HuggingFace datasets 디렉토리 경로 (기본값: ./lmsys-chat-1m/processed)")
    parser.add_argument("--output", default="./lmsys_english_shuffled.json",
                       help="출력 JSON 파일 경로 (기본값: ./lmsys_english_shuffled.json)")
    parser.add_argument("--seed", type=int, default=42,
                       help="셔플 시드 (기본값: 42)")
    parser.add_argument("--language", default="English",
                       help="필터링할 언어 (기본값: English)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"LMSYS-Chat-1M 데이터셋 전처리")
    print(f"{'='*60}")
    print(f"입력 경로: {args.input}")
    print(f"출력 경로: {args.output}")
    print(f"언어 필터: {args.language}")
    print(f"셔플 시드: {args.seed}")
    print()

    # HuggingFace datasets 로드
    start_time = time.time()
    print("데이터셋 로드 중...")

    try:
        from datasets import load_from_disk
    except ImportError:
        print("오류: 'datasets' 패키지가 필요합니다.")
        print("설치: uv pip install datasets")
        raise SystemExit(1)

    ds = load_from_disk(args.input)
    train_ds = ds["train"]
    print(f"총 데이터 수: {len(train_ds):,}개")

    # 언어 필터링
    print(f"'{args.language}' 언어 필터링 중...")
    filtered = train_ds.filter(
        lambda x: x["language"] == args.language,
        num_proc=4,
        desc=f"Filtering {args.language}"
    )
    print(f"필터링 후 데이터 수: {len(filtered):,}개")

    # 유효한 대화만 추출 (conversation이 있고, user 메시지가 포함된 것)
    print("유효한 대화 필터링 중...")
    items = []
    skipped = 0
    for i in range(len(filtered)):
        item = filtered[i]
        conversation = item["conversation"]

        # conversation이 비어있으면 건너뛰기
        if not conversation:
            skipped += 1
            continue

        # user 메시지가 하나라도 있는지 확인
        has_user = any(msg["role"] in ("user", "human") for msg in conversation)
        if not has_user:
            skipped += 1
            continue

        # JSON 직렬화를 위해 필요한 필드만 추출
        items.append({
            "conversation_id": item["conversation_id"],
            "conversation": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversation
            ],
            "turn": item["turn"],
            "language": item["language"],
        })

    print(f"유효한 대화 수: {len(items):,}개 (건너뜀: {skipped}개)")

    # 고정 seed로 셔플
    print(f"셔플 중 (seed={args.seed})...")
    random.seed(args.seed)
    random.shuffle(items)

    # JSON 파일로 저장
    print(f"JSON 파일로 저장 중: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)

    elapsed = time.time() - start_time
    file_size = output_path.stat().st_size / (1024 * 1024)

    print()
    print(f"{'='*60}")
    print(f"전처리 완료!")
    print(f"{'='*60}")
    print(f"출력 파일: {args.output}")
    print(f"파일 크기: {file_size:.1f} MB")
    print(f"총 대화 수: {len(items):,}개")
    print(f"소요 시간: {elapsed:.1f}초")
    print()
    print(f"사용 예시:")
    print(f"  # proxy_request_qps.py에서 사용")
    print(f"  python proxy_request_qps.py --sharegpt {args.output} --total 2000")
    print(f"  # run_repeated_experiments.sh에서 사용")
    print(f'  DATASET="{args.output}"')


if __name__ == "__main__":
    main()
