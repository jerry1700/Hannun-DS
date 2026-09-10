# 한눈 DS 일일 배치 이미지 — 코드·의존성만 담는다. 데이터(gold, ds_input)와
# 모델 캐시는 호스트 볼륨으로 받아 이미지 재빌드와 무관하게 유지된다.
# 빌드·실행 절차는 docs/ds1/EC2_RUNBOOK.md §6.
FROM python:3.12-slim

# tzdata: run_daily_chain.sh 가 TZ=Asia/Seoul 로 ds_input 날짜를 계산한다 —
# 없으면 조용히 UTC 로 떨어져 새벽 실행이 전날 파일을 집는다
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# torch CPU 를 먼저 — 가장 무겁고(수백 MB) 가장 안 바뀌는 레이어를 캐시 최상단에
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir ".[embedding,clustering]"

COPY scripts/run_daily_chain.sh scripts/
RUN chmod +x scripts/run_daily_chain.sh

# 컨테이너 안 경로는 고정 — 호스트의 실제 위치는 docker run 의 -v 가 정한다.
# NUMBA_CACHE_DIR: umap 의 JIT 컴파일 결과를 볼륨에 남겨 매 실행 재컴파일을 피한다
ENV GOLD_ROOT=/data/gold \
    DS_INPUT=/data/ds_input \
    HANNUN_PY=python \
    HF_HOME=/data/hf_cache \
    NUMBA_CACHE_DIR=/data/hf_cache/numba

ENTRYPOINT ["./scripts/run_daily_chain.sh"]
