#!/usr/bin/env bash
# 한눈 DS 일일 배치 — ds_input 최근 N일을 적재하고 48h 창으로 전체 체인을 돌린다.
#
# DE 계약(S15P21E105-118): 매일 04:00 KST 에 ds_input/<YYYY>/<YYYY-MM-DD>.jsonl 이
# 올라오고 최근 3일은 매일 다시 쓰인다(늦게 오는 기사). 하루는 항상 파일 하나.
# 모든 단계가 멱등이라 같은 창을 다시 처리해도 안전하다.
# cron 예(KST 서버): 30 4 * * * — 시각은 서버 TZ 기준, 등록 전 timedatectl 확인(TS-013)
#
#   ./run_daily_chain.sh                                  # 오늘 기준 48h 창, 전체 재군집
#   MODE=assign ./run_daily_chain.sh                      # 15분 배정 모드 — 재군집 대신 새 기사 배정(102)
#   START=2026-09-01 END=2026-09-02 ./run_daily_chain.sh  # 특정 창 재처리(백필)
#   INGEST_DAYS=0 ./run_daily_chain.sh                    # 적재 생략(체인만)
set -euo pipefail

GOLD_ROOT="${GOLD_ROOT:-$HOME/gold}"
DS_INPUT="${DS_INPUT:-$HOME/ds_input}"
DS_OUTPUT="${DS_OUTPUT:-$HOME/ds_output}"   # DS2 관점·발행용 JSONL (DE gold_issue_feed 가 읽음)
PY="${HANNUN_PY:-$HOME/S15P21E105/data/ds/.venv/bin/python}"
INGEST_DAYS="${INGEST_DAYS:-3}"   # ds_input 재작성 주기(최근 3일)와 맞춘다
WINDOW_DAYS="${WINDOW_DAYS:-2}"   # 48h 창
MODE="${MODE:-recluster}"         # recluster(매시) | assign(15분 — 경계 출렁임 없이 새 기사만 붙임)

start="${START:-$(date -u -d "$((WINDOW_DAYS - 1)) days ago" +%F)}"
end="${END:-$(date -u +%F)}"
echo "[chain] $(date -u +%FT%TZ) window ${start} ~ ${end} (UTC) mode=${MODE}"

for i in $(seq 0 $((INGEST_DAYS - 1))); do
    day=$(TZ=Asia/Seoul date -d "-${i} day" +%F)
    file="${DS_INPUT}/${day%%-*}/${day}.jsonl"
    legacy_dir="${DS_INPUT}/${day%%-*}/${day}"   # 구 구조(<날짜>/articles*.jsonl) 전환기 대비
    if [ -f "$file" ]; then
        "$PY" -m hannun.ingest.cli -g "$GOLD_ROOT" -i "$file"
    elif compgen -G "${legacy_dir}/*.jsonl" > /dev/null; then
        "$PY" -m hannun.ingest.cli -g "$GOLD_ROOT" -i "${legacy_dir}"/*.jsonl
    else
        echo "[chain] 입력 없음: ${file} (건너뜀)"
    fi
done

"$PY" -m hannun.preprocess.cli            -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
"$PY" -m hannun.dedup.cli                 -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
"$PY" -m hannun.embedding.cli             -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
if [ "$MODE" = "assign" ]; then
    # 15분 배정: 전체 재군집은 매번 UMAP 지도를 다시 그려 경계의 작은 이슈가 쪼개졌다
    # 붙었다 한다(실측 created 40~57/실행). 배정은 기존 이슈에 새 기사만 붙여 그 출렁임이
    # 없다. 이슈 표가 아직 없으면(자정 직후 첫 창) 배정할 곳이 없으니 재군집으로 폴백
    "$PY" -m hannun.quality.assign_cli        -g "$GOLD_ROOT" --start-date "$start" --end-date "$end" \
        || "$PY" -m hannun.clustering.cli     -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
else
    "$PY" -m hannun.clustering.cli            -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
fi
# 배정 뒤에도 승계를 돌려야 새로 붙은 기사가 registry(DE 피드의 멤버십)에 들어간다
"$PY" -m hannun.clustering.succession_cli -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
"$PY" -m hannun.quality.cli               -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
"$PY" -m hannun.feed.cli                  -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"

# DS2 관점 분석 + BE 연동 JSONL 내보내기 (S15P21E105-122) — DE 의 gold_issue_feed 가
# 이 파일을 이슈 피드에 병합한다. 파일명은 실행일(KST) 하나, 재실행은 원자적 덮어쓰기
mkdir -p "$DS_OUTPUT"
"$PY" scripts/export_ds2_jsonl.py --gold-root "$GOLD_ROOT" \
    --start "$start" --end "$end" --window "$start" \
    --output-day "$(TZ=Asia/Seoul date +%F)" --output-root "$DS_OUTPUT" --overwrite
echo "[chain] done $(date -u +%FT%TZ)"
