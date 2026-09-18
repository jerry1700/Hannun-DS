#!/usr/bin/env bash
# 한눈 DS 체인 — ds_input 최근 N일을 적재하고 48h 창으로 전체 단계를 돌린다.
#
# DE 계약(S15P21E105-118): article_ingest 가 15분마다 ds_input/<YYYY>/<YYYY-MM-DD>.jsonl 을
# 내려주고 최근 3일은 매번 다시 쓴다(늦게 오는 기사). 하루는 항상 파일 하나.
# 모든 단계가 멱등이라 같은 창을 다시 처리해도 안전하다.
# 호출은 Airflow 가 한다 — ds_chain(매시 :05, 재군집)·ds_assign(:20/:35/:50, 배정). docs/ds1/EC2_RUNBOOK.md §7
#
#   ./run_daily_chain.sh                                  # 오늘 기준 48h 창, 전체 재군집
#   MODE=assign ./run_daily_chain.sh                      # 15분 배정 모드 — 재군집 대신 새 기사 배정(티켓 102)
#   START=2026-09-01 END=2026-09-02 ./run_daily_chain.sh  # 특정 창 재처리(백필)
#   INGEST_DAYS=0 ./run_daily_chain.sh                    # 적재 생략(체인만)
set -euo pipefail

GOLD_ROOT="${GOLD_ROOT:-$HOME/gold}"
DS_INPUT="${DS_INPUT:-$HOME/ds_input}"
DS_OUTPUT="${DS_OUTPUT:-$HOME/ds_output}"   # DS2 관점·발행용 JSONL (DE gold_issue_feed 가 읽음)
PY="${HANNUN_PY:-$HOME/S15P21E105/data/ds/.venv/bin/python}"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"    # 어느 디렉토리에서 불려도 같은 폴더의 스크립트를 찾는다
INGEST_DAYS="${INGEST_DAYS:-3}"   # ds_input 재작성 주기(최근 3일)와 맞춘다
WINDOW_DAYS="${WINDOW_DAYS:-2}"   # 48h 창
MODE="${MODE:-recluster}"         # recluster(매시) | assign(15분 — 경계 출렁임 없이 새 기사만 붙임)

start="${START:-$(date -u -d "$((WINDOW_DAYS - 1)) days ago" +%F)}"
end="${END:-$(date -u +%F)}"
echo "[chain] $(date -u +%FT%TZ) window ${start} ~ ${end} (UTC) mode=${MODE}"

# 단계별 소요를 남긴다 — 어느 단계가 병목인지 로그만으로 알 수 있게(티켓 123). 종료 코드를
# 그대로 돌려줘 `step a … || step b …` 폴백이 동작한다
step() {
    local name=$1 t0 rc
    shift
    t0=$(date +%s)
    "$@"; rc=$?
    echo "[chain] step ${name} $(( $(date +%s) - t0 ))s rc=${rc}"
    return $rc
}

for i in $(seq 0 $((INGEST_DAYS - 1))); do
    day=$(TZ=Asia/Seoul date -d "-${i} day" +%F)
    file="${DS_INPUT}/${day%%-*}/${day}.jsonl"
    if [ -f "$file" ]; then
        step "ingest:${day}" "$PY" -m hannun.ingest.cli -g "$GOLD_ROOT" -i "$file"
    else
        echo "[chain] 입력 없음: ${file} (건너뜀)"
    fi
done

step preprocess "$PY" -m hannun.preprocess.cli -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
step dedup      "$PY" -m hannun.dedup.cli      -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
step embed      "$PY" -m hannun.embedding.cli  -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
if [ "$MODE" = "assign" ]; then
    # 15분 배정: 전체 재군집은 매번 UMAP 지도를 다시 그려 경계의 작은 이슈가 쪼개졌다 붙었다
    # 한다(티켓 102 주말 실측). 배정은 기존 이슈에 새 기사만 붙여 그 출렁임이 없다.
    # 이슈 표가 아직 없으면(자정 직후 첫 창) 배정할 곳이 없으니 재군집으로 폴백
    step assign  "$PY" -m hannun.quality.assign_cli -g "$GOLD_ROOT" --start-date "$start" --end-date "$end" \
        || step cluster "$PY" -m hannun.clustering.cli -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
else
    step cluster "$PY" -m hannun.clustering.cli -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
fi
# 배정 뒤에도 승계를 돌려야 새로 붙은 기사가 registry(DE 피드의 멤버십)에 들어간다
step succeed "$PY" -m hannun.clustering.succession_cli -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
step quality "$PY" -m hannun.quality.cli -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"
step feed    "$PY" -m hannun.feed.cli    -g "$GOLD_ROOT" --start-date "$start" --end-date "$end"

# DS2 관점 분석 + BE 연동 JSONL 내보내기 (S15P21E105-122) — DE 의 gold_issue_feed 가
# 이 파일을 이슈 피드에 병합한다. 파일명은 실행일(KST) 하나, 재실행은 원자적 덮어쓰기.
# 구성원·대표 제목·카테고리가 안 바뀐 이슈는 두 모드 모두 이전 결과를 재사용한다(티켓 123).
# 재군집은 UMAP 을 다시 그려 3분의 2가 바뀌므로 재사용이 적고, 관점 라벨(120)이 붙은 뒤 이슈당
# 분석이 3배 무거워져 이슈 단위로 3개 프로세스에 나눈다(4 vCPU 중 하나는 남긴다 — 티켓 123 §8).
# 하루 첫 실행은 그날 파일이 없어 자연히 전량 재분석이라 DS2 코드 변경은 다음 날 자정에 반영된다
EXPORT_WORKERS="${EXPORT_WORKERS:-3}"
mkdir -p "$DS_OUTPUT"
step export "$PY" "$SCRIPTS/export_ds2_jsonl.py" --gold-root "$GOLD_ROOT" \
    --start "$start" --end "$end" --window "$start" \
    --output-day "$(TZ=Asia/Seoul date +%F)" --output-root "$DS_OUTPUT" --overwrite --reuse-unchanged \
    --workers "$EXPORT_WORKERS"
echo "[chain] done $(date -u +%FT%TZ)"
