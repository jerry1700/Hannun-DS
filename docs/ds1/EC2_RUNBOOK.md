# EC2 운영 런북 — DS 파이프라인 (티켓 101)

A 서버(EC2 t3.xlarge, Ubuntu 24.04, 4 vCPU·16GB·GPU 없음)에서 실수집 데이터로
전체 체인을 돌리는 절차. 입력 계약은
[contracts/de-to-ds-article-json.md](../../../docs/contracts/de-to-ds-article-json.md).

## 1. 최초 셋업 (한 번만)

```bash
git clone https://lab.ssafy.com/s15-bigdata-dist-sub1/S15P21E105.git ~/S15P21E105
cd ~/S15P21E105/data/ds
python3 --version                     # 3.12.x 확인 (Ubuntu 24.04 기본)
python3 -m venv .venv
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 빌드(~200MB)
./.venv/bin/pip install -e ".[embedding,clustering]"
./.venv/bin/python -m pytest tests/ -q                                           # 전부 통과 확인
chmod +x scripts/run_daily_chain.sh
```

임베딩 모델(e5-small-ko-v2, 수백 MB)은 첫 실행 때 HuggingFace 에서 자동 다운로드된다.

## 2. 첫 스모크 (수동 1회)

```bash
cd ~/S15P21E105/data/ds
GOLD_ROOT=~/gold DS_INPUT=~/ds_input HANNUN_PY=./.venv/bin/python \
  ./scripts/run_daily_chain.sh 2>&1 | tee ~/chain_smoke.log
```

확인 순서:
1. ingest 출력의 `written`/`skipped_existing`/리젝트 수 — 리젝트가 많으면
   `~/gold/rejects/` 의 사유부터 (published 형식·빈 본문)
2. 각 단계 JSON stats — dedup `duplicates`, cluster `issues`/`noise`,
   quality `structured`/`rescued`, feed `issues`
3. `~/gold/issue_summary/` 가 생겼으면 성공 — BE 가 소비할 최종 산출물

## 3. 정기 실행 (cron)

DE 가 04:00 KST 에 ds_input 을 갱신하므로 04:30 KST 에 돌린다. 서버 TZ 가 UTC 면:

```cron
30 19 * * * cd $HOME/S15P21E105/data/ds && GOLD_ROOT=$HOME/gold DS_INPUT=$HOME/ds_input HANNUN_PY=./.venv/bin/python ./scripts/run_daily_chain.sh >> $HOME/hannun_chain.log 2>&1
```

- 멱등 설계라 겹쳐 돌아도 데이터는 안전하지만, **동시 기동은 피한다**(이전 실행이
  안 끝났으면 다음 턴을 건너뛰는 게 안전 — flock 예:
  `flock -n /tmp/hannun_chain.lock ./scripts/run_daily_chain.sh`)
- 백필: `START=2026-09-01 END=2026-09-02 INGEST_DAYS=0 ./scripts/run_daily_chain.sh`

## 4. 운영 순서와 산출물

```
ingest → preprocess → dedup → embed → cluster → succeed → quality → feed
gold/articles → clean → dedup → embedding → issue → issue_registry → issue_quality → issue_summary
```

BE 소비 지점: `issue_summary/`(피드: issue_id·대표 기사·hot_score) +
`issue_registry/`(기사→issue_id) + `issue_quality/`(기사 단위 최종 배정).

## 5. 아직 안 붙인 것

- **15분 배정 루프**(`hannun-assign`): ds_input 이 일 단위라 보류.
  DE 가 15분 단위 내려주기를 제공하면 ingest→preprocess→embed→assign→quality 로 붙인다
- 코드 갱신 반영: `git pull` 후 `./.venv/bin/pip install -e ".[embedding,clustering]"`
  재실행 (pyproject 의 콘솔 스크립트가 바뀌었을 수 있음 — TS-011)

## 6. 도커 전환 (승계 실전 검증 후)

venv 대신 이미지로 돌린다. 바뀌는 것은 "무엇을 실행하나"뿐 — 데이터(`~/gold`,
`~/ds_input`)와 모델 캐시는 호스트 볼륨으로 남고, cron·flock·로그도 호스트 유지.

```bash
cd ~/S15P21E105/data/ds
docker build -t hannun-ds .

# 스모크 (수동 1회) — venv 실행과 같은 창을 재처리하니 멱등으로 안전
docker run --rm -u "$(id -u):$(id -g)" \
  -v ~/gold:/data/gold -v ~/ds_input:/data/ds_input \
  -v ~/.cache/huggingface:/data/hf_cache \
  hannun-ds
```

- `-u`(호스트 사용자로 실행)를 빼먹으면 산출물이 root 소유가 돼 venv 병행 사용이 꼬인다
- 백필·옵션은 `-e` 로 전달: `-e START=2026-09-01 -e END=2026-09-02 -e INGEST_DAYS=0`
- 모델 캐시는 venv 시절 것(`~/.cache/huggingface`)을 그대로 마운트 — 재다운로드 없음

스모크 확인 후 cron 은 실행 명령만 교체:

```cron
30 19 * * * flock -n /tmp/hannun_chain.lock docker run --rm -u 1000:1000 -v $HOME/gold:/data/gold -v $HOME/ds_input:/data/ds_input -v $HOME/.cache/huggingface:/data/hf_cache hannun-ds >> $HOME/hannun_chain.log 2>&1
```

코드 갱신 반영은 `git pull` 후 `docker build -t hannun-ds .` 재빌드(레이어 캐시로
의존성은 건너뛰고 코드만 다시 담는다 — 수십 초).
