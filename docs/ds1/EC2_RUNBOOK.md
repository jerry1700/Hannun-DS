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

DE 가 04:00 KST 에 ds_input 을 갱신하므로 04:30 KST 에 돌린다. 서버 TZ 는
Asia/Seoul (2026-09-10 확인). crontab 시각은 서버 TZ 기준이므로 등록 전
`timedatectl` 로 확인하고, TZ 가 바뀐 서버라면 `sudo systemctl restart cron`
까지 해야 새 TZ 로 발화한다(cron 은 기동 시점 TZ 를 캐시 — TS-013).

```cron
30 4 * * * cd $HOME/S15P21E105/data/ds && GOLD_ROOT=$HOME/gold DS_INPUT=$HOME/ds_input HANNUN_PY=./.venv/bin/python ./scripts/run_daily_chain.sh >> $HOME/hannun_chain.log 2>&1
```

- 멱등 설계라 겹쳐 돌아도 데이터는 안전하지만, **동시 기동은 피한다**(이전 실행이
  안 끝났으면 다음 턴을 건너뛰는 게 안전 — flock 예:
  `flock -n /tmp/hannun_chain.lock ./scripts/run_daily_chain.sh`)
- 백필: `START=2026-09-01 END=2026-09-02 INGEST_DAYS=0 ./scripts/run_daily_chain.sh`

## 4. 운영 순서와 산출물

```
ingest → preprocess → dedup → embed → cluster(또는 assign) → succeed → quality → feed → export
gold/articles → clean → dedup(+dedup_sig) → embedding → issue(+umap_map) → issue_registry → issue_quality
  → issue_summary → ~/ds_output/<KST 날짜>.jsonl (+ _chain_status.json 완료 마커, 티켓 132)
```

BE 소비 지점: `issue_summary/`(피드: issue_id·대표 기사·hot_score) +
`issue_registry/`(기사→issue_id) + `issue_quality/`(기사 단위 최종 배정). DE 는 `ds_output/`
을 병합해 발행한다. `dedup_sig/`(서명 캐시, 티켓 123)와 `umap_map/`(창별 고정 지도, 티켓 128)은
중간 산출이라 하류가 읽지 않는다 — 지우면 다음 실행이 다시 만든다. `umap_map/` 은 창별로
`current` 포인터와 `gen=…` 세대 디렉터리 하나(티켓 132) — 세대가 둘 보이면 쓰다 죽은 것이고
다음 재학습이 치운다. `ds_output/_chain_status.json` 은 마지막 체인의 status(ok/failed)·창·단계별
소요를 담는다 — 체인이 끝났는지는 로그 대신 이 파일로 본다.

## 5. 코드 갱신 반영

- 컨테이너 운영(§6)이라 `git pull && docker build -t hannun-ds .` — 코드만 바뀌면 수 초
- 고정 지도를 지금 벡터로 다시 학습해야 하면 `hannun-cluster --refit-map`, 옛 방식(매번 재학습)으로
  돌려 보려면 `--no-map`. 둘 다 실험용이고 체인은 기본값을 쓴다
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
30 4 * * * flock -n /tmp/hannun_chain.lock docker run --rm -u 1000:1000 -v $HOME/gold:/data/gold -v $HOME/ds_input:/data/ds_input -v $HOME/.cache/huggingface:/data/hf_cache hannun-ds >> $HOME/hannun_chain.log 2>&1
```

코드 갱신 반영은 `git pull` 후 `docker build -t hannun-ds .` 재빌드(레이어 캐시로
의존성은 건너뛰고 코드만 다시 담는다 — 수십 초).

## 7. Airflow 전환 — DE 트리거로 이어 돌리기

체인은 **DS 소유의 별도 DAG** `ds_chain`(**`ds/dags/ds_chain_dag.py`** — compose 가
`ds/dags` 를 `/opt/airflow/dags/ds` 로 마운트, 스케줄 없음)에 있고, DE 의
`ds_daily`(04:00 KST)가 TriggerDagRunOperator 로 켜서 완료를 기다린 뒤 gold 이슈
피드(69)를 잇는다. ds_input 내려주기는 `article_ingest` DAG 가 **15분마다**
해두므로(118, 2026-09-11 개편) 체인 시작 시점에 입력은 이미 최신이다:

```
article_ingest (:00/:15/:30/:45):  수집 → silver → ds_input 내려주기        ← DE
ds_chain  (매시 :05):              전체 재군집 체인 (ssh → docker run hannun-ds)  ← DS1
ds_assign (:20/:35/:50):           배정 체인 (같은 이미지, -e MODE=assign)        ← DS1
ds_daily  (04:00):                 [trigger] ds_chain → gold_issue_feed          ← DE
```

주기 설계(102 §8): 15분 전체 재군집은 경계 출렁임(실행당 새 ID 40\~50)이 실측돼
철회. 매시 재군집이 새 이슈를 만들고, 그 사이 15분 배정이 새 기사를 기존 이슈에
붙인다(못 붙는 기사는 정시까지 노이즈). 두 DAG 는 같은 flock 을 10분 대기로 나눠
쓴다. :05/:20 출발은 article_ingest 가 파일을 다 쓴 뒤(:03 경) 읽기 위해서.

트리거 주체 이력: ds_export(내려주기+트리거 통합, ~09-10) → ds_daily 로 분리
(09-11, ds_export 는 paused 로 잔존). dag_id `ds_chain` 계약 덕에 우리 쪽 변경 0.

파일 경계: 사슬 내용이 바뀌어도 서로의 DAG 파일을 열지 않는다(dag_id `ds_chain`
이 계약, DS DAG 는 ds/ 폴더에서 관리). 실행은 호스트 ssh — docker.sock 마운트
(관리자 권한)를 피하고 backup_to_b 의 키 패턴을 재사용한다. compose 변경
(extra_hosts·키 마운트·ds/dags 마운트)만 공유 파일이고, 서버에서는 키만 만들면
된다. compose 의 마운트가 바뀐 날은 `docker compose up -d airflow` 로 컨테이너
재생성까지 해야 반영된다:

```bash
# 1. airflow → 호스트 ssh 키 (한 번만)
ssh-keygen -t ed25519 -f ~/.ssh/hannun-airflow -N "" -C "airflow-to-host-ds-chain"
cat ~/.ssh/hannun-airflow.pub >> ~/.ssh/authorized_keys
sudo install -m 600 -o 50000 -g 50000 ~/.ssh/hannun-airflow ~/.ssh/hannun-airflow.pem
rm ~/.ssh/hannun-airflow                     # 컨테이너용 사본(50000 소유)만 남긴다

# 2. compose 변경 반영 (airflow 컨테이너 재생성)
cd ~/S15P21E105/data/de/infra
docker compose up -d airflow

# 3. 배관 확인 — 컨테이너에서 호스트 docker 가 보이면 통과
docker compose exec airflow ssh -i /opt/airflow/.ssh/hannun.pem \
  -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
  ubuntu@host.docker.internal 'docker image ls hannun-ds'
```

전환 절차(완료): 첫날은 cron(§6)을 폴백으로 병행하고, Airflow 정기 실행이 초록으로
확인된 다음날 cron 줄을 지운다 — **2026-09-11 전환 완료, cron 제거됨.** 이후 실행
주체는 Airflow 뿐이고, 로그는 Airflow 태스크 로그와 `~/hannun_chain.log`(tee)
양쪽에 남는다. 수동 실행이 필요하면 UI 에서 ds_chain 을 직접 트리거하거나 §6 의
docker run 을 쓴다.
