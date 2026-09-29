# 개발자 안내 — `hannun` 패키지 구조·실행·검사

포트폴리오 소개는 [루트 README](../README.md). 이 문서는 코드를 열어 보거나 돌려 볼 사람을 위한
안내다. 원래 팀 모노레포의 `data/ds/` 가 파이썬 프로젝트 루트였고, 이 저장소에서는 저장소 루트가 그
자리다. 패키지 `hannun` 하나를 DS1 과 DS2 가 같이 만들었다.

## 데이터 흐름과 티켓

```
DE 공통 기사 JSON (ds_input/<날짜>.jsonl, 15분마다 최근 3일 재작성)
      │
      ▼  DS1 — 매시 :05 전체 재군집, :20/:35/:50 증분 배정 (48h 창)
ingest      검증 → Gold parquet (발행일 파티션, article_id 저장소 전체 유일)     99 · 132
preprocess  언론사별 규칙으로 본문 정제 → clean(content_clean)                  100 · 119
dedup       SHA-256 → 4-gram MinHash/LSH 후보 → TF-IDF 코사인·포함률 검증 → star 그룹   10 · 123 · 124
embed       multilingual-e5-small (제목+본문 앞 500자) → embedding, 입력 해시로 재사용   11 · 131
cluster     UMAP(창당 1회 고정 지도) + HDBSCAN + 2단계 새 이슈 탐지 → issue          89 · 90 · 128
assign      (15분 모드) 기존 이슈 중심 코사인 0.85 로 새 기사만 배정                    102
succeed     구성원 득표로 영속 이슈 ID 승계 → issue_registry                          97
quality     구조화 판정 · 노이즈 구제 · 품질 지표 → issue_quality                     93 · 91 · 104
feed        대표 기사 · hot score · 카테고리(팀 어휘 8개) → issue_summary              34 · 122
export      DS2 관점 분석을 이슈 단위 병렬로 → ds_output/<날짜>.jsonl + 완료 마커      122 · 123 · 132
      │
      ▼  DS2 — hannun.stance · enrichment (관점 그룹, 세부 견해 라벨, 브리핑)
      ▼  DE gold_issue_feed → BE
```

티켓 번호는 `ds1/tickets/S15P21E105-<번호>.md` 의 설계·실측 문서다. 운영 절차는
[ds1/EC2_RUNBOOK.md](ds1/EC2_RUNBOOK.md), 결정의 시간순 기록은 [ds1/WORKLOG.md](ds1/WORKLOG.md).

## 모듈

| 패키지 | 하는 일 | 소유 |
|---|---|---|
| `hannun.ingest` | 공통 기사 JSON 검증(pydantic), `GoldStore` 적재·읽기 | DS1 |
| `hannun.preprocess` | 정제 규칙(`rules.py`), `CleanStore` | DS1 |
| `hannun.dedup` | exact · candidates(MinHash/LSH, 서명 캐시) · verify · groups, `DedupStore` | DS1 |
| `hannun.embedding` | 인코더, `EmbeddingStore` (`input_sha1` 로 재사용 판정) | DS1 |
| `hannun.clustering` | `MapStore`(고정 UMAP 지도), clusterer, centroids, succession, `IssueStore`·`RegistryStore` | DS1 |
| `hannun.quality` | 구조화 판정·구제(`rules.py`), 증분 배정(`assign.py`), 골드셋 채점(`goldenset.py`) | DS1 |
| `hannun.feed` | 카테고리 정규화, hot score, `SummaryStore` | DS1 |
| `hannun.stance`, `hannun.enrichment` | 관점 그룹·세부 견해·브리핑 | DS2 |

각 패키지는 순수 로직 파일 + `pipeline.py`(단계 함수 하나) + `cli.py`(`hannun-<단계>` 명령) 로 이루어진다.
단계가 만든 컬럼은 뒤 단계가 이름을 바꾸거나 지우지 않는다. 저장은 전부 parquet 이고 스토어 클래스로만
읽고 쓴다(경로 문자열 조립 금지 — 파티션 규칙이 스토어 안에 있다). 기록은 임시 파일에 쓰고 교체한다.

## 디렉터리

```
src/hannun/<단계>/   패키지 (위 표)
scripts/             run_daily_chain.sh(운영 체인), export_ds2_jsonl.py(DS2 내보내기),
                     make_sample_data.py(합성 샘플), simulate_recluster_churn.py(재군집 출렁임 시뮬레이션),
                     score_goldenset_*.py(골드셋 채점), sweep_cluster_params.py(파라미터 스윕),
                     profile_dataset.py, make_*_labeling_html.py(라벨링 시트)
dags/                Airflow DAG — ds_chain(매시 :05)·ds_assign(:20/:35/:50), 호스트 ssh 로 docker run
tests/               pytest 406개, 모듈과 1:1. 인코더·UMAP 은 결정적 스텁이라 모델 없이 돈다
samples/             커밋되는 합성 샘플 입력
docs/ds1/            티켓 문서, WORKLOG, TROUBLESHOOTING, CODE_STYLE, EC2_RUNBOOK, LABEL_GUIDE, 골드셋 라벨링 자료
docs/ds2/            DS2 문서 (interface_ds1_to_ds2.md 는 08-27 제안안, 확정 계약은 contracts/)
docs/contracts/      DE → DS 기사 JSON, DS1 → DS2 이슈 표 계약
Dockerfile           운영 이미지 hannun-ds (ENTRYPOINT = scripts/run_daily_chain.sh)
gold/, local/        코드가 만드는 데이터·실험 산출. .gitignore
```

## 시작하기

파이썬 3.12 권장(EC2 와 동일, torch 호환). 임베딩·군집까지 돌리려면 `embedding`·`clustering` 그룹이 필요하다.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -e ".[dev]"                                  # 테스트만: 기본 + pytest·flake8
pip install -e ".[dev,embedding,clustering]"             # 전체 (torch CPU 수백 MB)

python scripts/make_sample_data.py                       # samples/articles_sample.jsonl 생성
hannun-ingest      --input samples/articles_sample.jsonl --gold-root gold -v
hannun-preprocess  --gold-root gold -v
hannun-dedup       --gold-root gold -v
hannun-embed       --gold-root gold -v
hannun-cluster     --gold-root gold --start-date 2026-08-20 --end-date 2026-08-21 -v
hannun-succeed     --gold-root gold --start-date 2026-08-20 --end-date 2026-08-21 -v
hannun-quality     --gold-root gold --start-date 2026-08-20 --end-date 2026-08-21 -v
hannun-feed        --gold-root gold --start-date 2026-08-20 --end-date 2026-08-21 -v
```

`hannun-assign` 은 15분 배정, `hannun-cluster --refit-map` / `--no-map` 은 지도 실험용이다. 전체 체인은
`scripts/run_daily_chain.sh` (`MODE=assign`, `START`/`END`, `INGEST_DAYS`, `GOLD_ROOT`, `DS_INPUT`, `DS_OUTPUT`).

## 산출물 읽기

모든 표는 `article_id` 또는 `issue_id` 로 조인한다.

```python
from hannun.ingest import GoldStore
from hannun.preprocess import CleanStore
from hannun.dedup import DedupStore
from hannun.embedding import EmbeddingStore

gold = GoldStore("gold").read("2026-08-20", "2026-08-21")                 # 원문
clean = CleanStore("gold").read("2026-08-20", "2026-08-21")               # content_clean, clean_status
dedup = DedupStore("gold").read("2026-08-20", "2026-08-21")               # duplicate_of, method, duplicate_count
vectors = EmbeddingStore("gold").read("2026-08-20", "2026-08-21")         # vector(384), input_sha1
```

뒤 단계는 원문이 필요한 곳(dedup 의 SHA-256)만 `content` 를 쓰고 나머지는 `content_clean` 을 쓴다.
이슈 표(`issue`, `issue_registry`, `issue_quality`, `issue_summary`)의 컬럼과 소비 규칙은
[contracts/ds1-to-ds2-issue-tables.md](contracts/ds1-to-ds2-issue-tables.md).

## 검사

```bash
pytest tests/ -q
flake8 src tests scripts dags --select=E9,F63,F7,F82,E501 --max-line-length=110
```

코드 규칙은 [ds1/CODE_STYLE.md](ds1/CODE_STYLE.md), 팀 시절의 브랜치·커밋·MR 규칙은
[GIT_CONVENTION.md](GIT_CONVENTION.md).
