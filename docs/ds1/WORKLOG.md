# 작업 기록 (DS1)

> 날짜별로 "무엇을 했고, 무엇을 결정했고, 다음에 뭘 할지"를 남긴다. 문제 해결 과정은 [TROUBLESHOOTING.md](TROUBLESHOOTING.md), 티켓별 상세는 [tickets/](tickets/).

---

## 2026-08-28

### 한 일
- **티켓 103 마무리** — 라벨 가이드 초안 MR 머지. 지라 연동 사고(TS-008)로 103 을 다시 열어 브랜치 재생성 후 진행.
- **과거 기사 CSV 수령·프로파일** — `ssafy_dataset_news_2023.csv`(4.4GB) · `2024_1st_half.csv`(1.7GB). 열어볼 수 없어 스트리밍 집계 스크립트(`scripts/profile_dataset.py`)를 만들어 훑음: 283만 건, 언론사 34개, `|` 구분, published 형식 4종(타임존 없음), 본문 빈값 2.4%/8.0%, 언론사별 보일러플레이트 고정. 정리는 [DATASET_PROFILE.md](DATASET_PROFILE.md).
- **DE 스키마 대조** — `de/schemas/article_v1.json` 이 BE 문서와 다른 곳 셋: `crawled_at` 필수 추가(우리는 무시하고 통과), `source_type` 에 `HTML`(우리 Enum 이 리젓트하던 것 → 허용), `article_id` 해시에 쿼리 제거 URL(디지털타임·국민일보는 쿼리가 기사 번호라 전부 같은 id 가 됨 → 우리는 추적 파라미터만 제거하고 계약 문서로 제기). `schema.py` 수정, 테스트 5개 추가.
- **S15P21E105-100 본문 전처리 구현** — `hannun.preprocess`: 언론사 × 패턴 규칙 표 30여 개, Gold 옆에 `clean/` 테이블(`content_clean`, `clean_status`, `rules_applied`), CLI `hannun-preprocess`. 실데이터 표본 5,105건(언론사별 100~200건, `scripts/csv_to_common_jsonl.py` 로 공통 JSON 변환)에 여섯 번 돌려 잔재를 보며 규칙을 다듬음: 첫 시도에서 한 줄 기사 82건이 저작권 줄 규칙에 통째로 지워진 것, 이메일 정규식의 `\w` 가 한글을 먹은 것, zero-width space 가 규칙을 빗나가게 한 것(TS-010)을 잡음. 최종 ok 4,981 · short 104 · empty 20, 이메일 잔재 13건은 대부분 본문의 연락처. 상세는 [tickets/S15P21E105-100.md](tickets/S15P21E105-100.md).
- **DE↔DS 계약 초안** — `data/docs/contracts/de-to-ds-article-json.md`: 필드별 DS 검증 규칙, URL 정규화 문제 제기, 과거 CSV 변환 규칙 제안 표(publisher_id 부여표·published 4형식·category·빈 본문·link 중복).
- Git Bash 히어독에서 `\\` 가 `\` 로 줄어 정규식 이스케이프가 깨지는 함정을 두 번 당함 (TS-009).

### 결정
| 결정 | 이유 |
|---|---|
| 정제 결과는 Gold 컬럼이 아닌 별도 `clean/` 테이블, `article_id` 조인 | 규칙이 바뀌면 clean 만 재생성. Gold 스키마 불변 |
| 저작권은 줄 삭제가 아니라 표지부터 줄 끝까지 자르기 | 한 줄짜리 기사(파이낸셜·머니S·뉴스1)가 통째로 사라지는 것을 봄 |
| 문장 사이 공백 복원(`sentence_space`)을 정제에 포함 | DS2 문장 분리기가 `\s+` 를 요구. KBS·뉴스1 원문은 공백이 없음 |
| URL 정규화는 추적 파라미터만 제거 | 쿼리가 기사 번호인 언론사가 있음. DE 규칙과의 차이는 `article_id_verified` 로 드러남 |
| 정형 콘텐츠(출연자 목록·시황·운세)는 전처리가 아니라 STEP 4 몫 | 지울 근거가 문장 형태에 없음 |

- **DS 입력 경로·역할 경계 확정** — DE 확인: DS 는 Kafka 를 직접 구독하지 않고 **HDFS 에 쌓인 파일을 읽는다**. DE 의 전처리는 **형식 통일까지**(`article_v1.json` 규격), **본문 정제는 DS1**. 알고리즘은 Spark 위에서 파이썬으로 돈다. 남은 것은 루트 경로·파일 형식·파일이 닫히는 주기, Spark 실행 환경.

### 다음
- [ ] DE 와 계약 문서 검토 — URL 정규화, CSV 변환 규칙 확정, HDFS 경로·형식·주기
- [ ] DE 크롤러(trafilatura) 결과로 전처리 재검증 — 과거 CSV 와 잔재가 다를 것
- [ ] 라벨 가이드 "제안" 항목 회의, 라벨링 주간 일정
- [ ] S15P21E105-10 중복 검출(STEP 1) — `content`(원문) 로 SHA-256, `content_clean` 으로 MinHash

---

## 2026-08-27

### 한 일
- **BE 문서(Kafka 수집 구조 · 공통 기사 JSON · API 스펙) 검토** → DS1 설계 문서와 대조
  - 설계 문서의 전제 "BE가 DB에 저장 → DS가 DB 읽기"가 실제와 다름. 실제: BE → Kafka `news.article.realtime.v1` → DE(HDFS). DS 입력은 공통 기사 JSON.
  - `published_at`은 **발행 시각**(RFC 3339 UTC)으로 확정. 크롤링 시각 필드는 없음.
  - API 응답 필드에서 DS1 산출물을 역산: `issueId`(안정적이어야 함), `pressCount`, `articleCount`, `score`(화제성 — 설계 문서에 없던 항목), 이슈 단위 `category`, `firstReportedAt/lastReportedAt`.
  - 상세는 [tickets/S15P21E105-99.md](tickets/S15P21E105-99.md) "BE 문서에서 확인한 것" 절.
- **프로젝트 골격 생성**: `src/hannun` 패키지(src 레이아웃), pyproject, venv(Python 3.14), pytest.
- **S15P21E105-99 공통 기사 JSON 입력 변환·Gold 저장 구현**
  - `schema.py` pydantic 검증 / `reader.py` JSON·JSONL·디렉토리 읽기 / `gold.py` parquet 파티션 저장소 / `pipeline.py` / `cli.py`
  - 샘플 데이터 10건(정상 5 + 배치 내 중복 1 + 리젝트 4) 생성 스크립트, 테스트 21개 통과.
  - CLI 실행 결과: records=10, valid=6, rejected=4, duplicates_in_batch=1, written=5, id_mismatch=1

- **코드 스타일 제정** — 이 저장소에 맞춰 `CODE_STYLE.md` 를 처음부터 씀 (이전 프로젝트 규칙은 참고만). ingest 모듈 전체를 그 규칙으로 다시 씀.
  - 처음 짠 코드에서 걸린 것: 모든 함수의 반환 타입, `Optional`/`from __future__`, 불릿·표가 든 docstring, 길이가 제각각인 구분선, 로그의 `%` 서식, 컬럼마다 붙은 설명 주석, 스크립트의 `sys.path` 조작.
  - 앞 단계를 위해 미리 정한 것: 임계값은 설정 dataclass 로 받기, 원본 컬럼 보존·파생 컬럼 추가, 난수 seed 고정과 파라미터 기록, 건별 문제는 예외 대신 리젝트, TODO 에 티켓 번호.
  - flake8(E9,F63,F7,F82,E501 / 110자) 검사 추가. 테스트 21개 그대로 통과.
- **팀 저장소 연동** — `lab.ssafy.com/s15-bigdata-dist-sub1/S15P21E105`. 모노레포(루트에 `back/ data/ front/ infra/`)라 우리 코드를 전부 `data/` 아래로 이동. 브랜치 `data/feat/S15P21E105-99-json-gold`(원격에 `data/dev` 기준으로 이미 있었음).
  - 커밋·MR 규칙을 `docs/GIT_CONVENTION.md`로 정함. BE가 이미 쓰는 `type: 제목 (지라키)` 형식과 같게 맞춤. 루트 README의 지라 연동 규칙(브랜치명에 키 → 진행 중, MR에 `Closes 키` → 완료) 반영.
  - 커밋 작성자를 다른 팀원 계정으로 잘못 넣어 푸시했다가 `jedabin` 으로 전부 정정 ([TS-006](TROUBLESHOOTING.md#ts-006)).
- **서비스명 확정 → 패키지 이름 변경** — 임시명 Alzza 대신 서비스명 '한눈'이 정해져 패키지를 `alzza` → `hannun`으로 바꿈(`git mv`로 이력 유지). 명령은 `hannun-ingest`. 로컬 폴더도 `Alzza/` → `hannun/`으로 바꿈.

- **데이터 파트 폴더 구조 정리** — `data/` 바로 아래에 pyproject 가 있어 파트 전체가 DS1 프로젝트처럼 보였다. 처음엔 사람별 `ds1/`로 옮겼다가, DS1·DS2 가 패키지를 같이 쓰는 게 낫다고 보고 **`data/ds/` 공용**으로 조정. 개인 문서(WORKLOG·TROUBLESHOOTING·tickets·CODE_STYLE)는 `ds/docs/ds1/`, `ds/docs/ds2/`로, `ds/README.md`는 둘이 함께 쓰는 문서로 다시 씀. 파트 공통은 `data/docs/`(GIT_CONVENTION, contracts/).

- **라벨링 파일럿 자료 정리** — 8/24에 6명이 30건을 독립 라벨한 결과(`Downloads/cluster_<이름>.csv`, `stance_report.html`)를 `docs/ds1/labeling/pilot-2026-08/`에 라벨만 남겨 옮김(`labels.csv` 180행, `articles_index.csv`). 기사 원문 `news30.jsonl`은 `.gitignore`. 집계: 이슈 묶음 쌍별 일치 95%(경계 2종 — 정상회담 3건을 한미훈련에 합칠지 3:3, 노경필 해명 2건 분리), 스탠스 일치 49%(만장일치 2/30). 스탠스가 갈린 건 논제를 무엇으로 잡는지가 안 정해져서 — 용산 이슈에서 두 사람 라벨이 정반대. 상세는 [labeling/pilot-2026-08/README.md](labeling/pilot-2026-08/README.md). 티켓 103 가이드의 경계 사례 재료.
- **티켓 103 MR 머지와 지라 연동 사고** — 파일럿 자료 MR(!7)을 `data/dev`에 머지. 지라가 완료로 안 바뀐 원인은 첫 커밋(99 키)에서 자동 생성된 MR 제목 ([TS-008](TROUBLESHOOTING.md#ts-008)). 수동 완료 후, 가이드 본문이 빠졌으므로 103을 다시 열고 브랜치 재생성.
- **라벨 가이드 초안** — 팀이 합의한 관점 판정 기준(판정 축·기준 6·보조 기준 4·판례 7·소분류·"같은 국면 통합")을 [LABEL_GUIDE.md](LABEL_GUIDE.md)로 옮기고, 파일럿에서 드러난 빈 곳을 "제안"으로 추가: 이슈마다 논제 문장 기록, 국면 통합 판정 테스트(원인·목적 명시 + 기간 겹침), 반응·해명 기사 규칙, 찬반 없는 발언 전달 = 중립, 중복 `동일`/`재작성`, 라벨 파일 형식, 배정·조정 절차. 판례에 article_id를 붙여 104가 기준점으로 쓸 수 있게 함. 결정 이유와 알고리즘 영향은 [tickets/S15P21E105-103.md](tickets/S15P21E105-103.md).
- **폴더 이름 변경 후 테스트 복구** — 로컬 폴더를 `hannun/`으로 바꾸자 editable 설치가 옛 경로를 가리켜 `import hannun` 실패. `pip install -e ".[dev]"` 재실행으로 복구 ([TS-007](TROUBLESHOOTING.md#ts-007)). 테스트 21개·flake8·CLI 샘플 적재(2회 투입 시 `skipped_existing=5`) 재확인.

### 결정
| 결정 | 이유 |
|---|---|
| DS1·DS2 는 `data/ds/` 에서 패키지 `hannun` 하나를 공유, 모듈 단위로 소유 | DS2 의 관점 대조는 DS1 의 클러스터·임베딩·Gold 위에서 돎. 폴더를 갈라 두 pyproject 로 가면 스키마 바뀔 때마다 두 곳을 맞춰야 함. 개인 문서만 `docs/ds1/`, `docs/ds2/` 로 분리. 다른 파트(DE·BE)와는 `data/docs/contracts/` 계약으로 |
| 패키지 영문명 `hannun` | '한눈'은 식별자로 못 쓰고, 국어의 로마자 표기법(한=han, 눈=nun)을 따르면 사람마다 다르게 적을 여지가 없음 |
| 우리 데이터 폴더 `data/samples`·`data/gold` → `samples/`·`gold/` | 모노레포 파트 디렉토리 `data/`와 이름이 겹쳐 `data/data/...`가 됨 ([TS-005](TROUBLESHOOTING.md#ts-005)) |
| 전처리(보일러플레이트 제거 등)는 DS1 소유 | 팀 결정. BE 문서의 "DE 실시간 전처리" 범위는 DE와 확인 필요 |
| Gold = "검증·타입 정규화·중복 제거가 끝난 기사 테이블", 원문 `content` 보존 | 전처리는 티켓 100. Gold를 두 번 만들지 않기 위해 원문을 여기 두고 정제 컬럼은 나중에 추가 |
| 저장 포맷 parquet, `published_date`(UTC) 파티션, 파티션당 파일 1개 | STEP 1·3이 시간 윈도우로 읽으므로 날짜 파티션이 자연스러움. HDFS 확정 전까지 로컬 |
| `article_id` 기준 멱등, 기본 `on_conflict=keep` | 같은 파일 재투입 시 결과 불변. 재크롤링 반영은 `--on-conflict replace`로 명시적으로 |
| `published_at` 타임존 없으면 리젝트, 오프셋 있으면 UTC 변환 | 발행 시각이 윈도우 기준. 모호한 값을 받으면 이슈가 잘못 묶임 |
| `article_id`가 BE 해시 규칙과 달라도 리젝트하지 않고 `article_id_verified=False` 기록 | BE가 URL 정규화를 할 수도 있음. 먼저 실측 |
| 이슈 ID 관리는 B안(알고리즘 번호 ↔ 서비스 ID 분리·승계) **MVP 필수**로 격상 | API가 `issueId`를 커서 페이지네이션 키로 사용. 재계산마다 바뀌면 무한스크롤·공유 링크가 깨짐 |

### 다음
- [ ] DE와 확인: HDFS 경로/포맷, DS가 읽는 방식, "실시간 전처리"의 범위
- [ ] BE와 확인: `content`에 HTML 잔존 여부, `article_id` 해시 시 URL 정규화 여부, 일 수집량, GPU
- [ ] PM에 질문: 단독 특종(-1이지만 살린 기사)·격리 기사를 화면에 보여줄 것인가 → STEP 4 범위 결정
- [ ] DS → BE 출력 계약 문서 (issue 테이블 컬럼·score 산식·전달 방식)
- [ ] S15P21E105-100 본문 전처리 규칙 — 실제 기사 샘플 확보가 선행
