# DS1 → DS2 · 이슈 군집 테이블 계약

DS2(관점 대조·키워드)가 소비하는 DS1 산출물의 위치·스키마·주의. 저장은 전부
parquet 이고 `hannun` 패키지의 스토어 클래스로 읽는다 (경로 문자열 조립 금지 —
파티션 규칙이 스토어 안에 있다).

## 1. 어디에 있나

| 테이블 | 위치 | 단위 | 읽기 |
|---|---|---|---|
| 기사→이슈 배정 (최종) | `<root>/issue_quality/` | 기사, 발행일(UTC) 파티션 | `QualityStore(root).read(start, end)` |
| 이슈 카드 (대표·화제성) | `<root>/issue_summary/` | 이슈, 창 파티션 | `SummaryStore(root).read(window_start)` |
| 기사→서비스 이슈 ID | `<root>/issue_registry/` | 기사, 창 파티션 | `RegistryStore(root).read_window(window_start)` |

root: EC2 운영 `/home/ubuntu/gold`, 로컬 개발은 각자 gold 루트.
갱신: 매일 04:30 KST 일일 배치가 최근 48h 창을 재계산한다.

## 2. 핵심 컬럼

**issue_quality** (기사 단위 — DS2 의 주 입력)

| 컬럼 | 뜻 |
|---|---|
| `article_id` | gold·clean 과 조인 키 |
| `issue_local` | 이슈 번호. **창 안에서만 유효한 임시 번호**, -1 은 노이즈 |
| `structured` | 정형(시황·인사 등 템플릿) 이슈 소속 — **관점 분석 대상에서 제외 권장** |
| `rescued` | 노이즈였다가 구제 배정된 기사 |

**issue_summary** (이슈 단위)

| 컬럼 | 뜻 |
|---|---|
| `issue_local` / `issue_id` | 임시 번호 / 승계되는 서비스 ID |
| `category` | **이슈 카테고리** — 구성 기사 카테고리(OTHER·빈값 제외) 다수결, 없으면 대표 기사 값. 값은 팀 확정 어휘 8개 `정치·경제·사회·국제·연예·스포츠·문화·IT/과학` 또는 `OTHER`(2026-09-15 확정; DE 원문 `산업`→`경제`, `IT과학`→`IT/과학` 은 DS1 이 이슈 단위에서 맞춘다, 기사 단위 `gold.category` 는 원문 그대로). 대표 기사 하나의 값보다 커버리지가 높으니(실측 기사 87%→이슈 99%) 이슈 카테고리는 이 컬럼을 권장 (34) |
| `representative` | 대표 기사 article_id (이슈 중심 최근접) |
| `issue_size` · `publishers` · `hot_score` | 규모 · 언론사 수 · 화제성(피드 정렬 키) |
| `first/last_published_at` | 이슈 시간 범위 (UTC) |

**issue_registry**: `article_id → issue_id`. 날짜를 넘나드는 추적은 `issue_local`
이 아니라 **`issue_id`** 로 — 재군집마다 issue_local 은 리셋되고 issue_id 가 승계된다.

## 3. 읽기 예시

```python
from hannun.quality import QualityStore
from hannun.feed import SummaryStore
from hannun.ingest import GoldStore

root = "/home/ubuntu/gold"
q = QualityStore(root).read("2026-09-07", "2026-09-08")
members = q[(q.issue_local >= 0) & (~q.structured)]          # 분석 대상
gold = GoldStore(root).read("2026-09-07", "2026-09-08",
                            columns=["article_id", "title", "content"])
per_issue = members.merge(gold, on="article_id").groupby("issue_local")
```

## 4. 주의

1. 파티션 날짜는 **UTC 발행일** — KST 하루는 UTC 이틀에 걸친다
2. `issue_local` 을 저장·전달 키로 쓰지 말 것 (창마다 리셋). 지속 키는 `issue_id`
3. 본문은 `CleanStore`(`content_clean`) 권장 — 실수집 UI 잔재가 정제돼 있다
4. 노이즈(-1)·정형(structured) 처리 정책은 소비 쪽 선택이지만, 관점 분석은 둘 다
   제외를 권장 (근거: 티켓 93·104 실측)

## 5. 역방향 (DS2 → 서비스/DS1) — 확정

- 기사 단위 산출(관점 라벨): `article_id` 키 — 어떤 창과도 무관하게 유효
- 이슈 단위 산출(키워드·관점 분포): `issue_id` 키 — 창을 넘어 유지

## 6. 연동 합의 (2026-09-09, DS1·DS2)

1. **테이블→JSON 변환은 DS2 몫** — DS2 가 parquet 테이블을 읽어 enrichment 입력
   형태로 변환한다. 기존 코드 호환을 위해 `issue_cluster_id = str(issue_id)`.
2. **`keywords` 는 당분간 optional(빈 배열 허용)** — DS1 이슈 키워드(티켓 31) 생성
   전이므로. 생성이 붙으면 연결한다.
3. 역방향 키는 §5 대로 확정.
