# DS1 → DS2 전달 데이터 (제안안 v1)

> **이 문서는 2026-08-27 의 제안안이다.** 확정된 계약은 [../contracts/ds1-to-ds2-issue-tables.md](../contracts/ds1-to-ds2-issue-tables.md) 를 본다.

> 작성: DS1 (2026-08-27) / 대상: DS2
> 상태: **제안안** — 필드명은 BE 공통 기사 JSON·Gold 초안과 맞췄고, 전달 방식(Kafka/HDFS/직접 호출)은 DE와 협의 중.
> 실데이터 샘플: `schemas/ds1_issue_output_v1.example.json` (반도체 이슈 3개, 기사 66건 — 더미 입력으로 바로 사용 가능)

## 0. 먼저 맞춰야 할 역할 경계 2가지

DS2 문서를 읽고 발견한 오해입니다. 스키마보다 이게 먼저입니다.

### ① 이슈명(Target)은 DS1이 아니라 DS2가 만듭니다

DS2 문서에 "DS1에서 생성된 이슈명을 Target으로 사용"이라고 돼 있는데, **DS1은 문장형 이슈명을 만들지 않습니다.**
DS1이 주는 것은 `keywords`(c-TF-IDF 상위 5개)와 `representative_title`(대표 기사 제목)까지입니다.
문장형 `event_name`은 DS2 중분류(LexRank + LLM)의 산출물이고, DS2 문서의 다른 곳("중분류에서 생성한 event_name을 Target으로")이 맞습니다.
→ **Target = DS2 중분류가 만든 `event_name`.** DS1 키워드와 대표 제목은 그 생성의 입력.

### ② 기사 목록은 "대표 기사만" 넘깁니다

STEP 1이 복제 기사(통신사 전재 등)를 `duplicate_of`로 접은 뒤, 클러스터에는 **대표 기사만** 들어갑니다.
- LexRank: 복붙 원문이 반복 입력되는 문제가 구조적으로 사라짐 (DS2 문서가 원한 효과)
- 소분류: 복제본은 대표 기사의 stance를 그대로 상속하면 됨 → `duplicate_count`만 전달
- 필요하면 복제본 목록은 `duplicate_of == 대표 article_id`로 별도 조회 가능

## 1. 전달 단위: 이슈 1건 = 이슈 정보 + 소속 기사 배열

DS2의 중분류는 이슈 단위, 소분류는 기사 단위라서 **이슈 객체 안에 기사 배열을 넣는 형태(DS2 문서의 방법 A)**가 처리하기 편합니다.
Gold 저장은 이슈/기사 테이블로 나뉘지만(방법 B), 그건 DE 저장 단계의 일이고 DS1→DS2 전달은 A로 제안합니다.

```json
{
  "schema_version": "1.0",
  "issue_cluster_id": "issue_20260824_0004",
  "keywords": ["최태원", "비공개", "회동", "메가투자", "SK"],
  "representative_article_id": "sha256:2d6d...",
  "representative_title": "李, 최태원과 비공개 만찬…반도체 메가투자 논의",
  "size": 22,
  "hot_score": 24,
  "press_count": 19,
  "status": "active",
  "first_seen": "2026-08-20T09:12:00Z",
  "last_updated": "2026-08-24T01:40:00Z",
  "rebuilt_at": "2026-08-24T02:30:00Z",
  "articles": [
    {
      "article_id": "sha256:2d6d...",
      "title": "李, 최태원과 비공개 만찬…반도체 메가투자 논의",
      "content": "이재명 대통령이 ...",
      "publisher_name": "매일경제",
      "published_at": "2026-08-20T09:12:00Z",
      "is_representative": true,
      "duplicate_count": 1,
      "cluster_probability": 0.93,
      "outlier_score": 0.12,
      "quality_status": "active"
    }
  ]
}
```

## 2. 필드 명세

### 이슈 정보

| 필드 | 타입 | 설명 | DS2 용도 |
|---|---|---|---|
| `issue_cluster_id` | String | **재군집 후에도 유지되는 이슈 ID** (실행마다 바뀌는 HDBSCAN 번호가 아님). Gold 초안의 필드명 그대로 사용 | 결과 저장 키 |
| `keywords` | array<String> | c-TF-IDF 상위 5개. **형태소 분석기 도입 후 제공 — 현재 샘플은 빈 배열** | 중분류 입력, event_name 생성 힌트 |
| `representative_article_id` | String | 대표 기사 (최초 발행 → 본문 길이 → 언론사 순) | — |
| `representative_title` | String | 대표 기사 제목. **이슈명 초안으로 쓸 수 있음** | event_name 생성 힌트 |
| `size` | Integer | 대표 기사 수 (= `articles` 길이) | — |
| `hot_score` | Integer | size + 복제본 수 — 화제성 정렬 기준 | 처리 우선순위 |
| `press_count` | Integer | 언론사 수 | 논쟁성 판단 보조 |
| `status` | String | `active` / `closed` / `quarantined` — **`active`만 처리** | 필터 |
| `first_seen`, `last_updated` | RFC 3339 UTC | 이슈 수명 | — |
| `rebuilt_at` | RFC 3339 UTC | 어느 재군집 실행의 결과인지 | 재처리 판단 |

### 기사 정보 (`articles[]`)

| 필드 | 타입 | 출처 | 설명 |
|---|---|---|---|
| `article_id` | String | BE | `sha256:...` (BE 공통 JSON 그대로) |
| `title`, `content`, `publisher_name`, `published_at` | — | BE | 공통 기사 JSON 그대로. `content`는 **STEP 0 전처리 후 본문**(제보 안내·기자 서명 등 제거) |
| `is_representative` | Boolean | DS1 | 이슈 카드 대표 기사 |
| `duplicate_count` | Integer | DS1 | 이 기사로 접힌 복제본 수. 복제본은 이 기사의 stance 상속 |
| `cluster_probability` | Float 0~1 | DS1 | 클러스터 소속 확신도. **낮으면(<0.5) 경계 기사 — stance 판정 시 주의** |
| `outlier_score` | Float 0~1 | DS1 | 클러스터 안에서의 이상치 점수. **높으면(>0.8) 이슈와 어긋나는 기사일 수 있음** |
| `quality_status` | String | DS1 | `active` / `solo` / `pending` / `quarantined` — 전달되는 건 `active`만 |

### 옵션 — 필요하면 추가 가능

| 필드 | 설명 |
|---|---|
| `embedding` (array<float>, 768) | DS1이 이미 계산한 Ko-SRoBERTa(seq512) 기사 벡터. **DS2 Baseline이 같은 모델을 쓰므로 재계산 불필요** — 원하면 넘김. 단 이슈 JSON에 넣으면 크기가 커져서(기사당 3KB) 별도 저장소 참조가 나을 수 있음 |
| `publisher_id` | BE 코드. 언론사별 집계 시 |
| `category` | BE 카테고리. 섹션별 처리 시 |

## 3. DS2가 확인해줄 것

1. 이슈 단위 전달(방법 A)로 괜찮은지
2. `keywords`가 당장 비어 있어도 더미 진행에 문제없는지 (형태소 분석기 도입 후 채움)
3. `embedding` 전달을 원하는지 — 원하면 형식(배열 포함 vs 참조) 결정
4. 소분류 결과의 기사 키는 `article_id`로, 이슈 키는 `issue_cluster_id`로 맞추면 Gold 저장에서 바로 조인 가능

## 4. 참고 — Baseline 관련 경험 공유 (강제 아님)

DS1에서 news30 골드셋(30건)과 미래대응기금 클러스터(64건)를 손으로 스탠스 분류해본 결과가 `docs/실험_news30_파이프라인_vs_수동라벨.md`, `docs/실험_스탠스_수동분류_미래기금.md`에 있습니다.
거기서 관측된 것 두 가지가 Baseline 설계에 참고가 될 수 있습니다:
- Ko-SRoBERTa 벡터는 **주제**로 뭉치고 **입장**으로는 거의 갈라지지 않았습니다 (같은 이슈의 긍정·부정 기사가 벡터 공간에서 거의 같은 자리). 기준 문장과의 코사인 비교 방식은 이 한계를 그대로 물려받을 가능성이 있어, 초반에 골드셋으로 빨리 확인해보시길 권합니다.
- 스탠스 신호는 대부분 **인용문과 [사설]·[칼럼] 표기**에 있었습니다 — Segment-level 분석의 근거가 될 수 있습니다.
