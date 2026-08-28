# DS — 이슈 군집화와 관점 대조

한눈 뉴스 파이프라인의 데이터 사이언스 파트. DS1 과 DS2 가 파이썬 패키지 `hannun` 하나를
같이 만든다. 이 디렉토리(`data/ds/`)가 파이썬 프로젝트 루트다.

```
DE 공통 기사 JSON
      │
      ▼  DS1
STEP 0  ingest      검증 → Gold(parquet)                           99 ✔
        preprocess  Gold 원문 정제 → clean(content_clean)            100 ✔
STEP 1  dedup       SHA-256 → MinHash/LSH → TF-IDF → Union-Find     10
STEP 2  embedding   ko-sroberta (제목+리드)                           11
STEP 3  clustering  UMAP + HDBSCAN + 이슈 ID 승계                     89 / 90 / 97
STEP 4  quality     노이즈 구제 · 어그로 판정 · 화제성 점수             93 / 91
      │
      ▼  DS2
        stance      이슈 안 긍정·부정·중립 그룹
        viewpoint   그룹 안 세부 견해 재군집, 정렬                      30 / 34
        keywords    이슈·소분류 키워드, 논쟁성 신호                     31 / 95
      │
      ▼
   BE 서빙 DB (계약: ../docs/contracts/)
```

## 모듈 소유

| 모듈 | 소유 | 개인 문서 |
|---|---|---|
| `hannun.ingest` `dedup` `embedding` `clustering` `quality` | DS1 제다빈 | `docs/ds1/` |
| `hannun.stance` `viewpoint` `keywords` | DS2 | `docs/ds2/` |

자기 모듈은 자유롭게 고치고, 상대 모듈이나 공용 파일(`pyproject.toml`, 이 README, `tests/conftest.py`)을
건드리는 MR 은 상대를 리뷰어로 넣는다. 한 단계가 만든 컬럼은 뒤 단계가 이름을 바꾸거나 지우지 않는다.

## 디렉토리

```
src/hannun/<step>/   단계별 패키지. 순수 로직 파일 + pipeline.py + (필요하면) cli.py
tests/               pytest. test_<모듈>.py 로 1:1
samples/             커밋되는 샘플 입력 (scripts/ 로 생성)
scripts/             보조 스크립트
gold/ 등 생성물      .gitignore. 커밋하지 않는다
docs/ds1/, docs/ds2/ 개인 작업 기록, 트러블슈팅, 티켓 문서, 코드 스타일
```

## 시작하기

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\make_sample_data.py     # 샘플 입력 생성
.\.venv\Scripts\python.exe -m pytest tests/ -v

# STEP 0: 공통 기사 JSON → Gold → clean
.\.venv\Scripts\hannun-ingest.exe --input samples\articles_sample.jsonl --gold-root gold -v
.\.venv\Scripts\hannun-preprocess.exe --gold-root gold -v
```

Gold 와 정제 본문 읽기 (`article_id` 로 조인):

```python
from hannun.ingest import GoldStore
from hannun.preprocess import CleanStore
gold = GoldStore("gold").read(start_date="2026-08-20", end_date="2026-08-21")
clean = CleanStore("gold").read(start_date="2026-08-20", end_date="2026-08-21")
df = gold.merge(clean[["article_id", "content_clean", "clean_status"]], on="article_id")
```

뒤 단계는 원문이 필요한 곳(STEP 1 의 SHA-256)만 `content` 를 쓰고 나머지는 `content_clean` 을 쓴다.
규칙과 근거는 [docs/ds1/tickets/S15P21E105-100.md](docs/ds1/tickets/S15P21E105-100.md).

## 검사

MR 올리기 전에 둘 다 통과해야 한다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m flake8 . --select=E9,F63,F7,F82,E501 --max-line-length=110 --exclude=.venv,build
```

## 문서

- 파트 공통 규칙(브랜치·커밋·MR): [../docs/GIT_CONVENTION.md](../docs/GIT_CONVENTION.md)
- 다른 파트(DE, BE)와 주고받는 데이터 형태: [../docs/contracts/](../docs/contracts/)
- 개인 기록은 `docs/<자기 폴더>/` 에. DS1 의 코드 규칙은 [docs/ds1/CODE_STYLE.md](docs/ds1/CODE_STYLE.md)
