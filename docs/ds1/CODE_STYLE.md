# 코드 스타일

이 저장소의 코드가 누가 어느 날 짰는지 티 나지 않도록 정한 규칙이다.
여기 적힌 건 전부 실제 코드에 반영돼 있다. 지키지 않을 규칙은 적지 않는다.

## 저장소 구조

팀 저장소는 모노레포고 DS1·DS2 는 `data/ds/` 에서 파이썬 패키지 `hannun` 을 같이 만든다.
이 문서는 DS1 이 맡은 모듈에 적용하는 규칙이고 `ds/docs/ds1/` 에 둔다. 아래 경로는 전부
프로젝트 루트 `data/ds/` 기준이다.

```
src/hannun/<step>/     ingest, dedup, embedding, clustering, quality — 파이프라인 순서대로 패키지 하나씩
    <일>.py            순수 로직. 파일 하나가 일 하나 (schema, reader, gold …)
    pipeline.py        그 단계의 입력과 출력을 잇는 함수 하나 (ingest, dedup …)
    cli.py             명령이 필요한 단계만. 인자 파싱과 출력 외에 로직을 두지 않는다
tests/test_<모듈>.py   모듈과 1:1
scripts/               한 번 돌리는 것. 패키지를 import 해서 쓰고, 로직을 복사해 오지 않는다
samples/               커밋한다. gold/ 처럼 코드가 만드는 것은 커밋하지 않는다
docs/ds1/, docs/ds2/   개인 문서 — WORKLOG, TROUBLESHOOTING, tickets/<Jira 키>.md, 이 파일
```

`utils.py`, `helpers.py`, `common.py` 는 만들지 않는다. 갈 곳이 없어 보이는 함수는
그걸 쓰는 파일에 둔다. 두 단계가 같이 쓰게 되는 날 옮긴다.

## 파일 안 순서

```
모듈 docstring → import → 상수와 로거 → 클래스 → 함수
클래스 안: __init__ → 공개 메서드(호출되는 순서) → 내부 메서드
```

import 는 표준 라이브러리, 서드파티, 우리 패키지 순으로 빈 줄 하나씩 띄운다.

한 파일에 섹션이 다섯 개 이상일 때만 구분 주석을 넣고, 형태는 이것 하나다.

```python
# --- 읽기 ---
```

모듈 docstring 첫 줄은 이 파일이 무엇을 하는지 한 문장이다. 명령 파일은
`실행이름 — 설명` 으로 시작해 `pyproject` 의 scripts 와 바로 이어지게 한다.

```python
"""Gold 저장소 — 검증이 끝난 기사를 parquet 으로 쌓고 읽는다."""
"""hannun-ingest — 공통 기사 JSON 을 검증해 Gold 에 적재하는 명령."""
```

## 이름

식별자는 영어 snake_case, 주석과 docstring 과 문서는 한글이다.

공통 기사 JSON 의 필드명(`published_at`, `publisher_id` …)은 BE 문서 그대로 쓴다.
파생 컬럼은 `content_len`, `published_date` 처럼 원본 이름 뒤에 붙인다.
한 단계가 만든 컬럼(`duplicate_of`, `cluster_id` …)은 그 단계의 티켓 문서에 적고,
뒤 단계가 이름을 바꾸거나 지우지 않는다.

`result`, `data`, `obj`, `tmp` 같은 이름은 세 줄 안에서 쓰고 버릴 때만 허용한다.

## 함수와 클래스

함수를 먼저 쓴다. 클래스는 상태를 들고 다녀야 할 때만 만든다 — 저장소 경로를
가진 `GoldStore`, 나중에 모델을 메모리에 올려둘 임베더 같은 것. 상속은 하지 않는다.

dataclass 는 두 용도로만 쓴다. 단계의 결과(`IngestStats`)와 단계의 설정.

임계값과 파라미터는 코드에 숫자로 박지 않고 설정 dataclass 로 받는다. 골드셋으로
바꿔가며 돌려야 하는 값이고, 그 값을 왜 골랐는지는 코드가 아니라
`docs/ds1/tickets/` 에 적는다.

```python
@dataclass
class DedupConfig:
    lsh_threshold: float = 0.7
    cosine_threshold: float = 0.95
    min_len: int = 300
```

입력을 고치지 않고 새 것을 돌려준다. 원본 컬럼은 지우지 않고 옆에 파생 컬럼을
붙인다. 중복 기사를 지우지 않고 `duplicate_of` 를 남기는 것, 격리 기사를 지우지
않고 상태만 바꾸는 것과 같은 이유다 — 임계값을 잘못 잡았을 때 되돌릴 수 있어야 한다.

## 타입 힌트

공개 함수의 인자에만 붙인다. 반환 타입은 적지 않는다. `_` 로 시작하는 내부
함수와 메서드에는 붙이지 않는다. pydantic 모델과 dataclass 의 필드는 문법상 다 적는다.

```python
def expected_article_id(publisher_id: str, url: str):
    ...

def _iter_jsonl(f):
    ...
```

`Optional[X]` 대신 `X | None`. `from __future__ import annotations` 는 쓰지 않는다.

## docstring

한 줄 요약. 더 할 말이 있으면 빈 줄을 두고 문장으로 이어 쓴다. 불릿, 표, 흐름도는
넣지 않는다. 그런 건 `docs/ds1/tickets/` 의 몫이다.

인자 설명은 이름만으로 뜻이 안 드러날 때만 쓴다. 대개 단위 문제다 — UTC 인지
KST 인지, 글자 수인지 토큰 수인지, 문자열인지 date 인지.

```python
def read_table(self, start_date: str | None = None, end_date: str | None = None, ...):
    """published_date 범위(UTC, YYYY-MM-DD, 양끝 포함)의 기사를 pyarrow Table 로."""
```

## 주석

'무엇'이 아니라 '왜'를 적는다. 코드가 이미 말하는 것을 되풀이하지 않는다.

```python
# 나쁨
# 파티션별로 묶는다
by_date = defaultdict(list)

# 좋음
# 임시 파일에 쓰고 교체한다. 쓰다가 죽어도 기존 파티션은 남는다.
pq.write_table(table, tmp)
os.replace(tmp, path)
```

한 번 당한 함정은 반드시 남긴다. 이 저장소에서 가장 값진 주석이다.

```python
# pandas 3.0 부터 문자열 기본 dtype 이 str(pyarrow) 로 바뀌었다. DataFrame 을
# 거쳐 쓰면 파티션마다 컬럼 타입이 달라질 수 있어 pyarrow 스키마로 고정한다.
```

TODO 는 티켓 번호를 붙인다. 번호가 없는 TODO 는 남기지 않는다.

```python
# TODO(S15P21E105-100): 보일러플레이트 제거 후 clean_content 컬럼 추가
```

## 문자열, 로그, 예외

f-string 만 쓴다. `%` 서식과 `.format()` 은 로그에서도 쓰지 않는다.

로거는 `log = logging.getLogger(__name__)`. `print` 는 cli 의 최종 출력에만 쓴다.
INFO 는 단계가 끝날 때 `written=5 skipped=0` 식의 집계 한 줄, DEBUG 는 건별이다.

데이터 한 건의 문제는 예외를 던지지 않고 리젝트로 흘려 사유를 남긴다. 예외는
설정이 틀렸거나 코드가 잘못됐을 때만 던진다. 예외 메시지와 리젝트 사유는 영어로
쓴다 — pydantic 이 만드는 영어 메시지와 같은 파일에 섞여 들어가기 때문이다.

## 시간과 재현

datetime 은 항상 타임존이 있는 UTC 다. 날짜 문자열은 UTC 기준 `YYYY-MM-DD`.
KST 로 바꾸는 건 화면에 보여줄 때 BE 가 한다.

난수를 쓰는 함수는 seed 를 인자로 받고 기본값을 고정한다. 결과를 저장할 때는
어떤 모델, 어떤 파라미터로 만들었는지 같은 파일에 기록한다. 같은 입력이면 같은
파일이 나와야 하므로 저장 전에 정렬 순서를 고정한다.

## 테스트

pytest. 이름은 `test_<무엇>_<기대>`. 한 테스트는 한 가지 사실만 확인한다.

샘플 입력은 손으로 만들지 않고 `scripts/make_sample_data.py` 같은 스크립트로
생성해 `samples/` 에 둔다. 실제 기사 데이터는 커밋하지 않는다.

## 하지 않는 것

생성 도구가 남기는 습관이다. 보이면 지운다.

- 모든 함수에 붙은 반환 타입, `-> None`
- docstring 안의 불릿, 표, 흐름도, `Args:` 나열
- 코드를 되풀이하는 주석, 컬럼마다 붙은 설명 주석
- `# ------ 이름` 처럼 길이가 제각각인 구분선
- `__all__`, 이모지, 구현이 하나뿐인 추상 클래스와 팩토리
- 있지도 않은 경우를 대비한 방어 코드

## 검사

```bash
pytest tests/ -v
flake8 . --select=E9,F63,F7,F82,E501 --max-line-length=110 --exclude=.venv,build
```

black 같은 자동 정렬 도구는 쓰지 않는다. 한글 주석 정렬이 깨진다.
