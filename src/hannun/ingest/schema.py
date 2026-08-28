"""공통 기사 JSON(BE 문서 §3, schema_version 1.0)의 검증 모델."""

import hashlib
from datetime import timezone
from enum import Enum
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import AwareDatetime, BaseModel, ConfigDict, field_validator

REQUIRED_TEXT_FIELDS = (
    "schema_version",
    "article_id",
    "publisher_id",
    "publisher_name",
    "url",
    "title",
    "content",
    "category",
    "language",
)
OPTIONAL_TEXT_FIELDS = ("author", "category_str", "thumbnail_url")
ARTICLE_ID_PREFIX = "sha256:"
# 해시 전에 URL 에서 떼는 파라미터. 같은 기사가 유입 경로마다 다른 id 를 받지 않게 한다.
# 쿼리를 통째로 지우지 않는 이유: 디지털타임(contents.html?article_no=), 국민일보(view.asp?arcid=)
# 처럼 쿼리가 곧 기사 번호인 언론사가 있다. de/schemas/article_v1.json 은 쿼리 전체 제거인데,
# 그대로면 그 언론사 기사가 전부 같은 id 가 되어 DE 와 조정 중이다(docs/contracts).
TRACKING_PARAMS = ("utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid")


class SourceType(str, Enum):
    DATASET = "DATASET"
    RSS = "RSS"
    HTML = "HTML"


class CommonArticle(BaseModel):
    """공통 기사 JSON 1건. 필드 타입이 곧 검증 규칙이다.

    title 과 content 는 어떤 변형도 하지 않는다. 전처리는 티켓 100 의 일이고,
    여기서 strip 이라도 해두면 원문이 무엇이었는지 나중에 알 수 없다.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: str
    article_id: str
    publisher_id: str
    publisher_name: str
    source_type: SourceType
    url: str
    title: str
    content: str
    author: str | None = None
    category: str
    category_str: str | None = None
    thumbnail_url: str | None = None
    language: str
    published_at: AwareDatetime

    @field_validator(*REQUIRED_TEXT_FIELDS, mode="before")
    @classmethod
    def _required_text(cls, v):
        # 본문 추출에 실패한 기사가 빈 문자열로 넘어올 수 있다. 빈 본문은 뒤
        # 단계 어디에서도 쓸 수 없으니 여기서 사유를 남기고 걸러낸다.
        if not isinstance(v, str):
            raise ValueError("must be a string")
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator(*OPTIONAL_TEXT_FIELDS, mode="before")
    @classmethod
    def _optional_text(cls, v):
        # BE 규칙은 "빈 문자열 대신 null" 이지만 믿지 않고 여기서 맞춘다.
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("must be a string or null")
        return v if v.strip() else None

    @field_validator("published_at", mode="after")
    @classmethod
    def _to_utc(cls, v):
        # 타임존 없는 값은 AwareDatetime 이 이미 걸렀다. 발행 시각은 STEP 1·3 의
        # 시간 윈도우 기준이라, KST 인지 UTC 인지 모호한 값을 받으면 9시간
        # 어긋난 채 이슈가 묶인다.
        return v.astimezone(timezone.utc)


def normalize_url(url: str):
    """article_id 해시에 쓰는 URL. 추적 파라미터와 프래그먼트를 떼고 나머지는 그대로 둔다."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in TRACKING_PARAMS]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def expected_article_id(publisher_id: str, url: str):
    """SHA-256(publisher_id + "|" + 정규화 URL) 로 article_id 를 만든다."""
    digest = hashlib.sha256(f"{publisher_id}|{normalize_url(url)}".encode("utf-8")).hexdigest()
    return ARTICLE_ID_PREFIX + digest


def article_id_matches(article: CommonArticle):
    """받은 article_id 가 우리 해시 규칙과 맞는지.

    안 맞아도 리젝트하지 않고 Gold 에 표시만 한다. DE 의 정규화 규칙(쿼리 전체 제거)과
    우리 규칙(추적 파라미터만 제거)이 다른 동안은 쿼리가 있는 URL 에서 False 가 나오는데,
    그 비율이 곧 조정이 필요한 기사의 양이다.
    """
    given = article.article_id.removeprefix(ARTICLE_ID_PREFIX).strip().lower()
    expected = expected_article_id(article.publisher_id, article.url)
    return given == expected.removeprefix(ARTICLE_ID_PREFIX)
