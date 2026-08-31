"""공통 기사 JSON(de/schemas/article_v1.json, schema_version 1.0)의 검증 모델."""

import hashlib
from datetime import timezone
from enum import Enum
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import AliasChoices, AwareDatetime, BaseModel, ConfigDict, Field, field_validator

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
# 처럼 쿼리가 곧 기사 번호인 언론사가 있다. 목록과 "남은 파라미터 정렬"은 DE 와 합의된
# 확정 규칙(2026-08-31, de/schemas/article_v1.json) — DE 실측 220만 건에서 식별자 손실 0%.
TRACKING_PARAMS = (
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "igshid", "page", "date", "ref", "from",
)


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
    # DE 가 article_v1 필드명을 데이터셋 8필드식으로 개편했다(2026-08-31). 속성명은 Gold
    # 컬럼이자 하류(DS2) 계약이라 그대로 두고, 입력 경계에서 두 이름을 모두 받아 흡수한다.
    publisher_name: str = Field(validation_alias=AliasChoices("company", "publisher_name"))
    source_type: SourceType
    url: str = Field(validation_alias=AliasChoices("link", "url"))
    title: str
    content: str = Field(validation_alias=AliasChoices("article", "content"))
    author: str | None = Field(None, validation_alias=AliasChoices("reporter", "author"))
    category: str
    category_str: str | None = None
    thumbnail_url: str | None = None
    language: str
    published_at: AwareDatetime = Field(validation_alias=AliasChoices("published", "published_at"))

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
    """article_id 해시에 쓰는 URL. 추적 파라미터와 프래그먼트를 떼고 남은 쿼리는 정렬한다.

    정렬하는 이유: 같은 기사라도 크롤 경로에 따라 파라미터 순서가 다를 수 있고,
    순서가 다르면 해시가 달라진다. 규칙 전체가 DE 와 합의된 확정본이다.
    """
    parts = urlsplit(url.strip())
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in TRACKING_PARAMS]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(sorted(kept)), ""))


def expected_article_id(publisher_id: str, url: str):
    """SHA-256(publisher_id + "|" + 정규화 URL) 로 article_id 를 만든다."""
    digest = hashlib.sha256(f"{publisher_id}|{normalize_url(url)}".encode("utf-8")).hexdigest()
    return ARTICLE_ID_PREFIX + digest


def article_id_matches(article: CommonArticle):
    """받은 article_id 가 우리 해시 규칙과 맞는지.

    안 맞아도 리젝트하지 않고 Gold 에 표시만 한다. 정규화 규칙은 DE 와 합의됐으므로
    False 는 규칙 불일치가 아니라 개별 데이터의 문제(id 재계산 누락 등)를 가리키는 신호다.
    """
    given = article.article_id.removeprefix(ARTICLE_ID_PREFIX).strip().lower()
    expected = expected_article_id(article.publisher_id, article.url)
    return given == expected.removeprefix(ARTICLE_ID_PREFIX)
