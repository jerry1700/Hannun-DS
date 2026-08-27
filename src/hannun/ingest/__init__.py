"""STEP 0 — 공통 기사 JSON 을 검증해 Gold 에 적재한다."""

from .schema import CommonArticle, SourceType, expected_article_id, article_id_matches
from .gold import GoldStore, GOLD_SCHEMA
from .pipeline import ingest, IngestStats
