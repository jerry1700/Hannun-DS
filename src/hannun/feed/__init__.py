"""이슈 피드 — 창의 이슈마다 대표 기사·카테고리·화제성을 요약해 issue_summary 에 쓴다."""

from .pipeline import SummaryStats, majority_category, summarize
from .scoring import FeedConfig, hot_score
from .store import SUMMARY_SCHEMA, SummaryStore
