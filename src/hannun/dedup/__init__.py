"""STEP 1 — 완전 중복과 근사 중복을 판정해 dedup 테이블에 쓴다. 지우지 않고 duplicate_of 를 남긴다."""

from .exact import ExactDuplicates, find_exact_duplicates
from .pipeline import DedupConfig, DedupStats, dedup
from .signatures import SignatureStore
from .store import DedupStore
