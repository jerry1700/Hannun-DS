"""STEP 2 — 대표 기사를 문장 벡터로 바꿔 embedding 테이블에 쌓는다. 한 기사는 한 번만 인코딩한다."""

from .encoder import EncoderConfig, build_input, encode_texts, load_model
from .pipeline import EmbedStats, embed
from .store import EMBEDDING_SCHEMA, EmbeddingStore
