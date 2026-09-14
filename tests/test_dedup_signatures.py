import pyarrow as pa

from hannun.dedup import DedupStore, SignatureStore, dedup
from hannun.dedup.candidates import (CandidateConfig, build_entries, minhash_from, pairs_from_entries,
                                     text_sha1)
from hannun.ingest import GoldStore, ingest
from hannun.ingest.gold import write_parquet_atomic
from hannun.preprocess import CleanStore, PreprocessConfig, preprocess

BASE = "정부는 오늘 새로운 청년 지원 정책을 발표했다. 청년 주거 안정과 취업 지원을 핵심으로 한다. " * 6
ROWS = [
    {"article_id": "a", "text": BASE},
    {"article_id": "b", "text": BASE.replace("발표했다", "발표하였다")},
    {"article_id": "c", "text": "프로야구 한화가 기아를 상대로 9회말 역전승을 거두며 3연승을 달렸다. " * 6},
]
DAY = "2026-08-20"


def test_minhash_from_hashvalues_reproduces_candidate_pairs():
    config = CandidateConfig()
    entries, fresh, _ = build_entries(ROWS, config)
    restored = [(a, minhash_from(fresh[a][1], config.num_perm, config.seed), fresh[a][2])
                for a, _, _ in entries]

    assert pairs_from_entries(entries, config).pairs == pairs_from_entries(restored, config).pairs
    for (_, computed, _), (_, rebuilt, _) in zip(entries, restored):
        assert (computed.hashvalues == rebuilt.hashvalues).all()


def test_build_entries_hits_cache_when_text_is_unchanged():
    config = CandidateConfig()
    _, fresh, _ = build_entries(ROWS, config)
    assert set(fresh) == {"a", "b", "c"}

    entries, fresh_again, _ = build_entries(ROWS, config, cached=fresh)
    assert fresh_again == {} and len(entries) == 3


def test_build_entries_recomputes_only_changed_text():
    config = CandidateConfig()
    _, fresh, _ = build_entries(ROWS, config)

    edited = [dict(r, text=r["text"] + " 추가 문장.") if r["article_id"] == "b" else r for r in ROWS]
    _, recomputed, _ = build_entries(edited, config, cached=fresh)
    assert set(recomputed) == {"b"}
    assert recomputed["b"][0] == text_sha1(edited[1]["text"])


def test_signature_store_roundtrips_hashvalues_and_shingle_count(tmp_path):
    store = SignatureStore(tmp_path / "gold")
    config = CandidateConfig()
    _, fresh, _ = build_entries(ROWS, config)
    store.merge_partition(DAY, fresh, config)

    cache = store.read_cache(DAY, DAY, config)
    assert set(cache) == {"a", "b", "c"}
    assert (cache["a"][1] == fresh["a"][1]).all() and cache["a"][2] == fresh["a"][2]


def test_signature_store_ignores_rows_with_other_config(tmp_path):
    store = SignatureStore(tmp_path / "gold")
    config = CandidateConfig()
    _, fresh, _ = build_entries(ROWS, config)
    store.merge_partition(DAY, fresh, config)

    # 설정이 다르면 그 행은 없는 것으로 — 변경 시 자연 재계산
    assert store.read_cache(DAY, DAY, CandidateConfig(shingle_size=5)) == {}
    assert store.read_cache(DAY, DAY, CandidateConfig(seed=7)) == {}


def test_signature_store_merge_replaces_same_article(tmp_path):
    store = SignatureStore(tmp_path / "gold")
    config = CandidateConfig()
    _, fresh, _ = build_entries(ROWS, config)
    store.merge_partition(DAY, fresh, config)

    assert store.merge_partition(DAY, {"a": fresh["a"]}, config) == 3


def test_signature_store_ignores_partition_with_old_columns(tmp_path):
    store = SignatureStore(tmp_path / "gold")
    config = CandidateConfig()
    old = pa.Table.from_pylist([{"article_id": "a", "text_sha1": "x"}])
    write_parquet_atomic(old, store.partition_path(DAY))

    assert store.read_cache(DAY, DAY, config) == {}
    _, fresh, _ = build_entries(ROWS, config)
    assert store.merge_partition(DAY, fresh, config) == 3   # 옛 행은 버리고 새로 쓴다


def test_pipeline_second_run_hits_cache_and_gives_same_table(tmp_path, article_jsonl):
    root = tmp_path / "gold"
    gold, clean, store = GoldStore(root), CleanStore(root), DedupStore(root)
    ingest([article_jsonl([(r["article_id"], "pub", f"{DAY}T05:00:00Z", r["text"]) for r in ROWS])], gold)
    preprocess(gold, clean, PreprocessConfig(min_clean_len=10))
    signatures = SignatureStore(root)

    first = dedup(gold, clean, store, signatures=signatures)
    table_first = store.read().sort_values("article_id").reset_index(drop=True)
    second = dedup(gold, clean, store, signatures=signatures)
    table_second = store.read().sort_values("article_id").reset_index(drop=True)

    assert first.signatures_computed == 3 and first.signatures_cached == 0
    assert second.signatures_computed == 0 and second.signatures_cached == 3
    for column in ("duplicate_of", "duplicate_count", "method"):
        assert table_first[column].tolist() == table_second[column].tolist()
