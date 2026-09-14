from test_dedup_pipeline import ARTICLES, build_stores

from hannun.dedup import SignatureStore, dedup
from hannun.dedup.candidates import (CandidateConfig, build_entries, minhash_from,
                                     minhash_scheme, pairs_from_entries)
from hannun.dedup.signatures import text_sha1

BASE = "정부는 오늘 새로운 청년 지원 정책을 발표했다. 청년 주거 안정과 취업 지원을 핵심으로 한다. " * 6
ROWS = [
    {"article_id": "a", "text": BASE},
    {"article_id": "b", "text": BASE.replace("발표했다", "발표하였다")},
    {"article_id": "c", "text": "프로야구 한화가 기아를 상대로 9회말 역전승을 거두며 3연승을 달렸다. " * 6},
]


def test_minhash_from_hashvalues_reproduces_candidate_pairs():
    config = CandidateConfig()
    entries, fresh, _ = build_entries(ROWS, config)
    restored = [(a, minhash_from(fresh[a][1], config.num_perm), fresh[a][2]) for a, _, _ in entries]

    assert pairs_from_entries(entries, config).pairs == pairs_from_entries(restored, config).pairs
    for (_, computed, _), (_, rebuilt, _) in zip(entries, restored):
        assert (computed.hashvalues == rebuilt.hashvalues).all()


def test_build_entries_hits_cache_and_invalidates_on_text_change():
    config = CandidateConfig()
    _, fresh, _ = build_entries(ROWS, config)
    assert set(fresh) == {"a", "b", "c"}

    entries, fresh_again, _ = build_entries(ROWS, config, cached=fresh)
    assert fresh_again == {} and len(entries) == 3          # 전부 캐시 적중

    edited = [dict(r, text=r["text"] + " 추가 문장.") if r["article_id"] == "b" else r for r in ROWS]
    _, recomputed, _ = build_entries(edited, config, cached=fresh)
    assert set(recomputed) == {"b"}                          # 정제본이 바뀐 것만 다시 계산
    assert recomputed["b"][0] == text_sha1(edited[1]["text"])


def test_signature_store_roundtrip_filters_other_config(tmp_path):
    store = SignatureStore(tmp_path / "gold")
    config = CandidateConfig()
    _, fresh, _ = build_entries(ROWS, config)
    scheme = minhash_scheme()
    store.merge_partition("2026-08-20", fresh, config.shingle_size, config.num_perm, scheme)

    cache = store.read_cache("2026-08-20", "2026-08-20",
                             config.shingle_size, config.num_perm, scheme)
    assert set(cache) == {"a", "b", "c"}
    assert (cache["a"][1] == fresh["a"][1]).all() and cache["a"][2] == fresh["a"][2]
    # 설정·해시 방식이 다르면 그 행은 없는 것으로 — 변경 시 자연 재계산
    assert store.read_cache("2026-08-20", "2026-08-20", 5, config.num_perm, scheme) == {}
    assert store.read_cache("2026-08-20", "2026-08-20", config.shingle_size, config.num_perm,
                            "other") == {}
    # 같은 기사를 다시 합치면 덧붙지 않고 바뀐다
    rows_after = store.merge_partition("2026-08-20", {"a": fresh["a"]},
                                       config.shingle_size, config.num_perm, scheme)
    assert rows_after == 3


def test_pipeline_second_run_hits_cache_and_gives_same_table(tmp_path):
    gold, clean, store = build_stores(tmp_path)
    signatures = SignatureStore(tmp_path / "gold")

    first = dedup(gold, clean, store, signatures=signatures)
    table_first = store.read().sort_values("article_id").reset_index(drop=True)
    second = dedup(gold, clean, store, signatures=signatures)
    table_second = store.read().sort_values("article_id").reset_index(drop=True)

    assert first.signatures_computed > 0 and first.signatures_cached == 0
    assert second.signatures_computed == 0 and second.signatures_cached == first.signatures_computed
    assert first.duplicates == second.duplicates == 3        # exact·near·containment 접힘 그대로
    for column in ("duplicate_of", "duplicate_count", "method"):
        assert table_first[column].tolist() == table_second[column].tolist()
    assert len(ARTICLES) == len(table_first)
