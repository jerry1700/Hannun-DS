from hannun.ingest import GoldStore, ingest
from hannun.preprocess import CleanStore, PreprocessConfig, preprocess

CFG = PreprocessConfig(min_clean_len=10)


def build(tmp_path, sample_path):
    gold = GoldStore(tmp_path / "gold")
    ingest([sample_path], gold)
    clean = CleanStore(tmp_path / "gold")
    return gold, clean


def test_clean_table_has_one_row_per_gold_article(tmp_path, sample_path):
    gold, clean = build(tmp_path, sample_path)
    stats = preprocess(gold, clean, CFG)

    assert stats.rows == len(gold.read()) == 5
    assert stats.partitions == gold.partition_dates() == clean.partition_dates()
    assert set(clean.read().article_id) == set(gold.read().article_id)


def test_boilerplate_removed_and_rules_recorded(tmp_path, sample_path):
    gold, clean = build(tmp_path, sample_path)
    preprocess(gold, clean, CFG)
    df = clean.read().set_index("publisher_id")

    yonhap = df.loc["yonhap"]
    assert "무단" not in yonhap.content_clean and "사진=" not in yonhap.content_clean
    assert {"copyright", "photo_credit"} <= set(yonhap.rules_applied)

    kbs = df.loc["kbs"]
    assert "[앵커]" not in kbs.content_clean and "KBS 뉴스" not in kbs.content_clean
    assert kbs.content_clean.startswith("정부가 청년 정책을 발표했습니다.")
    assert kbs.removed_chars > 0


def test_untouched_article_keeps_content_and_empty_rule_list(tmp_path, sample_path):
    gold, clean = build(tmp_path, sample_path)
    preprocess(gold, clean, CFG)
    df = clean.read().set_index("publisher_id")
    original = gold.read().set_index("publisher_id").loc["hani", "content"]

    assert df.loc["hani", "content_clean"] == original
    assert list(df.loc["hani", "rules_applied"]) == []
    assert df.loc["hani", "clean_status"] == "ok"


def test_rerun_is_deterministic(tmp_path, sample_path):
    gold, clean = build(tmp_path, sample_path)
    preprocess(gold, clean, CFG)
    first = clean.read(columns=["article_id", "content_clean", "rules_applied"])
    preprocess(gold, clean, CFG)
    second = clean.read(columns=["article_id", "content_clean", "rules_applied"])

    assert first.content_clean.tolist() == second.content_clean.tolist()
    assert first.article_id.tolist() == second.article_id.tolist()


def test_date_range_limits_partitions(tmp_path, sample_path):
    gold, clean = build(tmp_path, sample_path)
    stats = preprocess(gold, clean, CFG, start_date="2026-01-01")

    assert stats.partitions == ["2026-08-20"]
    assert clean.partition_dates() == ["2026-08-20"]


def test_stats_count_statuses_and_rule_hits(tmp_path, sample_path):
    gold, clean = build(tmp_path, sample_path)
    stats = preprocess(gold, clean, CFG)

    assert stats.ok + stats.short + stats.empty == stats.rows
    assert stats.rule_hits["copyright"] >= 1
    assert stats.rules_version == CFG.rules_version
    assert stats.to_dict()["rows"] == 5
