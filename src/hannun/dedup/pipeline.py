"""Gold 와 clean 을 읽어 중복을 판정하고 dedup 테이블에 쓴다 — STEP 1 의 네 관문을 잇는다."""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from hannun.ingest.gold import GoldStore
from hannun.preprocess.store import CleanStore

from .candidates import CandidateConfig, find_candidate_pairs
from .exact import find_exact_duplicates
from .groups import GroupConfig, build_groups
from .store import DedupStore
from .verify import VerifyConfig, verify_pairs

log = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    # 정제본이 이보다 짧으면 접지 않는다. 본문이 저작권 꼬리뿐인 속보·포토·영상 기사는
    # 실제 내용이 제목에 있는데 본문이 같다는 이유로 서로 접혀 사라진다 — 5,105건 실측에서
    # 나온 오탐 전부가 이 유형이었다. 100 은 크롤러 발행 기준·정제 short 기준과 같은 값이다.
    min_fold_len: int = 100
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    groups: GroupConfig = field(default_factory=GroupConfig)


@dataclass
class DedupStats:
    window_start: str | None = None
    window_end: str | None = None
    rows: int = 0
    excluded_short: int = 0
    exact_groups: int = 0
    exact_duplicates: int = 0
    candidate_pairs: int = 0
    skipped_short: int = 0
    confirmed_pairs: int = 0
    rejected_pairs: int = 0
    near_groups: int = 0
    split_oversized: int = 0
    duplicates: int = 0
    singles: int = 0
    partitions: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def dedup(gold: GoldStore, clean: CleanStore, store: DedupStore, config: DedupConfig | None = None,
          start_date: str | None = None, end_date: str | None = None):
    """날짜 범위(UTC, 양끝 포함)의 기사 전체를 한 번에 놓고 중복을 판정한다.

    범위가 곧 창이다 — 쪼개서 두 번 돌리면 경계를 넘는 쌍(자정 직전·직후 기사)을 놓치므로
    항상 창 하나를 통째로 돌린다. 완전 중복은 원문으로, 근사 중복은 정제본으로 보고
    정제본이 없는 기사는 원문으로 대신한다. 정제본이 min_fold_len 미만인 기사는
    판정 자체에서 제외해 단독으로 남긴다.
    """
    config = config or DedupConfig()
    stats = DedupStats()

    dates = [
        d for d in gold.partition_dates()
        if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
    ]
    if not dates:
        log.warning(f"범위에 Gold 파티션이 없습니다: {start_date}~{end_date}")
        return stats
    stats.window_start, stats.window_end = dates[0], dates[-1]

    rows = _load_rows(gold, clean, dates)
    stats.rows = len(rows)

    # 본문이 짧은 기사는 "같은 기사"라 볼 근거 자체가 없으므로 접지도, 대표가 되지도 않는다
    foldable = [row for row in rows if row["fold_len"] >= config.min_fold_len]
    stats.excluded_short = len(rows) - len(foldable)

    exact = find_exact_duplicates(foldable)
    stats.exact_groups = exact.groups
    stats.exact_duplicates = len(exact.duplicate_of)

    survivors = [row for row in foldable if row["article_id"] not in exact.duplicate_of]
    candidates = find_candidate_pairs(
        [{"article_id": row["article_id"], "text": row["text"]} for row in survivors],
        config.candidates,
    )
    stats.candidate_pairs = len(candidates.pairs)
    stats.skipped_short = candidates.skipped_short

    texts = {row["article_id"]: row["text"] for row in survivors}
    verified = verify_pairs(texts, candidates.pairs, config.verify)
    stats.confirmed_pairs = len(verified.confirmed)
    stats.rejected_pairs = verified.rejected

    # 대표 선정: 최초 발행 → 본문 긴 것 → article_id. 음수 길이로 "클수록 앞"을 만든다.
    order = {row["article_id"]: (row["published_at"], -len(row["content"]), row["article_id"])
             for row in survivors}
    near = build_groups(verified.confirmed, order, config.groups)
    stats.near_groups = near.groups
    stats.split_oversized = near.split_oversized

    records = _merge(rows, exact, near, verified.method)
    stats.duplicates = sum(1 for r in records.values() if r["duplicate_of"] is not None)
    stats.singles = sum(1 for r in records.values() if r["duplicate_count"] == 1)

    deduped_at = datetime.now(timezone.utc)
    by_date = {}
    for row in rows:
        record = records[row["article_id"]]
        by_date.setdefault(row["published_date"], []).append({
            "article_id": row["article_id"],
            "publisher_id": row["publisher_id"],
            "published_date": row["published_date"],
            "duplicate_of": record["duplicate_of"],
            "duplicate_count": record["duplicate_count"],
            "method": record["method"],
            "window_start": stats.window_start,
            "window_end": stats.window_end,
            "deduped_at": deduped_at,
        })
    for date_str in dates:
        store.write_partition(date_str, by_date.get(date_str, []))
        stats.partitions.append(date_str)

    log.info(
        f"dedup done: rows={stats.rows} exact={stats.exact_duplicates} candidates={stats.candidate_pairs} "
        f"confirmed={stats.confirmed_pairs} groups={stats.exact_groups + stats.near_groups} "
        f"duplicates={stats.duplicates}"
    )
    return stats


def _load_rows(gold, clean, dates):
    columns = ["article_id", "publisher_id", "published_date", "content", "published_at"]
    table = gold.read_table(dates[0], dates[-1], columns=columns)
    clean_table = clean.read_table(dates[0], dates[-1], columns=["article_id", "content_clean"])
    clean_of = dict(zip(clean_table.column("article_id").to_pylist(),
                        clean_table.column("content_clean").to_pylist()))
    rows = []
    for article_id, publisher_id, published_date, content, published_at in zip(
            *(table.column(c).to_pylist()
              for c in ("article_id", "publisher_id", "published_date", "content", "published_at"))):
        clean_text = clean_of.get(article_id)
        rows.append({
            "article_id": article_id,
            "publisher_id": publisher_id,
            "published_date": published_date,
            "content": content,
            "published_at": published_at,
            "text": clean_text or content,
            # 접기 자격은 정제본 길이로 판단한다 — 정제하니 비어 버린 기사는 0 으로 제외되고,
            # 정제본이 아예 없으면(전처리 미수행) 원문 길이로 대신한다
            "fold_len": len(clean_text) if clean_text is not None else len(content),
        })
    return rows


def _merge(rows, exact, near, pair_method):
    """완전 중복과 근사 중복을 합쳐 기사별 최종 판정을 만든다.

    exact 로 접힌 기사의 대표가 다시 근사 중복으로 접혔으면 최종 대표까지 따라간다.
    duplicate_count 는 최종 대표 밑에 모인 기사 수(자기 포함)다.
    """
    final_of, method = {}, {}
    for article_id, representative in near.duplicate_of.items():
        final_of[article_id] = representative
        pair = (article_id, representative) if article_id < representative else (representative, article_id)
        method[article_id] = pair_method.get(pair, "chained")
    for article_id, representative in exact.duplicate_of.items():
        final_of[article_id] = near.duplicate_of.get(representative, representative)
        method[article_id] = "sha256"

    counts = {}
    for row in rows:
        representative = final_of.get(row["article_id"], row["article_id"])
        counts[representative] = counts.get(representative, 0) + 1

    records = {}
    for row in rows:
        article_id = row["article_id"]
        folded = article_id in final_of
        records[article_id] = {
            "duplicate_of": final_of.get(article_id),
            "duplicate_count": 0 if folded else counts.get(article_id, 1),
            "method": method.get(article_id),
        }
    return records
