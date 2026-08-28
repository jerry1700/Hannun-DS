"""Gold 의 원문을 읽어 정제하고 clean 테이블에 쓴다."""

import collections
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from hannun.ingest.gold import GoldStore

from .clean import PreprocessConfig, clean_content
from .store import CleanStore

log = logging.getLogger(__name__)


@dataclass
class PreprocessStats:
    rules_version: str
    partitions: list[str] = field(default_factory=list)
    rows: int = 0
    ok: int = 0
    short: int = 0
    empty: int = 0
    removed_chars: int = 0
    rule_hits: dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def preprocess(gold: GoldStore, clean: CleanStore, config: PreprocessConfig | None = None,
               start_date: str | None = None, end_date: str | None = None):
    """Gold 파티션을 날짜 범위(UTC, 양끝 포함)만큼 읽어 clean 파티션을 다시 쓴다."""
    config = config or PreprocessConfig()
    cleaned_at = datetime.now(timezone.utc)
    stats = PreprocessStats(rules_version=config.rules_version)
    hits = collections.Counter()

    dates = [
        d for d in gold.partition_dates()
        if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
    ]
    for date_str in dates:
        table = gold.read_table(date_str, date_str, columns=["article_id", "publisher_id", "content"])
        rows = []
        for article_id, publisher_id, content in zip(*(table.column(c).to_pylist()
                                                        for c in ("article_id", "publisher_id", "content"))):
            result = clean_content(content, publisher_id, config)
            hits.update(result.rules_applied)
            stats.removed_chars += result.removed_chars
            setattr(stats, result.status, getattr(stats, result.status) + 1)
            rows.append({
                "article_id": article_id,
                "publisher_id": publisher_id,
                "published_date": date_str,
                "content_clean": result.content_clean,
                "content_clean_len": len(result.content_clean),
                "clean_status": result.status,
                "rules_applied": result.rules_applied,
                "removed_chars": result.removed_chars,
                "rules_version": config.rules_version,
                "cleaned_at": cleaned_at,
            })
        stats.rows += clean.write_partition(date_str, rows)
        stats.partitions.append(date_str)
        log.debug(f"clean {date_str} → {len(rows)} rows")

    stats.rule_hits = dict(hits.most_common())
    log.info(
        f"preprocess done: partitions={len(stats.partitions)} rows={stats.rows} ok={stats.ok} "
        f"short={stats.short} empty={stats.empty} removed_chars={stats.removed_chars}"
    )
    return stats
