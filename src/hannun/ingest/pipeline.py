"""입력 파일을 읽어 검증하고 Gold 에 적재한다."""

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .gold import GoldStore, Reject, to_gold_row
from .reader import RawRecord, expand_paths, iter_records
from .schema import CommonArticle

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    """한 번의 적재 결과.

    records 는 깨진 줄까지 센 수, valid 는 배치 안 중복을 포함한 수다.
    replace 가 아니면 valid - duplicates_in_batch - skipped_existing == written.
    """

    run_id: str
    files: int = 0
    records: int = 0
    valid: int = 0
    rejected: int = 0
    duplicates_in_batch: int = 0
    written: int = 0
    skipped_existing: int = 0
    replaced: int = 0
    id_mismatch: int = 0
    partitions: list[str] = field(default_factory=list)
    rejects_path: str | None = None

    def to_dict(self):
        return asdict(self)


def validate_record(raw: RawRecord):
    """(CommonArticle, None) 또는 (None, 사유) 를 돌려준다."""
    if raw.error is not None:
        return None, raw.error
    try:
        return CommonArticle.model_validate(raw.record), None
    except ValidationError as exc:
        return None, _format_validation_error(exc)


def ingest(inputs: Iterable[Path | str], store: GoldStore, on_conflict: str = "keep",
           run_id: str | None = None):
    ingested_at = datetime.now(timezone.utc)
    run_id = run_id or ingested_at.strftime("%Y%m%dT%H%M%SZ")
    stats = IngestStats(run_id=run_id)

    files = expand_paths(inputs)
    stats.files = len(files)
    if not files:
        log.warning(f"입력 파일이 없습니다: {list(inputs)}")
        return stats

    rows_by_id = {}
    rejects = []

    for raw in iter_records(files):
        stats.records += 1
        article, reason = validate_record(raw)
        if article is None:
            stats.rejected += 1
            rejects.append(Reject(raw.source_ref, reason, raw.record))
            log.debug(f"reject {raw.source_ref}: {reason}")
            continue

        stats.valid += 1
        if article.article_id in rows_by_id:
            stats.duplicates_in_batch += 1
            continue
        row = to_gold_row(article, raw.source_ref, ingested_at)
        if not row["article_id_verified"]:
            stats.id_mismatch += 1
        rows_by_id[article.article_id] = row

    upserted = store.upsert(rows_by_id.values(), on_conflict=on_conflict)
    stats.written = upserted.written
    stats.skipped_existing = upserted.skipped_existing
    stats.replaced = upserted.replaced
    stats.partitions = upserted.partitions

    rejects_path = store.write_rejects(rejects, run_id)
    stats.rejects_path = str(rejects_path) if rejects_path else None

    log.info(
        f"ingest done: files={stats.files} records={stats.records} valid={stats.valid} "
        f"rejected={stats.rejected} dup_in_batch={stats.duplicates_in_batch} "
        f"written={stats.written} skipped={stats.skipped_existing} "
        f"replaced={stats.replaced} id_mismatch={stats.id_mismatch}"
    )
    return stats


def _format_validation_error(exc):
    parts = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)
