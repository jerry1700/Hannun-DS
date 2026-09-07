"""issue·embedding 테이블을 읽어 STEP 4 판정을 내리고 issue_quality 테이블에 쓴다."""

import collections
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from hannun.clustering.store import IssueStore
from hannun.embedding.store import EmbeddingStore

from .rules import QualityConfig, issue_centroids, rescue_assignments, structured_issue_ids
from .store import QualityStore

log = logging.getLogger(__name__)


@dataclass
class QualityStats:
    window_start: str | None = None
    window_end: str | None = None
    rows: int = 0
    issues: int = 0
    structured_issues: int = 0
    structured_articles: int = 0
    noise_before: int = 0
    rescued: int = 0
    noise_after: int = 0
    partitions: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def qualify(issues: IssueStore, embeddings: EmbeddingStore, store: QualityStore,
            config: QualityConfig | None = None,
            start_date: str | None = None, end_date: str | None = None):
    """날짜 범위(UTC, 양끝 포함)의 군집 결과에 정형 판별·노이즈 구제를 적용한다.

    창 의미론은 STEP 3 과 동일 — 같은 범위로 돌려야 한 창의 판정이 된다.
    정형 이슈는 구제 대상에서 뺀다: 템플릿 덩어리가 벡터 공간에서 빽빽해
    노이즈를 빨아들이기 쉽고, 거기 붙는 순간 기사가 정형으로 오염된다.
    """
    config = config or QualityConfig()
    stats = QualityStats()

    table = issues.read_table(start_date, end_date)
    if table.num_rows == 0:
        log.warning(f"범위에 issue 파티션이 없습니다: {start_date}~{end_date}")
        return stats
    rows = table.to_pylist()
    stats.rows = len(rows)
    stats.window_start = rows[0]["window_start"]
    stats.window_end = rows[0]["window_end"]

    emb = embeddings.read_table(start_date, end_date, columns=["article_id", "vector"])
    vector_of = dict(zip(emb.column("article_id").to_pylist(), emb.column("vector").to_pylist()))

    # 이슈별 규모·언론사 수 → 정형 판별
    sizes = collections.Counter()
    publishers = collections.defaultdict(set)
    for row in rows:
        if row["issue_local"] >= 0:
            sizes[row["issue_local"]] += 1
            publishers[row["issue_local"]].add(row["publisher_id"])
    meta = {i: (sizes[i], len(publishers[i])) for i in sizes}
    stats.issues = len(meta)
    structured = structured_issue_ids(meta, config)
    stats.structured_issues = len(structured)

    # 노이즈 구제 — 정형이 아닌 이슈의 중심에만 붙인다
    vectors_by_issue = collections.defaultdict(list)
    noise_vectors = {}
    for row in rows:
        vector = vector_of.get(row["article_id"])
        if vector is None:
            continue
        if row["issue_local"] >= 0 and row["issue_local"] not in structured:
            vectors_by_issue[row["issue_local"]].append(vector)
        elif row["issue_local"] < 0:
            noise_vectors[row["article_id"]] = vector
    stats.noise_before = sum(1 for row in rows if row["issue_local"] < 0)
    centroid_matrix, centroid_ids = issue_centroids(vectors_by_issue)
    rescues = rescue_assignments(noise_vectors, centroid_matrix, centroid_ids, config)
    stats.rescued = len(rescues)
    stats.noise_after = stats.noise_before - stats.rescued

    final_sizes = collections.Counter(sizes)
    for issue_id, _ in rescues.values():
        final_sizes[issue_id] += 1

    qualified_at = datetime.now(timezone.utc)
    by_date = collections.defaultdict(list)
    for row in rows:
        issue_id = row["issue_local"]
        rescue = rescues.get(row["article_id"])
        if rescue is not None:
            issue_id = rescue[0]
        is_structured = issue_id in structured
        if is_structured:
            stats.structured_articles += 1
        by_date[row["published_date"]].append({
            "article_id": row["article_id"],
            "publisher_id": row["publisher_id"],
            "published_date": row["published_date"],
            "issue_local": issue_id,
            "issue_size": final_sizes.get(issue_id, 0),
            "structured": is_structured,
            "rescued": rescue is not None,
            "rescue_sim": rescue[1] if rescue is not None else None,
            "window_start": stats.window_start,
            "window_end": stats.window_end,
            "qualified_at": qualified_at,
        })
    for date_str in sorted(by_date):
        store.write_partition(date_str, by_date[date_str])
        stats.partitions.append(date_str)

    log.info(
        f"qualify done: rows={stats.rows} issues={stats.issues} "
        f"structured={stats.structured_issues}({stats.structured_articles}건) "
        f"noise {stats.noise_before}→{stats.noise_after} (rescued {stats.rescued})"
    )
    return stats
