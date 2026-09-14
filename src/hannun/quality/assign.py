"""증분 배정 — 새 대표 기사를 기존 이슈에 임시로 붙인다. 15분 루프의 배정 단계.

배정은 이슈를 만들지 못한다 — 시뮬레이션에서 새 기사 대부분은 붙을 이슈가 없었다(티켓 102).
새 이슈 탄생·분열·병합은 주기 재군집(hannun-cluster)이 처리하고, 그때 파티션이
통째로 덮이면서 여기서 만든 임시 배정도 함께 리셋된다.
"""

import collections
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from hannun.clustering.store import IssueStore
from hannun.embedding.store import EmbeddingStore

from .rules import QualityConfig, issue_centroids, nearest_issue_assignments, structured_issue_ids

log = logging.getLogger(__name__)


@dataclass
class AssignStats:
    window_start: str | None = None
    window_end: str | None = None
    existing: int = 0
    new_articles: int = 0
    assigned: int = 0
    unassigned: int = 0
    partitions: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def assign(issues: IssueStore, embeddings: EmbeddingStore, config: QualityConfig | None = None,
           start_date: str | None = None, end_date: str | None = None):
    """날짜 범위(현재 창)의 embedding 에는 있고 issue 에는 없는 기사를 이슈에 배정한다.

    정형 이슈는 배정 대상에서 뺀다 — 구제(티켓 93)와 같은 이유. 문턱 미달은 노이즈(-1)로
    적어 두고 다음 재군집에 맡긴다. 같은 범위로 재실행하면 새 기사가 없어 아무것도
    바꾸지 않는다(멱등).
    """
    config = config or QualityConfig()
    stats = AssignStats()

    table = issues.read_table(start_date, end_date)
    if table.num_rows == 0:
        log.warning(f"범위에 issue 파티션이 없습니다 — 첫 창은 재군집부터: {start_date}~{end_date}")
        return stats
    rows = table.to_pylist()
    stats.existing = len(rows)
    stats.window_start = rows[0]["window_start"]
    stats.window_end = rows[0]["window_end"]
    existing_ids = {row["article_id"] for row in rows}

    emb = embeddings.read_table(start_date, end_date,
                                columns=["article_id", "publisher_id", "published_date", "vector"])
    vector_of = dict(zip(emb.column("article_id").to_pylist(), emb.column("vector").to_pylist()))

    sizes = collections.Counter()
    publishers = collections.defaultdict(set)
    for row in rows:
        if row["issue_local"] >= 0:
            sizes[row["issue_local"]] += 1
            publishers[row["issue_local"]].add(row["publisher_id"])
    structured = structured_issue_ids({i: (sizes[i], len(publishers[i])) for i in sizes}, config)

    vectors_by_issue = collections.defaultdict(list)
    for row in rows:
        vector = vector_of.get(row["article_id"])
        if vector is not None and row["issue_local"] >= 0 and row["issue_local"] not in structured:
            vectors_by_issue[row["issue_local"]].append(vector)
    centroid_matrix, centroid_ids = issue_centroids(vectors_by_issue)

    new_rows = [
        {"article_id": a, "publisher_id": p, "published_date": d, "vector": v}
        for a, p, d, v in zip(emb.column("article_id").to_pylist(),
                              emb.column("publisher_id").to_pylist(),
                              emb.column("published_date").to_pylist(),
                              emb.column("vector").to_pylist())
        if a not in existing_ids
    ]
    stats.new_articles = len(new_rows)
    if not new_rows:
        log.info(f"assign done: existing={stats.existing} new=0 assigned=0 unassigned=0")
        return stats

    placements = nearest_issue_assignments({r["article_id"]: r["vector"] for r in new_rows},
                                           centroid_matrix, centroid_ids, config.assign_min_sim)
    stats.assigned = len(placements)
    stats.unassigned = stats.new_articles - stats.assigned
    for issue_id, _ in placements.values():
        sizes[issue_id] += 1

    clustered_at = datetime.now(timezone.utc)
    by_date = collections.defaultdict(list)
    for row in rows:
        row["issue_size"] = sizes.get(row["issue_local"], 0)
        by_date[row["published_date"]].append(row)
    for row in new_rows:
        placement = placements.get(row["article_id"])
        label = placement[0] if placement is not None else -1
        by_date[row["published_date"]].append({
            "article_id": row["article_id"],
            "publisher_id": row["publisher_id"],
            "published_date": row["published_date"],
            "issue_local": label,
            "issue_size": sizes.get(label, 0),
            "window_start": stats.window_start,
            "window_end": stats.window_end,
            "clustered_at": clustered_at,
        })
    for date_str in sorted(by_date):
        issues.write_partition(date_str, by_date[date_str])
        stats.partitions.append(date_str)

    log.info(
        f"assign done: existing={stats.existing} new={stats.new_articles} "
        f"assigned={stats.assigned} unassigned={stats.unassigned}"
    )
    return stats
