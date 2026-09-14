"""embedding 테이블의 벡터를 읽어 이슈로 묶고 issue 테이블에 쓴다."""

import collections
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from hannun.embedding.store import EmbeddingStore

from .clusterer import ClusterConfig, cluster_vectors, reduce_vectors
from .store import IssueStore

log = logging.getLogger(__name__)


@dataclass
class ClusterStats:
    window_start: str | None = None
    window_end: str | None = None
    rows: int = 0
    issues: int = 0
    noise: int = 0
    largest_issue: int = 0
    partitions: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def cluster(embeddings: EmbeddingStore, store: IssueStore, config: ClusterConfig | None = None,
            start_date: str | None = None, end_date: str | None = None, reduce_fn=None):
    """날짜 범위(UTC, 양끝 포함)의 대표 기사 벡터 전체를 한 창으로 놓고 이슈를 묶는다.

    dedup 과 같은 창 의미론 — 쪼개서 두 번 돌리면 경계를 넘는 이슈를 놓친다.
    입력은 embedding 테이블(대표 기사만)이라 접힌 기사는 애초에 없다.
    reduce_fn 은 테스트 주입용 — umap 없이 파이프라인을 검증한다.
    """
    config = config or ClusterConfig()
    stats = ClusterStats()

    dates = [
        d for d in embeddings.partition_dates()
        if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
    ]
    if not dates:
        log.warning(f"범위에 embedding 파티션이 없습니다: {start_date}~{end_date}")
        return stats
    stats.window_start, stats.window_end = dates[0], dates[-1]

    table = embeddings.read_table(dates[0], dates[-1])
    stats.rows = table.num_rows

    # 모델이 섞인 벡터 공간에서는 거리가 의미를 잃는다 — 조용히 이상한 이슈를 내느니 멈춘다
    models = set(table.column("model").to_pylist())
    if len(models) > 1:
        raise ValueError(f"embedding models are mixed within the window: {sorted(models)}")

    matrix = np.array(table.column("vector").to_pylist(), dtype="float32")
    if reduce_fn is not None:
        points = reduce_fn(matrix)
    elif config.umap_dims:
        points = reduce_vectors(matrix, config)
    else:
        points = matrix
    labels = cluster_vectors(points, config)

    sizes = collections.Counter(int(label) for label in labels if label >= 0)
    stats.issues = len(sizes)
    stats.noise = int((labels < 0).sum())
    stats.largest_issue = max(sizes.values(), default=0)

    clustered_at = datetime.now(timezone.utc)
    by_date = collections.defaultdict(list)
    for article_id, publisher_id, published_date, label in zip(
            table.column("article_id").to_pylist(), table.column("publisher_id").to_pylist(),
            table.column("published_date").to_pylist(), labels):
        label = int(label)
        by_date[published_date].append({
            "article_id": article_id,
            "publisher_id": publisher_id,
            "published_date": published_date,
            "issue_local": label,
            "issue_size": sizes.get(label, 0),
            "window_start": stats.window_start,
            "window_end": stats.window_end,
            "clustered_at": clustered_at,
        })
    for date_str in dates:
        store.write_partition(date_str, by_date.get(date_str, []))
        stats.partitions.append(date_str)

    log.info(
        f"cluster done: rows={stats.rows} issues={stats.issues} noise={stats.noise} "
        f"largest={stats.largest_issue}"
    )
    return stats
