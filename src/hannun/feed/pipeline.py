"""이슈 피드 요약 — 창의 이슈마다 대표 기사와 요약 지표를 뽑아 issue_summary 에 쓴다.

대표 기사 = 이슈 중심(정규화 평균 벡터)에 가장 가까운 기사. 최초 발행 기사는
프리뷰·곁가지가 이슈의 얼굴이 되는 사례가 많아(1/2 창 상위 12개 눈검사) 탈락 —
발행 시각은 first_published_at 으로 따로 보존한다.
"""

import collections
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from hannun.clustering.registry import RegistryStore
from hannun.embedding.store import EmbeddingStore
from hannun.ingest.gold import GoldStore
from hannun.quality.store import QualityStore

from .store import SummaryStore

log = logging.getLogger(__name__)


@dataclass
class SummaryStats:
    window_start: str | None = None
    window_end: str | None = None
    issues: int = 0
    structured_issues: int = 0
    with_issue_id: int = 0
    partitions: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def summarize(quality: QualityStore, embeddings: EmbeddingStore, registry: RegistryStore,
              gold: GoldStore, store: SummaryStore,
              start_date: str | None = None, end_date: str | None = None):
    """날짜 범위(현재 창)의 issue_quality 를 이슈 단위로 요약한다.

    운영 순서상 quality(93)·succeed(97) 다음에 돈다. 레지스트리가 아직 없으면
    issue_id 는 -1 로 두고 경고만 남긴다 — 요약 자체는 유효하다.
    """
    stats = SummaryStats()

    q = quality.read_table(start_date, end_date)
    if q.num_rows == 0:
        log.warning(f"범위에 issue_quality 파티션이 없습니다: {start_date}~{end_date}")
        return stats
    rows = [r for r in q.to_pylist() if r["issue_local"] >= 0]
    if not rows:
        log.warning("이슈가 없습니다 — 전부 노이즈")
        return stats
    stats.window_start = rows[0]["window_start"]
    stats.window_end = rows[0]["window_end"]

    emb = embeddings.read_table(start_date, end_date, columns=["article_id", "vector"])
    vector_of = dict(zip(emb.column("article_id").to_pylist(), emb.column("vector").to_pylist()))
    g = gold.read_table(start_date, end_date, columns=["article_id", "published_at"])
    published_of = dict(zip(g.column("article_id").to_pylist(), g.column("published_at").to_pylist()))

    id_of = {}
    reg = registry.read_window(stats.window_start)
    if reg.num_rows:
        id_of = dict(zip(reg.column("issue_local").to_pylist(), reg.column("issue_id").to_pylist()))
    else:
        log.warning(f"레지스트리에 창 {stats.window_start} 이 없습니다 — issue_id=-1 로 둡니다")

    members = collections.defaultdict(list)
    for row in rows:
        members[row["issue_local"]].append(row)

    summarized_at = datetime.now(timezone.utc)
    out = []
    for label, rows_of in members.items():
        ids = [r["article_id"] for r in rows_of]
        times = {a: published_of.get(a) for a in ids}
        known_times = {a: t for a, t in times.items() if t is not None}

        vectors = {a: vector_of[a] for a in ids if a in vector_of}
        if vectors:
            matrix = np.array(list(vectors.values()), dtype="float32")
            centroid = matrix.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-9)
            sims = matrix @ centroid
            best = max(sims)
            # 동률이면 최초 발행 — 정렬 키를 (유사도 차, 발행 시각) 으로
            candidates = [a for a, s in zip(vectors, sims) if best - s < 1e-6]
            representative = min(candidates, key=lambda a: (known_times.get(a) or summarized_at))
        else:
            representative = min(ids, key=lambda a: (known_times.get(a) or summarized_at))

        issue_id = id_of.get(label, -1)
        if issue_id >= 0:
            stats.with_issue_id += 1
        structured = bool(rows_of[0]["structured"])
        if structured:
            stats.structured_issues += 1
        out.append({
            "window_start": stats.window_start,
            "window_end": stats.window_end,
            "issue_local": label,
            "issue_id": issue_id,
            "issue_size": len(ids),
            "publishers": len({r["publisher_id"] for r in rows_of}),
            "structured": structured,
            "representative": representative,
            "first_published_at": min(known_times.values()) if known_times else None,
            "last_published_at": max(known_times.values()) if known_times else None,
            "summarized_at": summarized_at,
        })
    stats.issues = len(out)
    store.write_window(stats.window_start, out)
    stats.partitions.append(stats.window_start)

    log.info(
        f"summarize done: window={stats.window_start} issues={stats.issues} "
        f"structured={stats.structured_issues} with_issue_id={stats.with_issue_id}"
    )
    return stats
