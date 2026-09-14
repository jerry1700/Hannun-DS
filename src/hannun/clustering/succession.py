"""이슈 ID 승계 — 재군집으로 리셋된 임시 번호를 서비스 이슈 ID 로 잇는다.

클러스터 번호는 재군집마다 바뀌지만 기사는 불변이다. 새 군집의 구성원이 직전 창에서
어느 이슈에 있었는지 득표로 세어, 과반 겹침이면 그 이슈의 서비스 ID 를 승계한다.
시뮬레이션 실측(티켓 97): 창당 승계 42~50%(이슈 수명 분포와 부합), 문턱 0.3↔0.5
차이 3%p 로 둔감, 분열 ~7%·병합 ~4% 는 규칙으로 처리.

자기-승계: 같은 창의 직전 실행이 있으면 그 배정이 직전 창보다 우선한다 — 창 하나를
여러 번 재계산(15분 재군집, 데이터가 늘어난 재실행)해도 ID 가 흔들리지 않는 근거.
운영 실측(09-09): 이게 없던 시절 같은 창 재실행이 ID 를 전부 재발급했다.
"""

import collections
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .registry import RegistryStore
from .store import IssueStore

log = logging.getLogger(__name__)


@dataclass
class SuccessionConfig:
    # 겹침 기사(직전 창에도 있던 구성원) 중 최다 득표 이슈가 과반이고 2건 이상일 때 승계
    min_shared: int = 2
    min_overlap_frac: float = 0.5


@dataclass
class SuccessionStats:
    window_start: str | None = None
    window_end: str | None = None
    prev_window: str | None = None
    self_window: bool = False   # 같은 창의 직전 실행을 승계원으로 썼는가
    issues: int = 0
    inherited: int = 0
    created: int = 0
    splits: int = 0
    retired: int = 0   # 승계원에 있었는데 아무 군집도 이어받지 않은 이슈 — 출렁임 감시용
    articles: int = 0

    def to_dict(self):
        return asdict(self)


def succeed(issues: IssueStore, registry: RegistryStore, config: SuccessionConfig | None = None,
            start_date: str | None = None, end_date: str | None = None):
    """현재 창(재군집 직후)의 issue_local 에 서비스 issue_id 를 배정해 레지스트리에 쓴다.

    승계원은 둘이다: 직전 창(현재 창 시작일보다 앞선 것 중 가장 최근)과, 있다면
    **같은 창의 직전 실행**(자기-승계 — 이쪽이 우선). 그래서 같은 창을 데이터가
    늘어난 채 재실행해도 살아 있는 이슈의 ID 는 유지되고, 순수 재실행은 같은
    결과를 낸다. status 는 "직전 창 대비 계보"의 의미를 지키기 위해 자기-승계
    시 이전 값을 그대로 물려받는다(이 창에서 태어난 이슈는 재실행 뒤에도 new).
    """
    config = config or SuccessionConfig()
    stats = SuccessionStats()

    table = issues.read_table(start_date, end_date)
    if table.num_rows == 0:
        log.warning(f"범위에 issue 파티션이 없습니다: {start_date}~{end_date}")
        return stats
    rows = table.to_pylist()
    stats.window_start = rows[0]["window_start"]
    stats.window_end = rows[0]["window_end"]

    members = collections.defaultdict(list)
    for row in rows:
        if row["issue_local"] >= 0:
            members[row["issue_local"]].append(row)
    stats.issues = len(members)

    starts = registry.window_starts()
    prev_starts = [s for s in starts if s < stats.window_start]
    prev_id_of = {}
    if prev_starts:
        stats.prev_window = prev_starts[-1]
        prev_table = registry.read_window(stats.prev_window)
        prev_id_of = dict(zip(prev_table.column("article_id").to_pylist(),
                              prev_table.column("issue_id").to_pylist()))

    # 자기-승계 — 같은 창의 직전 실행이 있으면 그 배정이 직전 창보다 우선한다
    # (같은 article_id 가 양쪽에 있으면 현재 창의 배정이 최신 진실)
    self_status_of = {}
    if stats.window_start in starts:
        stats.self_window = True
        self_table = registry.read_window(stats.window_start)
        prev_id_of.update(zip(self_table.column("article_id").to_pylist(),
                              self_table.column("issue_id").to_pylist()))
        self_status_of = dict(zip(self_table.column("issue_id").to_pylist(),
                                  self_table.column("status").to_pylist()))

    # 새 군집별 직전 이슈 득표 → 같은 직전 이슈를 여럿이 주장하면 최다 득표가 승계(분열)
    claims = collections.defaultdict(list)
    for label, rows_of in members.items():
        votes = collections.Counter(
            prev_id_of[r["article_id"]] for r in rows_of if r["article_id"] in prev_id_of)
        if not votes:
            continue
        prev_issue_id, shared = votes.most_common(1)[0]
        overlap = sum(votes.values())
        if shared >= config.min_shared and shared / overlap >= config.min_overlap_frac:
            claims[prev_issue_id].append((shared, label))

    assigned = {}
    for prev_issue_id, entries in claims.items():
        entries.sort(reverse=True)
        assigned[entries[0][1]] = prev_issue_id
        stats.splits += len(entries) - 1
    stats.retired = len(set(prev_id_of.values()) - set(assigned.values()))

    # 채번은 전 창 통틀어 최대 ID 다음부터 — 자기-승계가 재실행 멱등을 책임지므로,
    # 옛 "현재 창 제외" 방식(재실행에서 죽은 이슈의 ID 가 다른 군집에 재사용될
    # 위험이 있던)은 버린다
    next_id = registry.max_issue_id() + 1
    succeeded_at = datetime.now(timezone.utc)
    out = []
    for label in sorted(members):
        if label in assigned:
            issue_id = assigned[label]
            # 자기-승계로 이어받은 이슈는 계보(status)도 그대로 — 이 창에서
            # 태어난 이슈는 재실행 뒤에도 new 로 남아야 BE 통계가 안 흔들린다
            status = self_status_of.get(issue_id, "inherited")
            stats.inherited += 1
        else:
            issue_id, status = next_id, "new"
            next_id += 1
            stats.created += 1
        for row in members[label]:
            out.append({
                "window_start": stats.window_start,
                "window_end": stats.window_end,
                "article_id": row["article_id"],
                "published_date": row["published_date"],
                "issue_local": label,
                "issue_id": issue_id,
                "status": status,
                "succeeded_at": succeeded_at,
            })
    stats.articles = len(out)
    registry.write_window(stats.window_start, out)

    log.info(
        f"succeed done: window={stats.window_start} issues={stats.issues} "
        f"inherited={stats.inherited} created={stats.created} splits={stats.splits} "
        f"retired={stats.retired} (prev={stats.prev_window}, self={stats.self_window})"
    )
    return stats
