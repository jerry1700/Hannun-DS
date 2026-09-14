"""확정 쌍을 별(star) 그룹으로 묶고 대표를 정한다.

접히는 기사는 반드시 대표와 직접 확정된 쌍이어야 한다. 처음에는 연쇄(A~B~C)도
한 그룹으로 접었지만, 실측 표본 전수 검수(티켓 10 §4.5)에서 연쇄로만 이어진 접힘의
정탐률이 직접 검증된 접힘의 절반에도 못 미쳐 연쇄 전파를 없앴다 — 대표의
직접 이웃만 접고, 나머지는 자기들끼리 다시 별을 만든다.

지우는 것은 없다. 중복 기사에는 duplicate_of(대표 article_id)를, 대표에는
duplicate_count(자기 포함 묶인 수)를 남긴다 — STEP 4 가 화제성 신호로 쓴다.
"""

import collections
from dataclasses import dataclass, field


@dataclass
class GroupConfig:
    max_group_size: int = 30


@dataclass
class DuplicateGroups:
    duplicate_of: dict[str, str] = field(default_factory=dict)
    duplicate_count: dict[str, int] = field(default_factory=dict)
    groups: int = 0
    split_oversized: int = 0


def build_groups(pairs: set[tuple[str, str]], order: dict[str, tuple], config: GroupConfig | None = None):
    """pairs 는 확정 중복 쌍, order 는 article_id → 대표 선정 정렬 키(작을수록 대표).

    정렬 키는 호출자가 만든다 — 파이프라인은 (published_at, -content_len, article_id) 를 쓴다.
    """
    config = config or GroupConfig()
    grouped = DuplicateGroups()

    neighbors = collections.defaultdict(set)
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)

    for component in _components(neighbors):
        for group in _star_groups(component, neighbors, order, config.max_group_size, grouped):
            if len(group) < 2:
                continue
            representative = min(group, key=lambda article_id: order[article_id])
            grouped.groups += 1
            grouped.duplicate_count[representative] = len(group)
            for article_id in group:
                if article_id != representative:
                    grouped.duplicate_of[article_id] = representative
    return grouped


def _components(neighbors):
    seen = set()
    for start in sorted(neighbors):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for other in neighbors[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        yield component


def _star_groups(component, neighbors, order, cap, grouped):
    """대표 + 대표와 직접 확정된 쌍만 한 그룹으로. 나머지는 자기들끼리 다시 별을 만든다.

    연쇄로만 이어진 기사는 대표와 직접 비교된 적이 없다 — 검수에서 그런 접힘은 셋 중
    둘이 다른 기사였다(티켓 10 §4.5). 상한은 직접 이웃이 폭주할 때의 안전판으로만 남는다.
    """
    representative = min(component, key=lambda article_id: order[article_id])
    star = [representative] + sorted(neighbors[representative] & set(component))
    if len(star) > cap:
        grouped.split_oversized += 1
    yield star[:cap]

    rest = set(component) - set(star[:cap])
    if len(rest) < 2:
        return
    rest_neighbors = {node: neighbors[node] & rest for node in rest}
    for sub in _components(rest_neighbors):
        yield from _star_groups(sub, rest_neighbors, order, cap, grouped)
