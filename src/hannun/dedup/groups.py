"""확정 쌍을 그룹으로 묶고 대표를 정한다 — Union-Find 와 그룹 상한.

A~B, B~C 가 각각 중복이면 A·B·C 를 한 그룹으로 본다(Union-Find). 다만 이 연쇄가 길어지면
서로 전혀 다른 기사가 한 그룹에 들어올 수 있어서, 그룹이 상한을 넘으면 대표와 직접
확정된 쌍만 남기고 나머지는 자기들끼리 다시 묶는다(설계의 Star 재계산).

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
    result = DuplicateGroups()

    neighbors = collections.defaultdict(set)
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)

    for component in _components(neighbors):
        for group in _split_if_oversized(component, neighbors, order, config.max_group_size, result):
            if len(group) < 2:
                continue
            representative = min(group, key=lambda article_id: order[article_id])
            result.groups += 1
            result.duplicate_count[representative] = len(group)
            for article_id in group:
                if article_id != representative:
                    result.duplicate_of[article_id] = representative
    return result


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


def _split_if_oversized(component, neighbors, order, cap, result):
    if len(component) <= cap:
        yield component
        return
    # 대표와 직접 확정된 쌍만 남긴다. 연쇄(A~B~C)로만 이어진 기사는 대표와 직접 비교된 적이
    # 없으므로 떼어내고, 남은 것들끼리 다시 묶는다.
    result.split_oversized += 1
    representative = min(component, key=lambda article_id: order[article_id])
    star = [representative] + sorted(neighbors[representative] & set(component))
    yield star[:cap]

    rest = set(component) - set(star[:cap])
    if len(rest) < 2:
        return
    rest_neighbors = {node: neighbors[node] & rest for node in rest}
    for sub in _components(rest_neighbors):
        yield from _split_if_oversized(sub, rest_neighbors, order, cap, result)
