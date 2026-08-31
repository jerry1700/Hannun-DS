from hannun.dedup.groups import GroupConfig, build_groups


def order_of(*ids):
    # 나열 순서가 곧 대표 우선순위 — 테스트에서는 발행 시각 대신 순번을 쓴다
    return {article_id: (i,) for i, article_id in enumerate(ids)}


def test_chained_pairs_form_one_group_with_earliest_representative():
    result = build_groups({("a", "b"), ("b", "c")}, order_of("a", "b", "c"))

    assert result.groups == 1
    assert result.duplicate_of == {"b": "a", "c": "a"}
    assert result.duplicate_count == {"a": 3}


def test_separate_pairs_form_separate_groups():
    result = build_groups({("a", "b"), ("x", "y")}, order_of("a", "b", "x", "y"))

    assert result.groups == 2
    assert result.duplicate_of == {"b": "a", "y": "x"}
    assert result.duplicate_count == {"a": 2, "x": 2}


def test_no_pairs_no_groups():
    result = build_groups(set(), {})
    assert result.groups == 0 and result.duplicate_of == {}


def test_oversized_chain_is_split_at_cap():
    # a0 - a1 - a2 - … 일렬 연쇄 40개. 상한 5 면 대표(a0)와 직접 이어진 a1 만 남고
    # 나머지는 자기들끼리 다시 묶여 여러 그룹이 된다. 한 그룹도 상한을 넘으면 안 된다.
    ids = [f"a{i:02d}" for i in range(40)]
    chain = {(ids[i], ids[i + 1]) for i in range(39)}
    result = build_groups(chain, order_of(*ids), GroupConfig(max_group_size=5))

    assert result.split_oversized >= 1
    assert all(count <= 5 for count in result.duplicate_count.values())
    assert result.duplicate_of["a01"] == "a00"
    # 모든 기사는 정확히 한 그룹에만 속한다
    members = list(result.duplicate_of) + list(result.duplicate_count)
    assert len(members) == len(set(members))


def test_star_shaped_group_survives_cap_by_direct_links():
    # 대표 r 와 직접 쌍이 4개 — 상한 5 안에서 전부 대표 밑에 남아야 한다
    pairs = {("r", f"m{i}") for i in range(4)}
    result = build_groups(pairs, order_of("r", "m0", "m1", "m2", "m3"), GroupConfig(max_group_size=5))

    assert result.groups == 1
    assert result.duplicate_count == {"r": 5}
    assert set(result.duplicate_of) == {"m0", "m1", "m2", "m3"}
