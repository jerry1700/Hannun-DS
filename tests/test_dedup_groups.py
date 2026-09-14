from hannun.dedup.groups import GroupConfig, build_groups


def order_of(*ids):
    # 나열 순서가 곧 대표 우선순위 — 테스트에서는 발행 시각 대신 순번을 쓴다
    return {article_id: (i,) for i, article_id in enumerate(ids)}


def test_chained_article_is_not_folded():
    # a~b, b~c 연쇄에서 c 는 대표 a 와 직접 확정된 적이 없다 — 접지 않는다.
    # 실측 검수에서 이런 접힘은 대부분 다른 기사였다(티켓 10 §4.5)
    result = build_groups({("a", "b"), ("b", "c")}, order_of("a", "b", "c"))

    assert result.groups == 1
    assert result.duplicate_of == {"b": "a"}
    assert result.duplicate_count == {"a": 2}


def test_detached_chain_regroups_among_itself():
    # a-b-c-d 일렬: [a,b] 별을 뜨고 남은 c~d 는 서로 직접 쌍이라 자기들끼리 새 그룹이 된다
    result = build_groups({("a", "b"), ("b", "c"), ("c", "d")}, order_of("a", "b", "c", "d"))

    assert result.duplicate_of == {"b": "a", "d": "c"}
    assert result.duplicate_count == {"a": 2, "c": 2}


def test_separate_pairs_form_separate_groups():
    result = build_groups({("a", "b"), ("x", "y")}, order_of("a", "b", "x", "y"))

    assert result.groups == 2
    assert result.duplicate_of == {"b": "a", "y": "x"}
    assert result.duplicate_count == {"a": 2, "x": 2}


def test_no_pairs_no_groups():
    result = build_groups(set(), {})
    assert result.groups == 0 and result.duplicate_of == {}


def test_long_chain_becomes_pairwise_stars():
    # a00 - a01 - a02 - … 일렬 연쇄 40개는 직접 쌍 단위 (a00,a01), (a02,a03) … 로만 접힌다
    ids = [f"a{i:02d}" for i in range(40)]
    chain = {(ids[i], ids[i + 1]) for i in range(39)}
    result = build_groups(chain, order_of(*ids), GroupConfig(max_group_size=5))

    assert all(count == 2 for count in result.duplicate_count.values())
    assert result.duplicate_of["a01"] == "a00"
    # 모든 기사는 정확히 한 그룹에만 속한다
    members = list(result.duplicate_of) + list(result.duplicate_count)
    assert len(members) == len(set(members))


def test_oversized_star_is_trimmed_at_cap():
    # 대표와 직접 쌍 9개, 상한 5 — 대표 포함 5개만 접고 나머지는 단독으로 남긴다
    pairs = {("r", f"m{i}") for i in range(9)}
    result = build_groups(pairs, order_of("r", *[f"m{i}" for i in range(9)]),
                          GroupConfig(max_group_size=5))

    assert result.split_oversized == 1
    assert result.duplicate_count == {"r": 5}
    assert len(result.duplicate_of) == 4


def test_star_shaped_group_survives_cap_by_direct_links():
    # 대표 r 와 직접 쌍이 4개 — 상한 5 안에서 전부 대표 밑에 남아야 한다
    pairs = {("r", f"m{i}") for i in range(4)}
    result = build_groups(pairs, order_of("r", "m0", "m1", "m2", "m3"), GroupConfig(max_group_size=5))

    assert result.groups == 1
    assert result.duplicate_count == {"r": 5}
    assert set(result.duplicate_of) == {"m0", "m1", "m2", "m3"}
