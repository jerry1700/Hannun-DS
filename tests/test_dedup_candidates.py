from hannun.dedup.candidates import CandidateConfig, find_candidate_pairs, shingle_set

LONG = (
    "정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다. "
    "국토교통부는 수도권에 청년 주택 3만 호를 공급하고 전세 보증금 대출 한도를 늘리기로 했다. "
    "고용노동부는 중소기업에 취업하는 청년에게 2년간 장려금을 지급하는 방안을 내놓았다. "
    "야당은 재원 대책이 빠졌다며 국회 심의 과정에서 따져 보겠다고 밝혔다."
)
# 제목만 다듬고 한 문장을 덧붙인 전재 — 자카드가 높다
NEAR = LONG + " 정부는 다음 달 세부 시행 계획을 발표할 예정이다."
# 종합기사(길다)와 그 앞부분만 떼어 낸 속보 — 자카드는 0.2 이하라 자카드 LSH 가 못 잡고
# containment 는 1.0 이라 Ensemble 만 잡는다. 이중 인덱스를 두는 이유가 이 쌍이다.
LONGER = LONG + (
    " 전문가들은 공급 물량이 수요에 못 미친다고 지적했다. 시민단체는 전세 사기 대책이 빠졌다고 비판했다. "
    "청년단체는 장려금보다 정규직 전환 지원이 우선이라고 주장했다. 정부는 추가 대책을 검토하겠다고 답했다. "
    "서울시는 자체 청년 주택 사업과의 중복 여부를 확인하겠다고 밝혔다. 국회는 다음 주 상임위에서 논의한다."
)
SHORT_CONTAINED = LONGER[:100]
OTHER = (
    "프로야구 한화가 기아를 상대로 9회말 역전승을 거두며 3연승을 달렸다. "
    "선발 투수가 6이닝을 2실점으로 막았고 불펜이 무실점으로 뒤를 받쳤다. "
    "타선에서는 4번 타자가 결승 2타점 2루타를 때려 승리를 이끌었다."
)


def rows(**named):
    return [{"article_id": k, "text": v} for k, v in named.items()]


def test_near_identical_pair_is_candidate():
    result = find_candidate_pairs(rows(a=LONG, b=NEAR, z=OTHER))
    assert ("a", "b") in result.pairs
    assert all("z" not in pair for pair in result.pairs)


def test_short_article_contained_in_long_is_candidate():
    result = find_candidate_pairs(rows(long=LONGER, short=SHORT_CONTAINED))
    assert ("long", "short") in result.pairs
    assert result.from_containment >= 1
    assert result.from_jaccard == 0  # 자카드 LSH 는 이 쌍을 못 본다 — Ensemble 이 잡았다는 증거


def test_unrelated_articles_produce_no_pairs():
    result = find_candidate_pairs(rows(a=LONG, z=OTHER))
    assert result.pairs == set()


def test_pairs_are_canonical_and_deduplicated():
    result = find_candidate_pairs(rows(b=NEAR, a=LONG))
    assert ("a", "b") in result.pairs
    assert ("b", "a") not in result.pairs


def test_text_shorter_than_shingle_is_skipped_not_crashed():
    result = find_candidate_pairs(rows(a="속보", b=LONG))
    assert result.skipped_short == 1
    assert result.pairs == set()


def test_same_input_gives_same_result():
    articles = rows(a=LONG, b=NEAR, c=SHORT_CONTAINED, z=OTHER)
    first = find_candidate_pairs(articles)
    second = find_candidate_pairs(articles)
    assert first.pairs == second.pairs


def test_shingle_set_size_and_config():
    assert "정부는 " in shingle_set(LONG)
    assert len(shingle_set("가나다라", 4)) == 1
    loose = find_candidate_pairs(rows(a=LONGER, c=SHORT_CONTAINED),
                                 CandidateConfig(containment_threshold=0.5))
    assert ("a", "c") in loose.pairs
