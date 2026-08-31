from hannun.dedup.verify import VerifyConfig, verify_pairs

BASE = (
    "정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다. "
    "국토교통부는 수도권에 청년 주택 3만 호를 공급하고 전세 보증금 대출 한도를 늘리기로 했다. "
    "고용노동부는 중소기업에 취업하는 청년에게 2년간 장려금을 지급하는 방안을 내놓았다. "
    "야당은 재원 대책이 빠졌다며 국회 심의 과정에서 따져 보겠다고 밝혔다. "
    "전문가들은 공급 물량이 수요에 못 미친다고 지적했다. 시민단체는 전세 사기 대책이 빠졌다고 비판했다. "
    "청년단체는 장려금보다 정규직 전환 지원이 우선이라고 주장했다. 정부는 추가 대책을 검토하겠다고 답했다. "
    "서울시는 자체 청년 주택 사업과의 중복 여부를 확인하겠다고 밝혔다. 국회는 다음 주 상임위에서 논의를 시작한다. "
    "기획재정부는 내년 예산안에 관련 항목을 반영하는 방안을 들여다보고 있다. "
    "부동산 업계에서는 공급 시점이 문제라며 착공까지 걸리는 기간을 변수로 꼽았다. "
    "대학가 주변 임대료가 최근 두 자릿수로 오른 것도 이번 대책의 배경으로 지목된다. "
    "정부 관계자는 관계 부처 협의를 거쳐 다음 달 중 세부 지침을 내놓겠다고 말했다."
)
# 코사인 경로: 700자 글에서 낱말 하나와 꼬리 문장 하나 차이 → 0.95 를 넘는다.
# 글이 300자쯤으로 짧으면 같은 수정에도 0.95 아래로 내려간다 — 그때는 containment 가 받는다.
REPRINT = BASE.replace("발표했다", "발표하였다") + " 자세한 내용은 부처 홈페이지에 게시된다."
REWRITE = (
    "청년 주거와 일자리를 함께 손보는 종합 대책이 나왔다. 수도권 임대 물량을 늘리고 보증금 부담을 줄이는 "
    "내용이 담겼다. 중소기업 취업자에게는 일정 기간 수당을 얹어 준다. 국회에서는 예산 논쟁이 예상된다. "
    "시민사회에서는 실효성을 두고 평가가 엇갈렸다. 다음 달 공청회가 열릴 예정이다."
)
BULLETIN = BASE[:310]
UNRELATED = (
    "프로야구 한화가 기아를 상대로 9회말 역전승을 거두며 3연승을 달렸다. "
    "선발 투수가 6이닝을 2실점으로 막았고 불펜이 무실점으로 뒤를 받쳤다. "
    "타선에서는 4번 타자가 결승 2타점 2루타를 때려 승리를 이끌었다."
)
TEXTS = {"base": BASE, "reprint": REPRINT, "rewrite": REWRITE, "bulletin": BULLETIN, "unrelated": UNRELATED}


def test_reprint_confirmed_by_cosine():
    result = verify_pairs(TEXTS, {("base", "reprint")})
    assert ("base", "reprint") in result.confirmed
    assert result.method[("base", "reprint")] == "cosine"


def test_bulletin_contained_in_full_article_confirmed_by_containment():
    result = verify_pairs(TEXTS, {("base", "bulletin")})
    assert ("base", "bulletin") in result.confirmed
    assert result.method[("base", "bulletin")] == "containment"


def test_rewrite_is_rejected_not_a_duplicate():
    # 재작성은 같은 이슈지만 중복이 아니다. 묶는 것은 STEP 3 의 몫 — 여기서 떨어져야 설계대로다.
    result = verify_pairs(TEXTS, {("base", "rewrite")})
    assert result.confirmed == set()
    assert result.rejected == 1


def test_short_containment_is_rejected_below_min_len():
    texts = dict(TEXTS, snippet=BASE[:120])
    result = verify_pairs(texts, {("base", "snippet")})
    assert result.confirmed == set()

    loose = verify_pairs(texts, {("base", "snippet")}, VerifyConfig(containment_min_len=100))
    assert ("base", "snippet") in loose.confirmed


def test_unrelated_candidate_is_rejected():
    result = verify_pairs(TEXTS, {("base", "unrelated")})
    assert result.confirmed == set()


def test_empty_pairs_do_not_crash():
    result = verify_pairs(TEXTS, set())
    assert result.confirmed == set() and result.rejected == 0
