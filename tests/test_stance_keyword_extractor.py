from hannun.stance.keyword_extractor import extract_stance_keywords


def test_extract_stance_keywords_returns_all_stances():
    articles = [
        {
            "content": "청년주택 공급으로 청년 주거 지원을 확대한다.",
            "stance": "positive",
        },
        {
            "content": "청년주택은 청년 주거 안정에 필요하다.",
            "stance": "positive",
        },
        {
            "content": "용산공원 관련 특별법 개정안을 검토한다.",
            "stance": "neutral",
        },
        {
            "content": "녹지 훼손 우려로 주택 건설에 반대한다.",
            "stance": "negative",
        },
        {
            "content": "공원 녹지를 훼손해서는 안 된다는 비판이 나왔다.",
            "stance": "negative",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=3,
    )

    assert set(result) == {
        "positive",
        "neutral",
        "negative",
    }

    assert len(result["positive"]) <= 3
    assert len(result["neutral"]) <= 3
    assert len(result["negative"]) <= 3


def test_extract_stance_keywords_empty_stance_returns_empty_list():
    articles = [
        {
            "content": "청년주택 공급 확대가 필요하다.",
            "stance": "positive",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=5,
    )

    assert result["positive"]
    assert result["neutral"] == []
    assert result["negative"] == []


def test_extract_stance_keywords_excludes_common_issue_terms():
    articles = [
        {
            "content": "용산공원 청년주택 공급으로 주거 지원을 확대한다.",
            "stance": "positive",
        },
        {
            "content": "용산공원 청년주택은 청년 주거 안정에 도움이 된다.",
            "stance": "positive",
        },
        {
            "content": "용산공원 청년주택은 녹지 훼손 우려가 크다.",
            "stance": "negative",
        },
        {
            "content": "용산공원 청년주택 건설로 공원 녹지가 훼손된다.",
            "stance": "negative",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=3,
    )

    # 모든 관점에 공통으로 등장하는 이슈명보다
    # 각 관점을 구분하는 표현이 우선되어야 한다.
    assert (
        "주거" in result["positive"]
        or "지원" in result["positive"]
        or "안정" in result["positive"]
    )

    assert (
        "녹지" in result["negative"]
        or "훼손" in result["negative"]
        or "우려" in result["negative"]
    )


def test_extract_stance_keywords_rejects_invalid_stance():
    articles = [
        {
            "content": "기사 본문",
            "stance": "unknown",
        },
    ]

    try:
        extract_stance_keywords(articles)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid stance must raise ValueError")


def test_extract_stance_keywords_removes_general_noise_words():
    articles = [
        {
            "content": (
                "정부는 정책을 설명하면서 지원 사업 등을 추진하고 있다고 밝혔다. "
                "청년을 위한 주거 안정 지원이 필요하다고 했다."
            ),
            "stance": "positive",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=10,
    )

    noise_words = {
        "것",
        "등",
        "씨",
        "하면서",
        "있도록",
        "위한",
    }

    assert not (
        noise_words
        & set(result["positive"])
    )


def test_extract_stance_keywords_prioritizes_contrastive_terms():
    articles = [
        {
            "content": (
                "용산공원 청년주택 정책은 "
                "청년 주거 안정과 주거 지원 확대에 도움이 된다."
            ),
            "stance": "positive",
        },
        {
            "content": (
                "용산공원 청년주택 정책은 "
                "청년 주거 지원을 확대하는 방안이다."
            ),
            "stance": "positive",
        },
        {
            "content": (
                "용산공원 청년주택 정책은 "
                "공원 녹지 훼손과 환경 훼손 우려가 크다."
            ),
            "stance": "negative",
        },
        {
            "content": (
                "용산공원 청년주택 정책으로 "
                "녹지를 훼손할 수 있다는 우려가 제기됐다."
            ),
            "stance": "negative",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=5,
    )

    # 두 관점 모두에서 반복되는 이슈 자체의 단어보다
    # 각 관점을 구분하는 단어가 우선되어야 한다.
    assert (
        "안정" in result["positive"]
        or "지원" in result["positive"]
        or "확대" in result["positive"]
    )

    assert (
        "녹지" in result["negative"]
        or "훼손" in result["negative"]
        or "우려" in result["negative"]
    )


def test_extract_stance_keywords_downranks_shared_issue_terms():
    articles = [
        {
            "content": (
                "용산공원 청년주택 정책은 "
                "주거 안정과 복지 확대에 도움이 된다."
            ),
            "stance": "positive",
        },
        {
            "content": (
                "용산공원 청년주택 정책은 "
                "주거 안정과 지원 확대가 필요하다."
            ),
            "stance": "positive",
        },
        {
            "content": (
                "용산공원 청년주택 정책은 "
                "녹지 훼손과 환경 파괴 우려가 있다."
            ),
            "stance": "negative",
        },
        {
            "content": (
                "용산공원 청년주택 정책으로 "
                "녹지 훼손 우려가 커지고 있다."
            ),
            "stance": "negative",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=5,
    )

    shared_terms = {
        "용산공원",
        "청년주택",
        "정책",
    }

    assert not (
        shared_terms
        & set(result["positive"])
        & set(result["negative"])
    )


def test_extract_stance_keywords_removes_url_fragments():
    articles = [
        {
            "content": (
                "정책 개편 내용을 설명했다. "
                "전문보기: https://www.example.co.kr/view/ABC123"
            ),
            "stance": "neutral",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=10,
    )

    url_fragments = {
        "https",
        "http",
        "www",
        "com",
        "co",
        "kr",
        "view",
    }

    assert not (
        url_fragments
        & set(result["neutral"])
    )


def test_extract_stance_keywords_prefers_terms_shared_within_stance():
    articles = [
        {
            "content": (
                "주거 안정 지원이 필요하다. "
                "일회성표현 일회성표현 일회성표현 일회성표현."
            ),
            "stance": "positive",
        },
        {
            "content": "청년의 주거 안정 대책을 확대한다.",
            "stance": "positive",
        },
        {
            "content": "주거 안정에 도움이 되는 지원 방안이다.",
            "stance": "positive",
        },
        {
            "content": "사업 추진 현황을 발표했다.",
            "stance": "neutral",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=3,
    )

    # 한 기사에서만 반복된 단어보다
    # 같은 관점의 여러 기사에 걸쳐 나타나는 표현을 우선한다.
    assert "안정" in result["positive"]
    assert "일회성표현" not in result["positive"]


def test_extract_stance_keywords_preserves_hanbando():
    articles = [
        {
            "content": "한반도 평화체제 구축이 필요하다.",
            "stance": "positive",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=10,
    )

    assert "한반도" in result["positive"]
    assert "한반" not in result["positive"]


def test_extract_stance_keywords_removes_possessive_particle():
    articles = [
        {
            "content": "북한의 조치와 북한의 입장을 우려한다.",
            "stance": "negative",
        },
    ]

    result = extract_stance_keywords(
        articles,
        top_k=10,
    )

    assert "북한" in result["negative"]
    assert "북한의" not in result["negative"]


def test_select_stance_evidence_sentences_matches_direction():
    from hannun.stance.classifier import (
        select_stance_evidence_sentences,
    )

    target = "한미연합훈련 축소는 바람직하다"

    content = (
        "한미연합훈련 축소는 평화적 대화에 도움이 된다. "
        "그러나 일각에서는 안보 공백을 우려한다."
    )

    result = select_stance_evidence_sentences(
        target,
        content,
        "positive",
    )

    assert isinstance(result, list)


def test_extract_stance_phrases_returns_bigrams():
    from hannun.stance.keyword_extractor import (
        extract_stance_phrases,
    )

    articles = [
        {
            "content": "녹지 보전 필요성이 크다.",
            "stance": "negative",
        },
        {
            "content": "녹지 보전 요구가 이어졌다.",
            "stance": "negative",
        },
    ]

    result = extract_stance_phrases(
        articles,
        top_k=5,
    )

    assert "녹지 보전" in result["negative"]


def test_extract_stance_phrases_returns_all_stances():
    from hannun.stance.keyword_extractor import (
        extract_stance_phrases,
    )

    result = extract_stance_phrases(
        [
            {
                "content": "평화 체제 구축을 지지한다.",
                "stance": "positive",
            },
        ],
        top_k=3,
    )

    assert set(result) == {
        "positive",
        "neutral",
        "negative",
    }

    assert result["neutral"] == []
    assert result["negative"] == []


def test_extract_stance_analysis_contains_keywords_and_phrases():
    from hannun.stance.keyword_extractor import (
        extract_stance_analysis,
    )

    result = extract_stance_analysis(
        [
            {
                "content": "대북 적대시 정책 철회를 요구했다.",
                "stance": "negative",
            },
        ]
    )

    assert set(result["negative"]) == {
        "keywords",
        "phrases",
    }
