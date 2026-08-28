from hannun.stance.evidence import (
    build_issue_evidence,
    build_sentence_evidence,
    group_evidence_by_sentence,
)


def test_build_sentence_evidence():
    article = {
        "article_id": "sha256:test123",
        "publisher_name": "연합뉴스",
        "content": (
            "정부는 오늘 정책을 발표했다. "
            "추가 지원이 필요하다. "
            "관련 예산은 100억 원이다."
        ),
    }

    result = build_sentence_evidence(article)

    assert result == [
        {
            "sentence": "정부는 오늘 정책을 발표했다.",
            "article_id": "sha256:test123",
            "publisher_name": "연합뉴스",
        },
        {
            "sentence": "관련 예산은 100억 원이다.",
            "article_id": "sha256:test123",
            "publisher_name": "연합뉴스",
        },
    ]


def test_build_issue_evidence_preserves_article_sources():
    articles = [
        {
            "article_id": "sha256:article1",
            "publisher_name": "연합뉴스",
            "content": "정부는 정책을 발표했다.",
        },
        {
            "article_id": "sha256:article2",
            "publisher_name": "한겨레",
            "content": "야당은 정책에 반발했다.",
        },
    ]

    result = build_issue_evidence(articles)

    assert result == [
        {
            "sentence": "정부는 정책을 발표했다.",
            "article_id": "sha256:article1",
            "publisher_name": "연합뉴스",
        },
        {
            "sentence": "야당은 정책에 반발했다.",
            "article_id": "sha256:article2",
            "publisher_name": "한겨레",
        },
    ]


def test_group_evidence_by_sentence():
    evidence = [
        {
            "sentence": "정부는 정책을 발표했다.",
            "article_id": "sha256:article1",
            "publisher_name": "연합뉴스",
        },
        {
            "sentence": "정부는 정책을 발표했다.",
            "article_id": "sha256:article2",
            "publisher_name": "한겨레",
        },
    ]

    result = group_evidence_by_sentence(evidence)

    assert result == [
        {
            "sentence": "정부는 정책을 발표했다.",
            "sources": [
                {
                    "article_id": "sha256:article1",
                    "publisher_name": "연합뉴스",
                },
                {
                    "article_id": "sha256:article2",
                    "publisher_name": "한겨레",
                },
            ],
        }
    ]
