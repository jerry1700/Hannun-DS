from .pipeline import preprocess_article


def build_sentence_evidence(article: dict) -> list[dict]:
    """기사의 정제 문장에 출처 정보를 연결한다."""

    sentences = preprocess_article(article["content"])

    return [
        {
            "sentence": sentence,
            "article_id": article["article_id"],
            "publisher_name": article["publisher_name"],
        }
        for sentence in sentences
    ]


def build_issue_evidence(articles: list[dict]) -> list[dict]:
    """여러 기사의 문장별 근거를 하나의 목록으로 만든다."""

    evidence = []

    for article in articles:
        evidence.extend(build_sentence_evidence(article))

    return evidence


def group_evidence_by_sentence(evidence: list[dict]) -> list[dict]:
    """같은 문장의 출처를 하나의 sources 목록으로 묶는다."""

    grouped = {}

    for item in evidence:
        sentence = item["sentence"]

        if sentence not in grouped:
            grouped[sentence] = {
                "sentence": sentence,
                "sources": [],
            }

        grouped[sentence]["sources"].append(
            {
                "article_id": item["article_id"],
                "publisher_name": item["publisher_name"],
            }
        )

    return list(grouped.values())
