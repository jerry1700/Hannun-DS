import json
from pathlib import Path
from typing import Any

from hannun.stance.classifier import (
    classify_stance,
    select_stance_evidence_sentences,
)
from hannun.stance.keyword_extractor import extract_stance_analysis

from .event_name import generate_event_name
from .fact_summary import generate_fact_summary
from .schema import validate_ds1_issue, validate_ds2_output


def enrich_issue(issue_data: dict[str, Any]) -> dict[str, Any]:
    """
    DS1 이슈 데이터를 입력받아
    실제 DS2 분석 결과를 Enrichment 형식으로 반환한다.
    """

    # DS1 입력 스키마 검증
    validate_ds1_issue(issue_data)

    issue_cluster_id = issue_data["issue_cluster_id"]
    input_keywords = issue_data.get("keywords") or []

    # 이슈 단위 분석
    event_name = generate_event_name(issue_data)
    fact_summary = generate_fact_summary(issue_data)

    # 기사별 관점 분석
    enriched_articles = []
    analysis_articles = []

    for article in issue_data.get("articles", []):
        stance_result = classify_stance(
            event_name,
            article["content"],
        )

        stance = stance_result["stance"]
        stance_confidence = stance_result["stance_confidence"]

        enriched_articles.append(
            {
                "article_id": article["article_id"],
                "stance": stance,
                "stance_confidence": stance_confidence,
            }
        )

        # Target 관련 문장 중 최종 stance와 같은 방향의
        # 근거 문장만 관점별 키워드·대표 표현 분석에 사용한다.
        evidence_sentences = select_stance_evidence_sentences(
            event_name,
            article["content"],
            stance,
        )

        if evidence_sentences:
            analysis_articles.append(
                {
                    "content": " ".join(evidence_sentences),
                    "stance": stance,
                }
            )

    viewpoint_analysis = extract_stance_analysis(
        analysis_articles
    )

    # 최종 DS2 Enrichment 출력
    result = {
        "issue_cluster_id": issue_cluster_id,
        "event_name": event_name,
        "keywords": input_keywords,
        "fact_summary": fact_summary,
        "articles": enriched_articles,
        "viewpoint_analysis": viewpoint_analysis,
    }

    # DS2 출력 스키마 검증
    validate_ds2_output(result)

    return result


def enrich_dummy(issue_data: dict[str, Any]) -> dict[str, Any]:
    """
    기존 테스트 및 호출부 호환을 위한 래퍼.

    실제 분석은 enrich_issue에서 수행한다.
    """
    return enrich_issue(issue_data)


if __name__ == "__main__":
    ds_root = Path(__file__).resolve().parents[3]

    sample_path = (
        ds_root
        / "samples"
        / "ds1_issue_output_v1.example.json"
    )

    with sample_path.open("r", encoding="utf-8") as file:
        issues = json.load(file)

    result = enrich_issue(issues[0])

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
