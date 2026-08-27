import json
from pathlib import Path
from typing import Any

from .schema import validate_ds1_issue, validate_ds2_output


def enrich_dummy(issue_data: dict[str, Any]) -> dict[str, Any]:
    """
    DS1 이슈 데이터를 입력받아
    DS2의 더미 enrichment 결과를 반환한다.

    실제 알고리즘 연결 전,
    DS1 -> DS2 입출력 구조를 확인하기 위한 테스트용 함수다.
    """

    # DS1 입력 스키마 검증
    validate_ds1_issue(issue_data)

    issue_cluster_id = issue_data["issue_cluster_id"]

    # DS1 대표 제목을 더미 event_name으로 임시 사용
    representative_title = issue_data.get("representative_title", "")

    # DS1 keywords가 비어 있을 수 있으므로 더미값 허용
    input_keywords = issue_data.get("keywords") or []

    # 기사별 stance 더미 결과 생성
    enriched_articles = []

    for article in issue_data.get("articles", []):
        enriched_articles.append(
            {
                "article_id": article["article_id"],
                "stance": "neutral",
                "stance_confidence": None,
            }
        )

    # DS2 더미 출력 생성
    result = {
        "issue_cluster_id": issue_cluster_id,
        "event_name": representative_title or "[DUMMY] 이슈명",
        "keywords": input_keywords or ["dummy-keyword"],
        "fact_summary": [
            "[DUMMY] 공통 사실 요약 1",
            "[DUMMY] 공통 사실 요약 2",
            "[DUMMY] 공통 사실 요약 3",
        ],
        "articles": enriched_articles,
        "viewpoint_analysis": "[DUMMY] 이슈 내 기사들의 관점 분석 결과",
    }

    # DS2 출력 스키마 검증
    validate_ds2_output(result)

    return result


if __name__ == "__main__":
    ds_root = Path(__file__).resolve().parents[3]

    sample_path = (
        ds_root
        / "samples"
        / "ds1_issue_output_v1.example.json"
    )

    with sample_path.open("r", encoding="utf-8") as file:
        issues = json.load(file)

    # 샘플 데이터의 첫 번째 이슈만 테스트
    result = enrich_dummy(issues[0])

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )