from typing import Any


DS1_REQUIRED_FIELDS = {
    "issue_cluster_id",
    "representative_title",
    "articles",
}

DS1_ARTICLE_REQUIRED_FIELDS = {
    "article_id",
    "title",
    "content",
    "publisher_name",
}

DS2_REQUIRED_FIELDS = {
    "issue_cluster_id",
    "event_name",
    "keywords",
    "fact_summary",
    "articles",
    "viewpoint_analysis",
}

DS2_ARTICLE_REQUIRED_FIELDS = {
    "article_id",
    "stance",
    "stance_confidence",
    "viewpoint_group_label",
}


def _require_fields(
    data: dict[str, Any],
    required_fields: set[str],
    context: str,
) -> None:
    missing = required_fields - data.keys()

    if missing:
        raise ValueError(
            f"{context} missing required fields: {sorted(missing)}"
        )


def validate_ds1_issue(issue_data: dict[str, Any]) -> None:
    _require_fields(
        issue_data,
        DS1_REQUIRED_FIELDS,
        "DS1 issue",
    )

    if not isinstance(issue_data["issue_cluster_id"], str):
        raise TypeError("issue_cluster_id must be str")

    if not isinstance(issue_data["articles"], list):
        raise TypeError("articles must be list")

    for index, article in enumerate(issue_data["articles"]):
        _require_fields(
            article,
            DS1_ARTICLE_REQUIRED_FIELDS,
            f"DS1 article[{index}]",
        )

        if not isinstance(article["article_id"], str):
            raise TypeError(
                f"DS1 article[{index}].article_id must be str"
            )


def validate_ds2_output(result: dict[str, Any]) -> None:
    _require_fields(
        result,
        DS2_REQUIRED_FIELDS,
        "DS2 output",
    )

    if not isinstance(result["articles"], list):
        raise TypeError("DS2 articles must be list")

    valid_stances = {
        "positive",
        "negative",
        "neutral",
    }

    for index, article in enumerate(result["articles"]):
        _require_fields(
            article,
            DS2_ARTICLE_REQUIRED_FIELDS,
            f"DS2 article[{index}]",
        )

        if article["stance"] not in valid_stances:
            raise ValueError(
                f"DS2 article[{index}].stance is invalid"
            )

        label = article["viewpoint_group_label"]

        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"DS2 article[{index}].viewpoint_group_label "
                "must be non-empty str"
            )

        if len(label) > 8:
            raise ValueError(
                f"DS2 article[{index}].viewpoint_group_label "
                "must be 8 characters or fewer"
            )

        confidence = article["stance_confidence"]

        if confidence is not None:
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError(
                    f"DS2 article[{index}].stance_confidence "
                    "must be between 0.0 and 1.0 or null"
                )
