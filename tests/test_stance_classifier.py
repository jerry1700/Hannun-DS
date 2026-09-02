from hannun.stance.classifier import classify_stance


TARGET = "미국이 한미연합훈련을 축소해 북한과 대화 여건을 만드는 것은 바람직하다"


def test_positive_when_article_supports_issue():
    result = classify_stance(
        TARGET,
        "정부는 한미훈련 축소를 환영하며 대화 여건 조성에 긍정적이라고 평가했다.",
    )

    assert result["stance"] == "positive"
    assert 0.0 <= result["stance_confidence"] <= 1.0


def test_negative_when_concern_is_article_direction():
    result = classify_stance(
        TARGET,
        "전문가들은 훈련 축소가 안보 부담을 키울 수 있다며 우려를 나타냈다.",
    )

    assert result["stance"] == "negative"
    assert 0.0 <= result["stance_confidence"] <= 1.0


def test_neutral_when_article_only_reports_facts():
    result = classify_stance(
        TARGET,
        "한미 양국은 다음 달 연합훈련 일정과 규모를 협의할 예정이다.",
    )

    assert result["stance"] == "neutral"


def test_neutral_when_both_sides_are_similarly_presented():
    result = classify_stance(
        TARGET,
        "여당은 훈련 축소를 환영했다. 야당은 안보 공백이 우려된다며 반대했다.",
    )

    assert result["stance"] == "neutral"


def test_positive_when_article_criticizes_issue_opposition():
    result = classify_stance(
        TARGET,
        "훈련 축소에 반대하는 주장은 대화 가능성을 외면한 것이라며 강하게 비판했다.",
    )

    assert result["stance"] == "positive"


def test_unrelated_positive_expression_is_neutral():
    result = classify_stance(
        TARGET,
        "정부는 반도체 산업 지원 정책을 환영하며 긍정적으로 평가했다.",
    )

    assert result["stance"] == "neutral"


def test_unrelated_sentiment_inside_related_article_is_neutral():
    result = classify_stance(
        TARGET,
        "한미훈련 일정과 규모가 발표됐다. "
        "정부는 반도체 산업 지원 정책을 환영했다.",
    )

    assert result["stance"] == "neutral"


def test_unrelated_sentiment_sentence_is_ignored():
    result = classify_stance(
        TARGET,
        "미국은 한미연합훈련 축소 일정을 발표했다. "
        "정부는 반도체 산업 지원 정책을 환영했다.",
    )

    assert result["stance"] == "neutral"


def test_quoted_negative_statement_follows_quote_stance():
    target = "조희대 대법원장의 대법관 후보 제청은 정당하다"

    result = classify_stance(
        target,
        '김민석은 "최악 대법원장, 조희대씨 물러나라"고 말했다.',
    )

    assert result["stance"] == "negative"


def test_quote_with_balanced_counterargument_is_neutral():
    target = "조희대 대법원장의 대법관 후보 제청은 정당하다"

    result = classify_stance(
        target,
        '한쪽은 "조희대씨 물러나라"고 비판했다. '
        '반면 다른 쪽은 조희대 대법원장의 제청을 지지한다고 밝혔다.',
    )

    assert result["stance"] == "neutral"


def test_confidence_is_always_between_zero_and_one():
    cases = [
        ("한미연합훈련 축소를 환영한다.", "positive"),
        ("한미연합훈련 축소로 안보 부담이 우려된다.", "negative"),
        ("한미연합훈련 축소 일정이 발표됐다.", "neutral"),
    ]

    for content, expected_stance in cases:
        result = classify_stance(TARGET, content)

        assert result["stance"] == expected_stance
        assert 0.0 <= result["stance_confidence"] <= 1.0


def test_procedural_approval_is_neutral():
    result = classify_stance(
        "윤관석·이성만 체포동의안 부결",
        "체포동의안이 가결되기 위해서는 재적의원 과반 출석과 "
        "출석 의원 과반의 찬성을 얻어야 한다.",
    )

    assert result["stance"] == "neutral"


def test_support_as_reported_fact_is_neutral():
    result = classify_stance(
        "신상진 성남시장 선거법 위반 벌금형",
        "신 시장은 체육동호회 회원 2만명의 지지 선언을 "
        "받았다는 허위 글을 게시한 혐의로 기소됐다.",
    )

    assert result["stance"] == "neutral"


def test_risk_area_is_neutral():
    result = classify_stance(
        "말라리아 환자 증가",
        "질병청은 말라리아 위험지역을 집중적으로 관리하고 있다.",
    )

    assert result["stance"] == "neutral"


def test_reduced_burden_is_neutral():
    result = classify_stance(
        "남산 터널 혼잡통행료 면제",
        "통행료가 오랫동안 유지되면서 이용자가 체감하는 부담이 줄었다.",
    )

    assert result["stance"] == "neutral"


def test_opposite_trend_is_neutral():
    result = classify_stance(
        "의류 신발 물가 상승",
        "이는 최근 전체 소비자물가 흐름과 반대되는 모습이다.",
    )

    assert result["stance"] == "neutral"


def test_critical_person_description_is_neutral():
    result = classify_stance(
        "한상혁 방통위원장 불구속기소",
        "한 위원장은 TV조선에 비판적 입장을 지닌 시민단체 출신 인사를 "
        "심사위원으로 선임한 혐의를 받는다.",
    )

    assert result["stance"] == "neutral"


def test_direct_concern_is_still_negative():
    result = classify_stance(
        "한미연합훈련 축소",
        "전문가들은 훈련 축소가 안보 공백을 키울 수 있다며 "
        "우려를 나타냈다.",
    )

    assert result["stance"] == "negative"


def test_direct_support_is_still_positive():
    result = classify_stance(
        "한미연합훈련 축소",
        "정부는 한미연합훈련 축소를 지지한다고 밝혔다.",
    )

    assert result["stance"] == "positive"


def test_procedural_approval_is_neutral_variant2():
    result = classify_stance(
        "윤관석 이성만 체포동의안 부결",
        "체포동의안이 가결되려면 재적의원 과반 출석과 "
        "출석 의원 과반의 찬성을 얻어야 한다.",
    )

    assert result["stance"] == "neutral"


def test_reported_support_fact_is_neutral():
    result = classify_stance(
        "신상진 성남시장 선거법 위반 벌금형",
        "신 시장은 체육동호회 회원 2만명의 지지 선언을 "
        "받았다는 허위 글을 게시한 혐의로 기소됐다.",
    )

    assert result["stance"] == "neutral"


def test_risk_area_term_is_neutral():
    result = classify_stance(
        "말라리아 환자 증가",
        "질병청은 말라리아 위험지역을 집중 관리하고 있다.",
    )

    assert result["stance"] == "neutral"


def test_reduced_burden_is_neutral_variant2():
    result = classify_stance(
        "남산 터널 혼잡통행료 면제",
        "혼잡통행료 정책의 효과를 분석했다. "
        "이용자가 체감하는 부담은 줄어든 것으로 나타났다.",
    )

    assert result["stance"] == "neutral"


def test_opposite_trend_is_neutral_variant2():
    result = classify_stance(
        "의류 신발 물가 상승",
        "의류와 신발 물가는 상승했다. "
        "이는 최근 전체 소비자물가 흐름과 반대되는 모습이다.",
    )

    assert result["stance"] == "neutral"


def test_critical_description_is_neutral():
    result = classify_stance(
        "한상혁 TV조선 재승인 불구속기소",
        "한상혁 위원장은 TV조선에 비판적 입장을 지닌 "
        "시민단체 출신 인사를 심사위원으로 선임한 혐의를 받는다.",
    )

    assert result["stance"] == "neutral"


def test_direct_concern_about_issue_is_negative():
    result = classify_stance(
        "한미연합훈련 축소",
        "전문가들은 한미연합훈련 축소가 안보 공백을 "
        "키울 수 있다며 우려를 나타냈다.",
    )

    assert result["stance"] == "negative"


def test_direct_support_for_issue_is_positive():
    result = classify_stance(
        "한미연합훈련 축소",
        "정부는 한미연합훈련 축소를 지지한다고 밝혔다.",
    )

    assert result["stance"] == "positive"


def test_issue_opponent_criticism_is_positive():
    result = classify_stance(
        "용산공원 청년주택 공급",
        "조국은 용산공원 청년주택 공급에 반대하는 오세훈을 "
        "무책임한 선동 정치라고 비판했다.",
    )

    assert result["stance"] == "positive"


def test_balanced_positions_are_neutral():
    result = classify_stance(
        "한미연합훈련 축소",
        "여당은 한미연합훈련 축소를 지지한다고 밝혔다. "
        "야당은 훈련 축소가 안보에 악영향을 줄 수 있다며 "
        "반대한다고 밝혔다.",
    )

    assert result["stance"] == "neutral"


def test_completed_event_is_neutral():
    result = classify_stance(
        "특별법 제정",
        "특별법 제정안이 국회 본회의에서 부결됐다.",
    )

    assert result["stance"] == "neutral"


def test_blocking_action_request_is_negative():
    result = classify_stance(
        "특별법 제정",
        "정부는 대통령에게 특별법 재의요구권 행사를 건의했다.",
    )

    assert result["stance"] == "negative"


def test_criticism_of_blocking_action_is_positive():
    result = classify_stance(
        "특별법 제정",
        "시민단체는 특별법에 대한 거부권 행사를 강하게 규탄했다.",
    )

    assert result["stance"] == "positive"


def test_opposition_in_progress_is_negative():
    result = classify_stance(
        "혼잡통행료 징수",
        "시민단체는 혼잡통행료 징수에 반대하고 있다.",
    )

    assert result["stance"] == "negative"


def test_explicit_approval_is_positive():
    result = classify_stance(
        "특별법 제정",
        "전문가는 국민을 위해 특별법이 통과되는 것이 맞다고 밝혔다.",
    )

    assert result["stance"] == "positive"


def test_factual_veto_execution_is_neutral():
    result = classify_stance(
        "특별법 제정",
        "대통령은 오늘 특별법 제정안에 재의요구권을 행사했다.",
    )

    assert result["stance"] == "neutral"


def test_passage_criticism_is_negative():
    result = classify_stance(
        "특별법 제정",
        "시민단체는 특별법 강행 처리를 폭거라고 규탄했다.",
    )

    assert result["stance"] == "negative"


def test_veto_criticism_supports_original_target():
    result = classify_stance(
        "특별법 제정",
        "야당은 특별법에 대한 대통령의 거부권 행사를 강하게 비난했다.",
    )

    assert result["stance"] == "positive"


def test_factual_rejection_is_neutral():
    result = classify_stance(
        "특별법 제정",
        "특별법 제정안은 재투표 끝에 부결돼 폐기됐다.",
    )

    assert result["stance"] == "neutral"


def test_rhetorical_blocking_reference_does_not_cancel_support():
    result = classify_stance(
        "특별법 제정",
        "전문가는 특별법이 통과되는 것이 맞다며 "
        "정부가 약속을 지킬지 아니면 폐기할지 선택해야 한다고 말했다.",
    )

    assert result["stance"] == "positive"


def test_direct_opposition_action_is_negative():
    result = classify_stance(
        "특별법 제정",
        "관련 단체들은 특별법 통과에 반발해 총파업을 예고했다.",
    )

    assert result["stance"] == "negative"


def test_vote_result_reporting_is_neutral():
    result = classify_stance(
        "특별법 제정",
        "특별법 재의결안은 찬성 178표, 반대 107표로 부결됐다.",
    )

    assert result["stance"] == "neutral"


def test_considering_veto_is_neutral():
    result = classify_stance(
        "특별법 제정",
        "대통령실은 특별법에 재의요구권을 행사할지 고심하고 있다.",
    )

    assert result["stance"] == "neutral"


def test_explicit_opposition_description_is_negative():
    result = classify_stance(
        "특별법 제정",
        "시민단체는 특별법 제정에 반대하고 있다.",
    )

    assert result["stance"] == "negative"


def test_praised_completion_is_positive():
    result = classify_stance(
        "특별법 제정",
        "대표는 특별법 처리를 마무리하게 돼 다행이라고 말했다.",
    )

    assert result["stance"] == "positive"


def test_request_to_stop_passage_is_negative():
    result = classify_stance(
        "특별법 제정",
        "야당은 특별법 강행 처리를 중단해달라고 요구했다.",
    )

    assert result["stance"] == "negative"


def test_blocking_word_inside_other_law_name_does_not_invert():
    result = classify_stance(
        "특별법 제정",
        "관련 단체들은 특별법과 면허취소법 통과에 반발해 파업을 예고했다.",
    )

    assert result["stance"] == "negative"


def test_factual_veto_with_parenthetical_name_is_neutral():
    result = classify_stance(
        "특별법 제정",
        "대통령은 국회를 통과한 특별법 제정안에 "
        "재의요구권(거부권)을 행사했다.",
    )

    assert result["stance"] == "neutral"


def test_explicit_opposition_progressive_form_is_negative():
    result = classify_stance(
        "특별법 제정",
        "관련 단체는 특별법에 반대하고 있다.",
    )

    assert result["stance"] == "negative"


def test_clear_necessity_evaluation_is_positive():
    result = classify_stance(
        "신기술 도입",
        "전문가는 신기술 도입이 산업 경쟁력 강화에 필수불가결하다고 평가했다.",
    )

    assert result["stance"] == "positive"


def test_clear_success_evaluation_is_positive():
    result = classify_stance(
        "기술 협력",
        "관계자는 기술 협력을 대표적인 성공 사례라며 향후 성과가 기대된다고 밝혔다.",
    )

    assert result["stance"] == "positive"


def test_clear_rejection_evaluation_is_negative():
    result = classify_stance(
        "도감청 의혹",
        "대변인은 도감청 의혹을 터무니없는 거짓이라고 강하게 부정했다.",
    )

    assert result["stance"] == "negative"


def test_clear_criticism_evaluation_is_negative():
    result = classify_stance(
        "해양 방류 계획",
        "전문가는 해양 방류 계획을 강행하는 것은 실망스럽고 불안하다고 비판했다.",
    )

    assert result["stance"] == "negative"



def test_poll_support_rate_is_not_stance_signal():
    result = classify_stance(
        "윤 대통령 신년인사회",
        "윤 대통령 신년인사회가 열렸다. "
        "여론조사에서 대통령 지지율은 40%로 집계됐다.",
    )

    assert result["stance"] == "neutral"


def test_factual_promotion_plan_is_neutral():
    result = classify_stance(
        "반도체 세액공제율 상향",
        "정부는 반도체 세액공제율 상향을 "
        "추진할 계획이라고 밝혔다.",
    )

    assert result["stance"] == "neutral"


def test_explicit_target_criticism_is_negative():
    result = classify_stance(
        "윤 대통령 신년인사회에서 정상화에 속도를 내야 한다",
        "야당은 윤 대통령이 통합과 협치를 외면했다며 "
        "정부의 태도를 강하게 비판하고 있다.",
    )

    assert result["stance"] == "negative"


def test_worrisome_target_evaluation_is_negative():
    result = classify_stance(
        "윤 대통령 신년인사회",
        "야당은 윤 대통령의 태도가 "
        "우려스럽다고 평가했다.",
    )

    assert result["stance"] == "negative"



def test_factual_blocking_event_with_blocking_word_in_target_is_neutral():
    result = classify_stance(
        "전장연 지하철 탑승 시위 재개 삼각지역 승차 저지",
        "서울교통공사는 전장연의 지하철 탑승을 저지했다.",
    )

    assert result["stance"] == "neutral"


def test_followup_disappointment_evaluation_is_negative():
    result = classify_stance(
        "주호영 문재인 신년사 평가",
        "주호영은 문재인 전 대통령의 신년사를 평가했다. "
        "그는 내용을 보고 매우 실망스러웠다고 밝혔다.",
    )

    assert result["stance"] == "negative"


def test_clear_progress_evaluation_is_positive():
    result = classify_stance(
        "고체연료 우주발사체 시험",
        "국방부는 고체연료 우주발사체 시험 결과를 공개했다. "
        "이번 결과는 중요한 이정표이며 "
        "우주강국 도약을 위해 진일보한 것이라고 평가했다.",
    )

    assert result["stance"] == "positive"


def test_factual_test_success_alone_is_neutral():
    result = classify_stance(
        "고체연료 우주발사체 시험",
        "국방부는 고체연료 우주발사체의 "
        "두 번째 비행시험에 성공했다고 밝혔다.",
    )

    assert result["stance"] == "neutral"
