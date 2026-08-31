from hannun.stance.quote_parser import extract_quotes


def test_extract_double_quoted_text():
    text = '정부 관계자는 "올해 안에 시행하겠다"고 밝혔다.'

    result = extract_quotes(text)

    assert result == ["올해 안에 시행하겠다"]


def test_extract_curly_double_quoted_text():
    text = "정부 관계자는 “올해 안에 시행하겠다”고 밝혔다."

    result = extract_quotes(text)

    assert result == ["올해 안에 시행하겠다"]


def test_extract_quotes_returns_empty_list_without_quotes():
    assert extract_quotes("") == []
    assert extract_quotes("정부는 오늘 정책을 발표했다.") == []


def test_extract_multiple_quotes():
    text = (
        '정부는 "정책을 추진하겠다"고 밝혔다. '
        '야당은 “재검토가 필요하다”고 말했다.'
    )

    result = extract_quotes(text)

    assert result == [
        "정책을 추진하겠다",
        "재검토가 필요하다",
    ]


def test_single_quoted_label_is_not_extracted():
    text = "정부는 ‘3대 메가 프로젝트’를 추진하고 있다."

    result = extract_quotes(text)

    assert result == []


def test_unclosed_quote_is_not_extracted():
    text = '정부 관계자는 "정책을 추진하겠다고 밝혔다.'

    result = extract_quotes(text)

    assert result == []


def test_parse_quote_sentences_structures_reported_speech():
    from hannun.stance.quote_parser import parse_quote_sentences

    text = (
        '정부는 "정책을 추진하겠다"고 밝혔다. '
        "야당은 재검토가 필요하다고 말했다. "
        "관련 예산은 100억 원이다."
    )

    result = parse_quote_sentences(text)

    assert result == [
        {
            "sentence": '정부는 "정책을 추진하겠다"고 밝혔다.',
            "quotes": ["정책을 추진하겠다"],
            "speech_type": "direct_quote",
        },
        {
            "sentence": "야당은 재검토가 필요하다고 말했다.",
            "quotes": [],
            "speech_type": "reported_speech",
        },
    ]


def test_parse_curly_direct_quote_without_reporting_marker():
    from hannun.stance.quote_parser import parse_quote_sentences

    text = "관계자의 답변은 “올해 안에 시행하겠다”."

    result = parse_quote_sentences(text)

    assert result == [
        {
            "sentence": "관계자의 답변은 “올해 안에 시행하겠다”.",
            "quotes": ["올해 안에 시행하겠다"],
            "speech_type": "direct_quote",
        }
    ]


def test_straight_single_quoted_label_is_not_reported_speech():
    from hannun.stance.quote_parser import parse_quote_sentences

    text = "정부는 '3대 메가 프로젝트'를 추진하고 있다."

    result = parse_quote_sentences(text)

    assert result == []
