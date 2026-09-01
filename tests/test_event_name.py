import json
from pathlib import Path

from hannun.enrichment.event_name import generate_event_name


SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "ds1_issue_output_v1.example.json"
)


def test_generate_event_name_from_similar_article_titles():
    issue = {
        "representative_title": "인하대 반도체·바이오 교육동 준공",
        "articles": [
            {
                "title": "인하대 반도체·바이오 교육동 준공…첨단산업 인재 양성",
            },
            {
                "title": "인하대, 반도체·바이오 교육동 준공…클린룸 등 실습실 갖춰",
            },
            {
                "title": "인하대 반도체·바이오 교육동 준공…실무 인재 양성",
            },
            {
                "title": "프로야구 개막전 관중 신기록",
            },
        ],
    }

    event_name = generate_event_name(issue)

    assert "인하대" in event_name
    assert "교육동" in event_name
    assert "프로야구" not in event_name


def test_representative_title_is_used_when_articles_are_empty():
    issue = {
        "representative_title": "KDI, 올해 성장률 3.2%로 상향",
        "articles": [],
    }

    assert generate_event_name(issue) == "KDI, 올해 성장률 3.2%로 상향"


def test_news_prefix_is_removed():
    issue = {
        "representative_title": "[속보] 간호법 재의안 국회 본회의서 부결",
        "articles": [],
    }

    assert generate_event_name(issue) == "간호법 재의안 국회 본회의서 부결"


def test_existing_ds1_sample_generates_non_dummy_event_name():
    with SAMPLE.open("r", encoding="utf-8") as file:
        issues = json.load(file)

    for issue in issues:
        event_name = generate_event_name(issue)

        assert isinstance(event_name, str)
        assert event_name.strip()
        assert "[DUMMY]" not in event_name


def test_neutral_central_title_wins_over_editorial_outlier():
    issue = {
        "representative_title": "간호법 제정안 국회 본회의 통과",
        "articles": [
            {
                "title": "간호법 제정안 국회 본회의 통과",
            },
            {
                "title": "간호법 국회 본회의 통과…의료계 반응 엇갈려",
            },
            {
                "title": "간호법 제정안 본회의 통과…향후 절차 주목",
            },
            {
                "title": "野 입법 폭주 간호법 강행처리",
            },
        ],
    }

    event_name = generate_event_name(issue)

    assert "폭주" not in event_name
    assert "강행처리" not in event_name
    assert "간호법" in event_name


def test_multiline_contaminated_title_uses_only_first_line():
    issue = {
        "representative_title": "",
        "articles": [
            {
                "title": (
                    "바리톤 김태한, 퀸 엘리자베스 콩쿠르 우승"
                    "\n\n(서울=뉴스1) 조재현 기자 = 기사 본문"
                    "\n머니S 뉴스1 제공"
                ),
            },
        ],
    }

    assert (
        generate_event_name(issue)
        == "바리톤 김태한, 퀸 엘리자베스 콩쿠르 우승"
    )


def test_live_header_line_is_skipped():
    issue = {
        "representative_title": "",
        "articles": [
            {
                "title": (
                    "LIVE :\n"
                    "[포토타임] 초여름에 만나는 안성팜랜드 코스모스 활짝"
                ),
            },
        ],
    }

    assert (
        generate_event_name(issue)
        == "초여름에 만나는 안성팜랜드 코스모스 활짝"
    )


def test_editorial_modifiers_are_removed():
    cases = [
        (
            "금감원, ‘주가폭락‘ CFD 관련 키움증권 검사 전격 착수",
            "전격",
        ),
        (
            "‘직원 잔혹 살해‘ 스포츠센터 대표 징역 25년 확정",
            "잔혹",
        ),
        (
            "‘막대기 엽기 살해‘ 스포츠센터 대표 징역 25년 확정",
            "엽기",
        ),
    ]

    for title, modifier in cases:
        issue = {
            "representative_title": "",
            "articles": [
                {"title": title},
            ],
        }

        event_name = generate_event_name(issue)

        assert modifier not in event_name
        assert event_name.strip()


def test_empty_cleaned_titles_fall_back_to_representative_title():
    issue = {
        "representative_title": "실제 대표 이슈 제목",
        "articles": [
            {"title": "LIVE :"},
            {"title": ""},
        ],
    }

    assert generate_event_name(issue) == "실제 대표 이슈 제목"
