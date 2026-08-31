from datetime import datetime, timezone

from hannun.dedup import find_exact_duplicates

BODY = "정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다."


def row(article_id, content=BODY, published="2026-08-20T05:30:00", publisher_id="yonhap"):
    return {
        "article_id": article_id,
        "content": content,
        "published_at": datetime.fromisoformat(published).replace(tzinfo=timezone.utc),
        "publisher_id": publisher_id,
    }


def test_identical_content_is_grouped_under_earliest_article():
    rows = [
        row("a2", published="2026-08-20T06:00:00", publisher_id="kbs"),
        row("a1", published="2026-08-20T05:30:00"),
        row("a3", published="2026-08-20T07:00:00", publisher_id="sbs"),
    ]
    result = find_exact_duplicates(rows)

    assert result.groups == 1
    assert result.duplicate_of == {"a2": "a1", "a3": "a1"}


def test_different_content_is_not_grouped():
    rows = [row("a1"), row("b1", content=BODY + " 한 글자만 달라도 완전 중복이 아니다.")]
    assert find_exact_duplicates(rows).duplicate_of == {}


def test_whitespace_difference_is_not_exact_duplicate():
    rows = [row("a1"), row("b1", content=BODY + " ")]
    assert find_exact_duplicates(rows).duplicate_of == {}


def test_multiple_groups_counted_separately():
    other = "다른 사건 기사 본문이다."
    rows = [row("a1"), row("a2"), row("b1", content=other), row("b2", content=other),
            row("c1", content="단독 기사")]
    result = find_exact_duplicates(rows)

    assert result.groups == 2
    assert set(result.duplicate_of) == {"a2", "b2"}


def test_same_published_at_falls_back_to_article_id_for_determinism():
    rows = [row("z9"), row("a1")]
    assert find_exact_duplicates(rows).duplicate_of == {"z9": "a1"}
