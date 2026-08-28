"""make_sample_data — 테스트용 공통 기사 JSON 샘플을 samples/ 에 만든다.

정상 건과 리젝트되어야 하는 건을 섞는다. article_id 는 BE 규칙대로 실제 계산한다.
"""

import json
from pathlib import Path

from hannun.ingest.schema import expected_article_id

OUT = Path(__file__).resolve().parents[1] / "samples" / "articles_sample.jsonl"


def article(publisher_id, publisher_name, url, **overrides):
    base = {
        "schema_version": "1.0",
        "article_id": expected_article_id(publisher_id, url),
        "publisher_id": publisher_id,
        "publisher_name": publisher_name,
        "source_type": "RSS",
        "url": url,
        "title": "정부, 새로운 청년 정책 발표",
        "content": "정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다.",
        "author": "홍길동",
        "category": "POLITICS",
        "category_str": "정치>행정",
        "thumbnail_url": "https://news.example.com/images/123.jpg",
        "language": "ko",
        "published_at": "2026-08-20T05:30:00Z",
    }
    base.update(overrides)
    return base


def main():
    # 본문 끝의 사진 출처와 저작권 문구는 preprocess 가 지워야 하는 것. 실데이터에서 가장 흔한 꼴이다.
    a1 = article(
        "yonhap", "연합뉴스", "https://www.yna.co.kr/view/AKR20260820000100001",
        content="정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다."
                "\n\n(사진=연합뉴스)\n<저작권자(c) 연합뉴스, 무단 전재-재배포 금지>",
    )
    lines = []

    # 1. 정상 (RSS)
    lines.append(json.dumps(a1, ensure_ascii=False))

    # 2. 정상 (과거 DATASET, 선택 필드 null)
    lines.append(json.dumps(article(
        "hani", "한겨레", "https://www.hani.co.kr/arti/politics/2024/03/15/1001.html",
        source_type="DATASET", author=None, thumbnail_url=None, category_str=None,
        title="지방선거 앞두고 공천 갈등 격화",
        content="여야가 지방선거 공천을 앞두고 내부 갈등을 겪고 있다.",
        published_at="2024-03-15T00:00:00Z",
    ), ensure_ascii=False))

    # 3. 1번과 같은 기사 → 배치 안 중복으로 1건만 남아야 함
    lines.append(json.dumps(a1, ensure_ascii=False))

    # 4. article_id 가 BE 해시 규칙과 다름 → 적재되되 article_id_verified=False
    lines.append(json.dumps(article(
        "chosun", "조선일보", "https://www.chosun.com/politics/2026/08/20/EJEZYM2ZS/",
        article_id="sha256:" + "0" * 64,
        title="청년 정책 발표에 여야 엇갈린 반응",
        content="정부의 청년 정책 발표에 여당은 환영, 야당은 재정 부담을 우려했다.",
        published_at="2026-08-20T06:10:00Z",
    ), ensure_ascii=False))

    # 5. 본문 빈 문자열 → 리젝트
    lines.append(json.dumps(article(
        "newsis", "뉴시스", "https://newsis.com/view/NISX20260820_0001",
        content="",
        published_at="2026-08-20T05:45:00Z",
    ), ensure_ascii=False))

    # 6. published_at 에 타임존 없음 → 리젝트
    lines.append(json.dumps(article(
        "news1", "뉴스1", "https://www.news1.kr/articles/5001",
        published_at="2026-08-20T05:30:00",
    ), ensure_ascii=False))

    # 7. author 빈 문자열 → null 로 정규화되어 적재. 본문은 방송 원고 꼴 — 마커·서명·제작진이 붙는다
    lines.append(json.dumps(article(
        "kbs", "KBS", "https://news.kbs.co.kr/news/view.do?ncd=8001",
        author="",
        title="[속보] 정부 청년 정책 발표",
        content="[앵커] 정부가 청년 정책을 발표했습니다. 김석 기자가 보도합니다. [리포트] 청년 주거 안정이 핵심입니다."
                "KBS 뉴스 김석입니다. 촬영기자:김종우/영상편집:여동용",
        published_at="2026-08-20T05:31:00Z",
    ), ensure_ascii=False))

    # 8. 깨진 JSON 줄 → 이 줄만 리젝트, 파일은 계속 읽어야 함
    lines.append('{"schema_version": "1.0", "article_id": "sha256:abc", "publisher_id": ')

    # 9. +09:00 → UTC 로 바뀌어 2026-08-20 파티션에 들어가야 함
    lines.append(json.dumps(article(
        "mbc", "MBC", "https://imnews.imbc.com/news/2026/politics/article/7001.html",
        title="청년 정책, 현장 반응은",
        content="청년들은 정책의 실효성에 반신반의하는 모습이었다.",
        published_at="2026-08-21T02:00:00+09:00",
    ), ensure_ascii=False))

    # 10. 정의되지 않은 source_type → 리젝트
    lines.append(json.dumps(article(
        "sbs", "SBS", "https://news.sbs.co.kr/news/endPage.do?news_id=N9001",
        source_type="API",
        published_at="2026-08-20T07:00:00Z",
    ), ensure_ascii=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines → {OUT}")


if __name__ == "__main__":
    main()
