"""csv_to_common_jsonl — 과거 기사 CSV 를 공통 기사 JSONL 로 바꿔 로컬에 둔다. 개발용.

실제 변환은 DE 몫이다(docs/contracts/de-to-ds-article-json.md). 이 스크립트는 DE 변환이
나오기 전에 우리 파이프라인을 실데이터로 돌려 보기 위한 것이고, 변환 규칙은 DE 스키마
(de/schemas/article_v1.json)를 따른다. URL 정규화는 DE 와 합의된 규칙(hannun.ingest.schema)이다.
언론사별로 고정 seed 표본을 뽑아 한 번만 훑는다. 결과에는 기사 원문이 들어가므로
local/ 아래에만 쓰고 커밋하지 않는다.

    python scripts/csv_to_common_jsonl.py ../../ssafy_dataset_news_2023.csv \
        --per-publisher 100 -o local/dataset_2023_sample.jsonl
"""

import argparse
import collections
import csv
import json
import random
import re
from datetime import datetime, timedelta, timezone

from hannun.ingest.schema import expected_article_id

csv.field_size_limit(10**9)

KST = timezone(timedelta(hours=9))

# 데이터셋의 company 표기 → publisher_id 제안. "한겨례" 는 데이터셋 오타 그대로 받는다.
# DE 가 확정하면 그쪽 표를 따르고 이 표는 지운다.
PUBLISHER_IDS = {
    "KBS뉴스": "kbs", "뉴스1": "news1", "파이낸셜뉴스": "fnnews", "서울경제": "sedaily", "머니S": "moneys",
    "동아일보": "donga", "중앙일보": "joongang", "서울신문": "seoul", "SBS뉴스": "sbs", "한겨례": "hani",
    "세계일보": "segye", "연합뉴스": "yonhap", "한국경제TV": "wowtv", "데일리안": "dailian",
    "오마이뉴스": "ohmynews", "국민일보": "kmib", "매일신문": "imaeil", "디지털타임": "dt",
    "문화일보": "munhwa", "더팩트": "tf", "조세일보": "joseilbo", "MBN": "mbn", "TV조선": "tvchosun",
    "JTBC": "jtbc", "시사저널": "sisajournal", "아시아경제": "asiae", "블로터": "bloter",
    "경기일보": "kyeonggi", "코메디닷컴": "kormedi", "미디어오늘": "mediatoday", "여성신문": "womennews",
    "비즈니스워치": "bizwatch", "국제신문": "kookje", "디지털데일리": "ddaily",
}
AMPM = re.compile(r"^(\d{4}-\d{2}-\d{2}) (오전|오후) (\d{1,2}):(\d{2}):(\d{2})$")
DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M")


def parse_published(text):
    """4가지 표기를 KST 로 읽어 UTC ISO 문자열로. 못 읽으면 None."""
    s = text.strip()
    m = AMPM.match(s)
    if m:
        day, ampm, hour, minute, second = m.groups()
        hour = int(hour) % 12 + (12 if ampm == "오후" else 0)
        s = f"{day} {hour:02d}:{minute}:{second}"
    for fmt in DATE_FORMATS:
        try:
            local = datetime.strptime(s, fmt).replace(tzinfo=KST)
        except ValueError:
            continue
        return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def to_common(row, converted_at):
    company, title, link, published, category, category_str, reporter, article = row
    publisher_id = PUBLISHER_IDS.get(company)
    published_at = parse_published(published)
    if publisher_id is None or published_at is None or not article.strip() or not title.strip():
        return None
    return {
        "schema_version": "1.0",
        "article_id": expected_article_id(publisher_id, link),
        "publisher_id": publisher_id,
        "company": company,
        "source_type": "DATASET",
        "link": link,
        "title": title,
        "article": article,
        "reporter": reporter.strip() or None,
        # 개편된 계약(2026-08-31): category 는 코드 enum 이 아니라 언론사 원문 문자열, 빈값만 OTHER
        "category": category.strip() or "OTHER",
        "category_str": category_str.strip() or None,
        "thumbnail_url": None,
        "language": "ko",
        "published": published_at,
        "crawled_at": converted_at,
    }


def sample_rows(path, per_publisher, seed, publishers):
    """언론사별 저수지 표본. 한 번 훑으면서 각 언론사에서 per_publisher 건을 균등 확률로 남긴다."""
    rng = random.Random(seed)
    seen = collections.Counter()
    kept = collections.defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|", quotechar='"')
        next(reader)
        for row in reader:
            if len(row) != 8:
                continue
            company = row[0]
            if publishers and company not in publishers:
                continue
            seen[company] += 1
            bucket = kept[company]
            if len(bucket) < per_publisher:
                bucket.append(row)
            else:
                j = rng.randrange(seen[company])
                if j < per_publisher:
                    bucket[j] = row
    return kept, seen


def main():
    p = argparse.ArgumentParser(description="과거 기사 CSV → 공통 기사 JSONL (개발용 표본)")
    p.add_argument("csv", nargs="+")
    p.add_argument("-o", "--output", default="local/dataset_sample.jsonl")
    p.add_argument("--per-publisher", type=int, default=100, help="언론사별 표본 수")
    p.add_argument("--publishers", nargs="*", default=None, help="이 언론사만 (데이터셋 표기)")
    p.add_argument("--seed", type=int, default=20260828)
    args = p.parse_args()

    converted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written, dropped = 0, collections.Counter()
    with open(args.output, "w", encoding="utf-8") as out:
        for path in args.csv:
            kept, seen = sample_rows(path, args.per_publisher, args.seed, args.publishers)
            for company in sorted(kept):
                for row in kept[company]:
                    record = to_common(row, converted_at)
                    if record is None:
                        dropped[company] += 1
                        continue
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
            print(f"{path}: {len(seen)} publishers, {sum(seen.values()):,} rows scanned")
    print(f"written {written:,} → {args.output}")
    if dropped:
        print("dropped (no publisher_id / bad date / empty body or title):", dict(dropped))


if __name__ == "__main__":
    main()
