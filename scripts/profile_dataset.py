"""profile_dataset — 과거 기사 CSV(| 구분)를 한 번 훑어 언론사·날짜 형식·빈값·보일러플레이트 통계를 낸다.

파일이 수 GB 라 행을 메모리에 남기지 않고 집계만 쌓는다. 결과는 JSON 으로 저장하고
요약을 표준 출력에 찍는다. 이 통계가 docs/ds1/DATASET_PROFILE.md 의 근거다.

    python scripts/profile_dataset.py ../../ssafy_dataset_news_2023.csv -o local/profile_2023.json
"""

import argparse
import collections
import csv
import json
import re
import time

from hannun.preprocess.rules import BROKEN_ENTITY

csv.field_size_limit(10**9)

DATE_SHAPES = (
    ("YYYY-MM-DD HH:MM:SS", re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")),
    ("YYYY-MM-DD HH:MM", re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")),
    ("YYYY.MM.DD HH:MM", re.compile(r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}$")),
    ("YYYY-MM-DD 오전/오후 h:MM:SS", re.compile(r"^\d{4}-\d{2}-\d{2} (오전|오후) \d{1,2}:\d{2}:\d{2}$")),
)
HTML_TAG = re.compile(r"<[^>]{1,40}>")
LEN_BUCKETS = ((0, "0"), (200, "<200"), (500, "<500"), (1000, "<1000"), (3000, "<3000"), (10000, "<10000"))


def date_shape(s):
    for name, rx in DATE_SHAPES:
        if rx.match(s):
            return name
    return "기타"


def len_bucket(n):
    for limit, name in LEN_BUCKETS:
        if n < limit or (limit == 0 and n == 0):
            return name
    return ">=10000"


def profile(path, progress_every=200000):
    started = time.time()
    stats = {
        "rows": 0,
        "bad_field_count": 0,
        "company": collections.Counter(),
        "date_min": {},
        "date_max": {},
        "date_shape": collections.Counter(),
        "date_shape_by_company": collections.defaultdict(collections.Counter),
        "empty": collections.Counter(),
        "category_by_company": collections.defaultdict(collections.Counter),
        "title_broken_entity": 0,
        "article_html_tag": 0,
        "last_line": collections.Counter(),
        "first_line": collections.Counter(),
        "article_len": collections.Counter(),
        "dup_link": 0,
        "dup_title_in_company": 0,
    }
    links, titles = set(), set()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|", quotechar='"')
        header = next(reader)
        for r in reader:
            stats["rows"] += 1
            if len(r) != len(header):
                stats["bad_field_count"] += 1
                continue
            company, title, link, published, category, category_str, reporter, article = r
            stats["company"][company] += 1
            day = published[:10]
            if company not in stats["date_min"] or day < stats["date_min"][company]:
                stats["date_min"][company] = day
            if company not in stats["date_max"] or day > stats["date_max"][company]:
                stats["date_max"][company] = day
            shape = date_shape(published)
            stats["date_shape"][shape] += 1
            stats["date_shape_by_company"][company][shape] += 1
            for name, value in (("category", category), ("category_str", category_str),
                                ("reporter", reporter), ("article", article), ("title", title)):
                if not value.strip():
                    stats["empty"][name] += 1
            stats["category_by_company"][company][category or "(빈값)"] += 1
            if BROKEN_ENTITY.search(title):
                stats["title_broken_entity"] += 1
            if HTML_TAG.search(article):
                stats["article_html_tag"] += 1
            lines = [x.strip() for x in article.split("\n") if x.strip()]
            if lines:
                stats["first_line"][lines[0][:25]] += 1
                stats["last_line"][lines[-1][:60]] += 1
            stats["article_len"][len_bucket(len(article))] += 1
            if link in links:
                stats["dup_link"] += 1
            links.add(link)
            if (company, title) in titles:
                stats["dup_title_in_company"] += 1
            titles.add((company, title))
            if stats["rows"] % progress_every == 0:
                print(f"  {stats['rows']:,} rows  {time.time() - started:.0f}s", flush=True)
    stats["seconds"] = round(time.time() - started)
    stats["company"] = stats["company"].most_common()
    stats["date_shape"] = dict(stats["date_shape"])
    stats["date_shape_by_company"] = {c: dict(v) for c, v in stats["date_shape_by_company"].items()}
    stats["empty"] = dict(stats["empty"])
    stats["category_by_company"] = {c: v.most_common(15) for c, v in stats["category_by_company"].items()}
    stats["last_line"] = stats["last_line"].most_common(40)
    stats["first_line"] = stats["first_line"].most_common(25)
    stats["article_len"] = dict(stats["article_len"])
    return stats


def print_summary(path, stats):
    n = stats["rows"]
    print(f"\n{path}: {n:,} rows, bad field count {stats['bad_field_count']}, {stats['seconds']}s")
    print(f"publishers {len(stats['company'])}, date shapes {stats['date_shape']}")
    print("empty: " + ", ".join(f"{k} {v:,} ({v / n:.1%})" for k, v in stats["empty"].items()))
    print(f"title broken entity {stats['title_broken_entity']:,}, "
          f"article html tag {stats['article_html_tag']:,}, "
          f"dup link {stats['dup_link']:,}, dup title in company {stats['dup_title_in_company']:,}")
    for company, count in stats["company"]:
        shapes = ", ".join(f"{s}:{v:,}" for s, v in stats["date_shape_by_company"][company].items())
        span = f"{stats['date_min'][company]}~{stats['date_max'][company]}"
        print(f"  {company:8s} {count:9,}  {span}  {shapes}")


def main():
    p = argparse.ArgumentParser(description="과거 기사 CSV 프로파일")
    p.add_argument("csv", nargs="+", help="| 구분 CSV 파일")
    p.add_argument("-o", "--output", default="local/dataset_profile.json", help="집계 JSON 저장 위치")
    args = p.parse_args()

    profiles = {}
    for path in args.csv:
        print(f"profiling {path}", flush=True)
        profiles[path] = profile(path)
        print_summary(path, profiles[path])
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=1)
    print(f"\nsaved → {args.output}")


if __name__ == "__main__":
    main()
