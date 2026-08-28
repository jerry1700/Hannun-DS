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

csv.field_size_limit(10**9)

DATE_SHAPES = (
    ("YYYY-MM-DD HH:MM:SS", re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")),
    ("YYYY-MM-DD HH:MM", re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")),
    ("YYYY.MM.DD HH:MM", re.compile(r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}$")),
    ("YYYY-MM-DD 오전/오후 h:MM:SS", re.compile(r"^\d{4}-\d{2}-\d{2} (오전|오후) \d{1,2}:\d{2}:\d{2}$")),
)
# 원문 CSV 에는 '&apos;' 의 '&' 가 떨어진 'apos;' 꼴이 남아 있다.
BROKEN_ENTITY = re.compile(r"(?<!&)\b(apos|quot|amp|lt|gt|nbsp);")
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
    st = {
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
            st["rows"] += 1
            if len(r) != len(header):
                st["bad_field_count"] += 1
                continue
            company, title, link, published, category, category_str, reporter, article = r
            st["company"][company] += 1
            day = published[:10]
            if company not in st["date_min"] or day < st["date_min"][company]:
                st["date_min"][company] = day
            if company not in st["date_max"] or day > st["date_max"][company]:
                st["date_max"][company] = day
            shape = date_shape(published)
            st["date_shape"][shape] += 1
            st["date_shape_by_company"][company][shape] += 1
            for name, value in (("category", category), ("category_str", category_str),
                                ("reporter", reporter), ("article", article), ("title", title)):
                if not value.strip():
                    st["empty"][name] += 1
            st["category_by_company"][company][category or "(빈값)"] += 1
            if BROKEN_ENTITY.search(title):
                st["title_broken_entity"] += 1
            if HTML_TAG.search(article):
                st["article_html_tag"] += 1
            lines = [x.strip() for x in article.split("\n") if x.strip()]
            if lines:
                st["first_line"][lines[0][:25]] += 1
                st["last_line"][lines[-1][:60]] += 1
            st["article_len"][len_bucket(len(article))] += 1
            if link in links:
                st["dup_link"] += 1
            links.add(link)
            if (company, title) in titles:
                st["dup_title_in_company"] += 1
            titles.add((company, title))
            if st["rows"] % progress_every == 0:
                print(f"  {st['rows']:,} rows  {time.time() - started:.0f}s", flush=True)
    st["seconds"] = round(time.time() - started)
    st["company"] = st["company"].most_common()
    st["date_shape"] = dict(st["date_shape"])
    st["date_shape_by_company"] = {c: dict(v) for c, v in st["date_shape_by_company"].items()}
    st["empty"] = dict(st["empty"])
    st["category_by_company"] = {c: v.most_common(15) for c, v in st["category_by_company"].items()}
    st["last_line"] = st["last_line"].most_common(40)
    st["first_line"] = st["first_line"].most_common(25)
    st["article_len"] = dict(st["article_len"])
    return st


def print_summary(path, st):
    n = st["rows"]
    print(f"\n{path}: {n:,} rows, bad field count {st['bad_field_count']}, {st['seconds']}s")
    print(f"publishers {len(st['company'])}, date shapes {st['date_shape']}")
    print("empty: " + ", ".join(f"{k} {v:,} ({v / n:.1%})" for k, v in st["empty"].items()))
    print(f"title broken entity {st['title_broken_entity']:,}, article html tag {st['article_html_tag']:,}, "
          f"dup link {st['dup_link']:,}, dup title in company {st['dup_title_in_company']:,}")
    for company, count in st["company"]:
        shapes = ", ".join(f"{s}:{v:,}" for s, v in st["date_shape_by_company"][company].items())
        print(f"  {company:8s} {count:9,}  {st['date_min'][company]}~{st['date_max'][company]}  {shapes}")


def main():
    p = argparse.ArgumentParser(description="과거 기사 CSV 프로파일")
    p.add_argument("csv", nargs="+", help="| 구분 CSV 파일")
    p.add_argument("-o", "--output", default="local/dataset_profile.json", help="집계 JSON 저장 위치")
    args = p.parse_args()

    result = {}
    for path in args.csv:
        print(f"profiling {path}", flush=True)
        result[path] = profile(path)
        print_summary(path, result[path])
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\nsaved → {args.output}")


if __name__ == "__main__":
    main()
