"""DS1 결과를 DS2로 분석해 BE 연동용 JSONL로 내보낸다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hannun.clustering.registry import RegistryStore
from hannun.enrichment.pipeline import enrich_issue
from hannun.feed.store import SummaryStore
from hannun.ingest.gold import GoldStore
from hannun.preprocess.store import CleanStore
from hannun.quality.store import QualityStore


def text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_previous(target: Path) -> dict:
    """이전 실행이 남긴 오늘 파일을 clusterId → (원문 줄, 링크 집합, 대표 제목, 카테고리, 빈 요약 여부)
    로 읽는다. 구성원·대표·카테고리가 그대로인 이슈는 DS2 분석 결과도 같으므로(입력이 같으면
    출력이 같다) 줄을 재사용한다 — 15분 배정 실행에서 바뀌는 이슈는 수십 개뿐인데 전부 다시
    분석하던 232초(실행의 60%)를 줄이는 지점 (S15P21E105-123)."""
    previous = {}
    if not target.exists():
        return previous
    with target.open(encoding="utf-8") as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            links = frozenset(text(a.get("link")) for a in message.get("articles", []))
            previous[message.get("clusterId")] = (
                line, links, message.get("factSummary"), message.get("category"),
                not message.get("commonFactsBriefing"),
            )
    return previous


def unchanged(entry, links: frozenset, title: str, category: str) -> bool:
    return entry is not None and entry[1] == links and entry[2] == title and entry[3] == category


def export(args) -> None:
    # DS1 산출물 읽기
    quality = QualityStore(args.gold_root).read(
        args.start,
        args.end,
    )
    summary = SummaryStore(args.gold_root).read(
        args.window,
    )
    registry = RegistryStore(
        args.gold_root
    ).read_window(
        args.window,
    ).to_pandas()
    gold = GoldStore(args.gold_root).read(
        args.start,
        args.end,
        columns=[
            "article_id",
            "publisher_name",
            "title",
            "url",
            "category",
            "published_at",
        ],
    )
    clean = CleanStore(args.gold_root).read(
        args.start,
        args.end,
        columns=[
            "article_id",
            "content_clean",
        ],
    )

    # stance 분석 대상 기사만 사용
    quality = quality[
        (quality["issue_local"] >= 0)
        & (~quality["structured"].fillna(True).astype(bool))
    ][
        ["article_id", "issue_local"]
    ]

    quality = quality[
        ["article_id"]
    ].merge(
        registry[
            ["article_id", "issue_id"]
        ],
        on="article_id",
        how="inner",
    )

    articles = (
        quality
        .merge(gold, on="article_id", how="inner")
        .merge(clean, on="article_id", how="inner")
    )

    articles = articles[
        articles["content_clean"].notna()
        & (
            articles["content_clean"]
            .astype(str)
            .str.strip()
            != ""
        )
    ]

    target = (
        Path(args.output_root)
        / f"{args.output_day}.jsonl"
    )
    temp = target.with_name(
        f".{target.name}.tmp"
    )

    # dry-run일 때는 파일을 만들지 않는다.
    if args.dry_run:
        output = None
    else:
        if not target.parent.is_dir():
            raise FileNotFoundError(
                target.parent
            )

        # --overwrite 는 일일 체인·Airflow 재시도용 — 임시 파일을 쓴 뒤
        # replace 하므로 교체는 원자적이다 (S15P21E105-122)
        if target.exists() and not args.overwrite:
            raise FileExistsError(
                target
            )

        output = temp.open(
            "w",
            encoding="utf-8",
        )

    previous = load_previous(target) if args.reuse_unchanged else {}

    stats = {
        "summary_issues": len(summary),
        "exported_issues": 0,
        "reused_issues": 0,
        "skipped_no_articles": 0,
        "empty_briefings": 0,
        "articles": 0,
        "target": str(target),
        "dry_run": args.dry_run,
    }

    try:
        for issue in summary.itertuples(index=False):
            rows = articles[
                articles["issue_id"]
                == issue.issue_id
            ].sort_values(
                "published_at",
                kind="stable",
            )

            if rows.empty:
                stats["skipped_no_articles"] += 1
                continue

            representative = gold[
                gold["article_id"]
                == issue.representative
            ]

            if representative.empty:
                raise RuntimeError(
                    f"issue_id={issue.issue_id}: "
                    "대표 기사 없음"
                )

            representative_title = text(
                representative.iloc[0]["title"]
            )
            # 이슈 카테고리는 issue_summary 의 구성 기사 다수결(34)을 우선, 그 컬럼이 없는
            # 옛 창이면 대표 기사 값으로
            category = text(getattr(issue, "category", None)) or text(
                representative.iloc[0]["category"]
            )

            if not representative_title:
                raise RuntimeError(
                    f"issue_id={issue.issue_id}: "
                    "대표 제목 없음"
                )

            if not category:
                raise RuntimeError(
                    f"issue_id={issue.issue_id}: "
                    "category 없음"
                )

            entry = previous.get(int(issue.issue_id))
            links = frozenset(text(row.url) for row in rows.itertuples(index=False))
            if unchanged(entry, links, representative_title, category):
                stats["exported_issues"] += 1
                stats["reused_issues"] += 1
                stats["articles"] += len(rows)
                if entry[4]:
                    stats["empty_briefings"] += 1
                if output is not None:
                    output.write(entry[0] + "\n")
                continue

            # 기존 enrich_issue 입력 형식으로 변환
            issue_data = {
                "issue_cluster_id": str(
                    issue.issue_id
                ),
                "representative_title":
                    representative_title,
                "articles": [
                    {
                        "article_id": str(
                            row.article_id
                        ),
                        "title": text(row.title),
                        "content": text(
                            row.content_clean
                        ),
                        "publisher_name": text(
                            row.publisher_name
                        ),
                    }
                    for row
                    in rows.itertuples(index=False)
                ],
            }

            # 기존 DS2 분석 실행
            enriched = enrich_issue(
                issue_data
            )

            stance_by_id = {
                str(article["article_id"]):
                    article["stance"]
                for article
                in enriched["articles"]
            }

            viewpoint_by_id = {
                str(article["article_id"]):
                    article["viewpoint_group_label"]
                for article
                in enriched["articles"]
            }

            # BE 는 commonFactsBriefing 을 List<String> 으로 받는다. generate_fact_summary
            # 가 공통 사실 문장을 리스트로 주므로 합치지 않고 그대로 넘긴다 (S15P21E105-69)
            briefing = [
                text(sentence)
                for sentence
                in enriched["fact_summary"]
                if text(sentence)
            ]

            clustered_at = pd.Timestamp(
                issue.summarized_at
            )

            if clustered_at.tzinfo is None:
                clustered_at = (
                    clustered_at.tz_localize("UTC")
                )

            # BE 전달 형식
            message = {
                "clusterId": int(
                    issue.issue_id
                ),
                "category": category,
                "factSummary":
                    representative_title,
                "commonFactsBriefing":
                    briefing,
                "clusteredAt": (
                    clustered_at
                    .tz_convert("Asia/Seoul")
                    .isoformat()
                ),
                "articles": [
                    {
                        "link": text(row.url),
                        "viewpointGroupLabel":
                            viewpoint_by_id[
                                str(row.article_id)
                            ],
                        "stance":
                            stance_by_id[
                                str(row.article_id)
                            ],
                    }
                    for row
                    in rows.itertuples(index=False)
                ],
            }

            stats["exported_issues"] += 1
            stats["articles"] += len(
                message["articles"]
            )

            if not briefing:
                stats["empty_briefings"] += 1

            if output is not None:
                output.write(
                    json.dumps(
                        message,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if output is not None:
            output.close()
            output = None
            temp.replace(target)

    except Exception:
        if output is not None:
            output.close()

        if temp.exists():
            temp.unlink()

        raise

    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        required=True,
    )
    parser.add_argument(
        "--end",
        required=True,
    )
    parser.add_argument(
        "--window",
        required=True,
    )
    parser.add_argument(
        "--output-day",
        required=True,
    )
    parser.add_argument(
        "--gold-root",
        default="/home/ubuntu/gold",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/ds_output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--reuse-unchanged",
        action="store_true",
    )

    export(
        parser.parse_args()
    )


if __name__ == "__main__":
    main()
