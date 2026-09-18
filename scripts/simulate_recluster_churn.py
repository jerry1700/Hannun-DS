"""재군집 출렁임 시뮬레이션 — 창에 기사가 시간순으로 들어오는 상황을 재현해 재학습과 고정 지도를 비교한다 (티켓 128).

창의 마지막 날 기사를 몇 시간 단위로 잘라 단계마다 스크래치 루트에 다시 적재하고, 운영과 같은
cluster → succeed 를 돌려 승계 통계(created·splits·retired)를 찍는다. 마지막 단계에는 quality 까지
돌려 구제 뒤 노이즈를, 쌍 라벨을 주면 골드셋 채점(hannun.quality.goldenset)까지 낸다.
운영 테이블은 읽기만 한다.

    python scripts/simulate_recluster_churn.py --gold-root local/gold2023 \
        --start 2023-01-01 --end 2023-01-02 --out-root local/sim_churn --variant fixed
    python scripts/simulate_recluster_churn.py --gold-root ~/gold --start 2026-09-07 --end 2026-09-08 \
        --out-root ~/sim_churn --variant fixed --labels ~/issue_pairs_labeled_merged.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hannun.clustering import ClusterConfig, IssueStore, MapStore, RegistryStore, cluster, succeed
from hannun.embedding import EmbeddingStore
from hannun.ingest import GoldStore
from hannun.quality import QualityStore, qualify
from hannun.quality.goldenset import load_pairs, score_pairs


def main():
    p = argparse.ArgumentParser(description="재군집 출렁임 시뮬레이션 (티켓 128)")
    p.add_argument("--gold-root", required=True, help="embedding·gold 를 읽을 운영 루트 (읽기 전용)")
    p.add_argument("--start", required=True, help="창 시작 (UTC)")
    p.add_argument("--end", required=True, help="창 끝 (UTC) — 이 날의 기사를 시간 단위로 잘라 넣는다")
    p.add_argument("--out-root", required=True, help="변형별 스크래치 루트")
    p.add_argument("--variant", choices=["refit", "fixed"], default="fixed",
                   help="refit=매 단계 UMAP 재학습, fixed=첫 단계 지도 고정 + 새 이슈 탐지")
    p.add_argument("--first-hours", type=int, default=6, help="첫 단계에 넣을 마지막 날 시각(시)")
    p.add_argument("--step-hours", type=int, default=3, help="단계 간격(시)")
    p.add_argument("--new-issue-sim", type=float, default=ClusterConfig.new_issue_sim)
    p.add_argument("--labels", default=None, help="이슈 쌍 라벨 CSV — 주면 마지막 단계를 골드셋으로 채점")
    args = p.parse_args()

    config = ClusterConfig(new_issue_sim=args.new_issue_sim)
    emb = EmbeddingStore(args.gold_root).read_table(args.start, args.end).to_pandas()
    gold = GoldStore(args.gold_root).read(args.start, args.end, columns=["article_id", "published_at"])
    emb = emb.merge(gold, on="article_id").sort_values("published_at").reset_index(drop=True)
    day_end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    cuts = [pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(hours=h)
            for h in range(args.first_hours, 24, args.step_hours)] + [day_end]

    root = Path(args.out_root) / args.variant
    embeddings, issues, registry = EmbeddingStore(root), IssueStore(root), RegistryStore(root)
    map_store = MapStore(root) if args.variant == "fixed" else None
    print(f"{args.variant}: window {args.start}~{args.end} rows={len(emb)} steps={len(cuts)}", flush=True)

    log = []
    for step, until in enumerate(cuts):
        arrived = emb[emb.published_at < until]
        for date_str, part in arrived.groupby("published_date"):
            embeddings.write_partition(date_str, part.drop(columns=["published_at"]).to_dict("records"))
        t0 = time.perf_counter()
        cstats = cluster(embeddings, issues, config, start_date=args.start, end_date=args.end,
                         map_store=map_store)
        sstats = succeed(issues, registry, start_date=args.start, end_date=args.end)
        log.append({"step": step, "until": until.isoformat(), "rows": len(arrived),
                    "issues": cstats.issues, "noise": cstats.noise, "new_issues": cstats.new_issues,
                    "created": sstats.created, "splits": sstats.splits, "retired": sstats.retired,
                    "secs": round(time.perf_counter() - t0)})
        print(json.dumps(log[-1], ensure_ascii=False), flush=True)

    later = log[1:]
    summary = {
        "variant": args.variant,
        "created_per_step": round(sum(x["created"] for x in later) / len(later), 1),
        "retired_per_step": round(sum(x["retired"] for x in later) / len(later), 1),
        "splits_per_step": round(sum(x["splits"] for x in later) / len(later), 1),
    }
    qstats = qualify(issues, embeddings, QualityStore(root), start_date=args.start, end_date=args.end)
    summary.update(final_issues=qstats.issues, final_noise_before=qstats.noise_before,
                   final_rescued=qstats.rescued, final_noise_after=qstats.noise_after)
    if args.labels:
        quality = QualityStore(root).read(args.start, args.end)
        label_of = dict(zip(quality.article_id, quality.issue_local))
        vector_of = {a: np.array(v, dtype="float32") for a, v in zip(emb.article_id, emb.vector)}
        summary["goldenset"] = score_pairs(load_pairs(args.labels), label_of, vector_of)
    report = json.dumps({"steps": log, "summary": summary}, ensure_ascii=False, indent=1)
    (root / "summary.json").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
