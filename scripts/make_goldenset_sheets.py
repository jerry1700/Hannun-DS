"""골드셋 라벨링 시트 생성 — 라벨링 주간에 팀이 채울 CSV 를 만든다 (티켓 104).

군집 시트(clusters_labeler<n>.csv)는 대표 기사의 연속 시간대 슬라이스다 — 무작위 표본은
같은 이슈 짝이 사라져 채점 신호가 약해진다. 비슷한 기사가 이웃하도록 파이프라인 군집
순서로 정렬하되 번호는 숨긴다(정박 편향 방지). 라벨러는 cluster 칸에 이슈 번호를 적는다.
중복 쌍 시트(dedup_pairs.csv)는 임베딩 유사도가 높은데 dedup 이 접지 않은 쌍(recall 의
"놓쳤을 법한" 후보)과 문턱 아래 띠의 무작위 표본이고, 군집 쌍 시트(issue_pairs.csv)는
"두 기사가 같은 이슈인가"를 같은이슈:경계:무작위 = 2:1:1 로 섞은 것이다. 둘 다 O/X 를 적는다.

기사 제목이 들어가므로 출력은 local/ 아래로만 — 커밋 금지.

    python scripts/make_goldenset_sheets.py --gold-root local/gold2023 --date 2023-01-02 \
        --labelers 6 --out-dir local/goldenset_sheets
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hannun.dedup import DedupStore
from hannun.embedding import EmbeddingStore
from hannun.ingest import GoldStore
from hannun.preprocess import CleanStore
from hannun.quality import QualityStore
from hannun.quality.goldenset import representative_of

SEED = 20260908


def body_head(text, n=160):
    return (text or "")[:n].replace("\n", " ")


def main():
    p = argparse.ArgumentParser(description="골드셋 라벨링 시트 생성 (티켓 104)")
    p.add_argument("--gold-root", default="local/gold2023")
    p.add_argument("--date", required=True, help="표본 하루 (UTC 발행일)")
    p.add_argument("--cluster-articles", type=int, default=300, help="군집 시트 기사 수")
    p.add_argument("--labelers", type=int, default=6, help="군집 시트 복사본 수")
    p.add_argument("--pair-high", type=float, default=0.90, help="중복 후보 상위 띠 문턱")
    p.add_argument("--pair-band", type=float, default=0.85, help="중복 후보 아래 띠 하한")
    p.add_argument("--pair-count", type=int, default=300, help="중복 쌍 수(상위 띠 우선)")
    p.add_argument("--issue-pairs", type=int, default=400,
                   help="군집 쌍 판정 시트 크기 (같은이슈:경계:무작위 = 2:1:1)")
    p.add_argument("--out-dir", default="local/goldenset_sheets")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    emb = EmbeddingStore(args.gold_root).read(args.date, args.date)
    gold = GoldStore(args.gold_root).read_table(
        args.date, args.date, columns=["article_id", "title", "published_at", "publisher_id"]
    ).to_pandas().set_index("article_id")
    clean = CleanStore(args.gold_root).read_table(
        args.date, args.date, columns=["article_id", "content_clean"]
    ).to_pandas().set_index("article_id")
    quality = QualityStore(args.gold_root).read(args.date, args.date)
    label_of = dict(zip(quality.article_id, quality.issue_local)) if not quality.empty else {}

    # 군집 시트 — 표본은 무작위가 아니라 연속 시간대 슬라이스. 이슈는 시간적으로 몰려 터지므로
    # 연속 구간을 통째로 뜨면 이슈 동료가 자연히 함께 들어온다. 기계 번호를 노출하면 라벨이
    # 그걸 베껴 채점이 오염되므로 정렬에만 쓴다
    ordered = sorted(emb.article_id, key=lambda a: gold.loc[a].published_at)
    size = min(args.cluster_articles, len(ordered))
    start_at = int(rng.integers(0, len(ordered) - size + 1)) if len(ordered) > size else 0
    sample = ordered[start_at:start_at + size]

    rows = pd.DataFrame({
        "article_id": sample,
        "published_kst": [pd.Timestamp(gold.loc[a].published_at).tz_convert("Asia/Seoul")
                          .strftime("%m-%d %H:%M") for a in sample],
        "publisher": [gold.loc[a].publisher_id for a in sample],
        "title": [str(gold.loc[a].title).split("\n")[0][:90] for a in sample],
        "body_head": [body_head(clean.content_clean.get(a)) for a in sample],
        "cluster": "",
        "_group": [int(label_of.get(a, -1)) for a in sample],
    })
    # 노이즈(-1)는 맨 뒤로, 그 안에서는 발행순 — 정렬 힌트일 뿐 라벨은 사람이 단다
    rows["_noise"] = rows["_group"] < 0
    rows = (rows.sort_values(["_noise", "_group", "published_kst"])
            .drop(columns=["_group", "_noise"]))
    for i in range(1, args.labelers + 1):
        rows.to_csv(out_dir / f"clusters_labeler{i}.csv", index=False, encoding="utf-8-sig")

    # 중복 쌍 시트 — 대표끼리 유사도 상위인데 접히지 않은 쌍 (recall 후보). 쌍 후보는 하루 전체에서
    ids = list(emb.article_id)
    vectors = np.array(emb.vector.tolist(), dtype="float32")
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, 0.0)
    dedup_frame = DedupStore(args.gold_root).read(args.date, args.date,
                                                  columns=["article_id", "duplicate_of"])
    root_of = representative_of(dedup_frame)

    upper = np.triu_indices_from(sims, k=1)
    values = sims[upper]
    order = np.argsort(-values)
    high, band = [], []
    for k in order:
        sim = float(values[k])
        if sim < args.pair_band or len(high) >= args.pair_count:
            break
        a, b = ids[upper[0][k]], ids[upper[1][k]]
        if root_of.get(a) == root_of.get(b):
            continue  # 이미 같은 그룹 — recall 후보 아님
        (high if sim >= args.pair_high else band).append((a, b, sim))
    band = [band[i] for i in rng.permutation(len(band))[:max(args.pair_count - len(high), 0)]]

    pairs = high + band
    sheet = pd.DataFrame({
        "sim": [round(s, 4) for _, _, s in pairs],
        "publisher_a": [gold.loc[a].publisher_id for a, _, _ in pairs],
        "title_a": [str(gold.loc[a].title).split("\n")[0][:70] for a, _, _ in pairs],
        "body_a": [body_head(clean.content_clean.get(a), 120) for a, _, _ in pairs],
        "publisher_b": [gold.loc[b].publisher_id for _, b, _ in pairs],
        "title_b": [str(gold.loc[b].title).split("\n")[0][:70] for _, b, _ in pairs],
        "body_b": [body_head(clean.content_clean.get(b), 120) for _, b, _ in pairs],
        "판정(O=같은기사,X=다른기사)": "",
        "article_id_a": [a for a, _, _ in pairs],
        "article_id_b": [b for _, b, _ in pairs],
    })
    sheet.to_csv(out_dir / "dedup_pairs.csv", index=False, encoding="utf-8-sig")

    # 군집 쌍 판정 시트 — 시트 방식보다 판단이 가볍고 분담이 쉬워 본 채점의 기본 방식. 세 층을
    # 섞되 층 정보는 시트에 넣지 않는다(라벨러가 층을 알면 편향) — 채점기가 파이프라인 상태에서 재유도
    structured_issues = set(quality[quality.structured].issue_local) if not quality.empty else set()

    def eligible(article_id):
        label = label_of.get(article_id, -1)
        return label >= 0 and label not in structured_issues

    members = {}
    for a in ids:
        if eligible(a):
            members.setdefault(label_of[a], []).append(a)
    multi = [m for m in members.values() if len(m) >= 2]

    n_same, n_boundary, n_random = args.issue_pairs // 2, args.issue_pairs // 4, args.issue_pairs // 4
    seen, issue_pairs = set(), []

    def add_pair(a, b):
        key = (a, b) if a < b else (b, a)
        if a != b and key not in seen:
            seen.add(key)
            issue_pairs.append(key)
            return True
        return False

    attempts = 0
    while multi and len(issue_pairs) < n_same and attempts < n_same * 50:
        attempts += 1
        group = multi[int(rng.integers(len(multi)))]
        a, b = rng.choice(group, size=2, replace=False)
        add_pair(a, b)

    band_idx = np.where((values >= 0.75) & (values < 0.95))[0]
    rng.shuffle(band_idx)
    added = 0
    for k in band_idx:
        if added >= n_boundary:
            break
        a, b = ids[upper[0][k]], ids[upper[1][k]]
        if eligible(a) and eligible(b) and label_of[a] != label_of[b] and add_pair(a, b):
            added += 1

    added, attempts = 0, 0
    while added < n_random and attempts < n_random * 50:
        attempts += 1
        a, b = (ids[int(rng.integers(len(ids)))] for _ in range(2))
        if eligible(a) and eligible(b) and label_of[a] != label_of[b] and add_pair(a, b):
            added += 1

    order2 = rng.permutation(len(issue_pairs))
    issue_pairs = [issue_pairs[i] for i in order2]
    pair_sheet = pd.DataFrame({
        "publisher_a": [gold.loc[a].publisher_id for a, _ in issue_pairs],
        "title_a": [str(gold.loc[a].title).split("\n")[0][:70] for a, _ in issue_pairs],
        "body_a": [body_head(clean.content_clean.get(a), 120) for a, _ in issue_pairs],
        "publisher_b": [gold.loc[b].publisher_id for _, b in issue_pairs],
        "title_b": [str(gold.loc[b].title).split("\n")[0][:70] for _, b in issue_pairs],
        "body_b": [body_head(clean.content_clean.get(b), 120) for _, b in issue_pairs],
        "판정(O=같은이슈,X=다른이슈)": "",
        "article_id_a": [a for a, _ in issue_pairs],
        "article_id_b": [b for _, b in issue_pairs],
    })
    pair_sheet.to_csv(out_dir / "issue_pairs.csv", index=False, encoding="utf-8-sig")

    print(f"군집 시트: 기사 {len(rows)}건 × 라벨러 {args.labelers}부")
    print(f"중복 쌍 시트: 상위 띠(≥{args.pair_high}) {len(high)}쌍 + 띠({args.pair_band}~) {len(band)}쌍")
    print(f"군집 쌍 시트: {len(issue_pairs)}쌍 (같은이슈 {n_same}·경계 {n_boundary}·무작위 {n_random}, 순서 섞음)")
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
