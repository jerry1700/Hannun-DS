"""골드셋 라벨링 시트 생성 — 라벨링 주간에 팀이 채울 CSV 두 종을 만든다.

① 군집 시트(clusters_<이름>.csv × 라벨러 수): 하루치 대표 기사 표본. 발행 시각순.
   라벨러는 cluster 칸에 이슈 번호(자유 형식)를 적는다 — 파일럿과 같은 방식.
② 중복 쌍 시트(dedup_pairs.csv): 임베딩 유사도가 높은데 dedup 이 접지 않은 쌍
   (recall 의 "놓쳤을 법한" 후보) + 문턱 아래 띠의 무작위 표본. O/X 를 적는다.

기사 제목이 들어가므로 출력은 local/ 아래로만 — 커밋 금지.

    python scripts/make_goldenset_sheets.py --gold-root local/gold2023 --date 2023-01-02 \
        --labelers 6 --out-dir local/goldenset_sheets
"""

import argparse
from pathlib import Path

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
    p.add_argument("--out-dir", default="local/goldenset_sheets")
    args = p.parse_args()

    import numpy as np
    import pandas as pd

    from hannun.dedup import DedupStore
    from hannun.embedding import EmbeddingStore
    from hannun.ingest import GoldStore
    from hannun.preprocess import CleanStore

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

    # ① 군집 시트 — 대표 기사 표본, 발행 시각순 (라벨러가 흐름을 따라 읽게)
    ids = list(emb.article_id)
    sample = list(rng.choice(ids, size=min(args.cluster_articles, len(ids)), replace=False))
    rows = pd.DataFrame({
        "article_id": sample,
        "published_kst": [pd.Timestamp(gold.loc[a].published_at).tz_convert("Asia/Seoul")
                          .strftime("%m-%d %H:%M") for a in sample],
        "publisher": [gold.loc[a].publisher_id for a in sample],
        "title": [str(gold.loc[a].title).split("\n")[0][:90] for a in sample],
        "body_head": [body_head(clean.content_clean.get(a)) for a in sample],
        "cluster": "",
    }).sort_values("published_kst")
    for i in range(1, args.labelers + 1):
        rows.to_csv(out_dir / f"clusters_labeler{i}.csv", index=False, encoding="utf-8-sig")

    # ② 중복 쌍 시트 — 대표끼리 유사도 상위인데 접히지 않은 쌍 (recall 후보)
    vectors = np.array(emb.vector.tolist(), dtype="float32")
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, 0.0)
    dd = DedupStore(args.gold_root).read(args.date, args.date, columns=["article_id", "duplicate_of"])
    # 결측 duplicate_of 는 pandas 에서 NaN — is None 검사는 뚫린다
    root_of = {a: (a if pd.isna(r) else r) for a, r in zip(dd.article_id, dd.duplicate_of)}

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

    print(f"군집 시트: 기사 {len(rows)}건 × 라벨러 {args.labelers}부")
    print(f"중복 쌍 시트: 상위 띠(≥{args.pair_high}) {len(high)}쌍 + 띠({args.pair_band}~) {len(band)}쌍")
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
