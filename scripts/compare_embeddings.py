"""임베딩 모델 후보를 같은 조건에서 비교한다 — 티켓 11 선정 실험.

지표는 셋이고 뒤로 갈수록 서비스 목적에 가깝다. 속도는 정제 본문 표본을 CPU 로 배치
인코딩한 문서/초(EC2 t3.xlarge 를 흉내내도록 스레드를 제한한다). 분리력은 STEP 1 검증기가
확정한 중복 쌍(같은 글)과 무작위 쌍(무관한 글)의 코사인 간격 — 같은 글도 못 붙이는 모델은
이슈는 더 못 묶는다. 이슈 재현은 라벨링 파일럿의 이슈 라벨을 군집화로 재현하는 정도(ARI)로,
본문 파일(news30.jsonl)이 로컬에만 있어 --pilot 로 받을 때만 계산한다.

임베딩 입력은 운영 인코더(hannun.embedding.encoder)와 같은 "제목 + 정제 본문 앞부분"이다.

    .venv312/Scripts/python.exe scripts/compare_embeddings.py \
        --gold-root local/gold2023 --date 2023-03-15 --out local/embed_compare.json
"""

import argparse
import json
import random
from dataclasses import replace

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from hannun.dedup.candidates import find_candidate_pairs
from hannun.dedup.verify import verify_pairs
from hannun.embedding.encoder import EncoderConfig, build_input, encode_texts, load_model
from hannun.ingest import GoldStore
from hannun.preprocess import CleanStore
from hannun.quality.goldenset import consensus_components, load_labels

MODELS = {
    "ko-sroberta": EncoderConfig(model_name="jhgan/ko-sroberta-multitask", passage_prefix=""),
    "e5-small-ko": EncoderConfig(model_name="dragonkue/multilingual-e5-small-ko-v2"),
    "kure-v1": EncoderConfig(model_name="nlpai-lab/KURE-v1", passage_prefix=""),
}

SEED = 20260901


def load_day(gold_root, date_str):
    """하루치 기사에서 (article_id, title, 정제 본문) 을 모은다. 정제 실패분은 뺀다."""
    gold = GoldStore(gold_root).read(columns=["article_id", "title", "published_date"])
    gold = gold[gold.published_date == date_str]
    clean = CleanStore(gold_root).read(columns=["article_id", "content_clean", "published_date"])
    clean = clean[clean.published_date == date_str]
    merged = gold.merge(clean[["article_id", "content_clean"]], on="article_id")
    merged = merged[merged.content_clean.fillna("").str.len() >= 100]
    return merged.reset_index(drop=True)


def duplicate_and_random_pairs(rows, sample_cap):
    """STEP 1 후보→검증을 그대로 돌려 확정 쌍을 얻고, 같은 수의 무작위 쌍을 뽑는다."""
    texts = dict(zip(rows.article_id, rows.content_clean))
    candidates = find_candidate_pairs([{"article_id": a, "text": t} for a, t in texts.items()])
    verified = verify_pairs(texts, candidates.pairs)
    confirmed = sorted(verified.confirmed)[:sample_cap]

    rng = random.Random(SEED)
    ids = sorted(texts)
    randoms, taken = [], set(confirmed)
    while len(randoms) < len(confirmed):
        pair = tuple(sorted(rng.sample(ids, 2)))
        if pair not in taken:
            taken.add(pair)
            randoms.append(pair)
    return confirmed, randoms


def encode(model, config, texts):
    vectors, docs_per_s = encode_texts(model, config, texts)
    return np.asarray(vectors, dtype="float32"), docs_per_s


def pair_cosines(vectors, index_of, pairs):
    return [float(vectors[index_of[a]] @ vectors[index_of[b]]) for a, b in pairs]


def load_pilot(articles_path, labels_path, body_chars, min_agree):
    """파일럿 본문과 합의 정답을 만든다. 합의 방식은 goldenset.consensus_components 와 같다."""
    bodies = {}
    with open(articles_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            bodies[record["article_id"]] = build_input(record["title"], record["body"], body_chars)

    gold = consensus_components(load_labels(labels_path), min_agree)
    ids = sorted(a for a in bodies if a in gold)
    return {"ids": ids, "texts": [bodies[a] for a in ids],
            "gold": [gold[a] for a in ids], "n_issues": len({gold[a] for a in ids})}


def pilot_ari(vectors, gold):
    predicted = AgglomerativeClustering(n_clusters=len(set(gold)), metric="cosine",
                                        linkage="average").fit_predict(vectors)
    return adjusted_rand_score(gold, predicted)


def main():
    p = argparse.ArgumentParser(description="임베딩 모델 후보 비교 (티켓 11)")
    p.add_argument("--gold-root", default="local/gold2023")
    p.add_argument("--date", default="2023-03-15", help="비교에 쓸 하루치 (UTC 발행일)")
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--threads", type=int, default=4, help="EC2 t3.xlarge 의 vCPU 수에 맞춤")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--body-chars", type=int, default=500)
    p.add_argument("--pair-cap", type=int, default=200)
    p.add_argument("--max-articles", type=int, default=0,
                   help="0 이면 하루치 전체. 실측 밀도에서 쌍 생성이 수 분 걸리므로 반복 실험은 줄여서")
    p.add_argument("--pilot", default="", help="news30.jsonl 경로. 주면 이슈 재현(ARI) 을 계산")
    p.add_argument("--pilot-labels", default="docs/ds1/labeling/pilot-2026-08/labels.csv")
    p.add_argument("--pilot-agree", type=int, default=4, help="같은 이슈로 인정할 최소 라벨러 수(6명 중)")
    p.add_argument("--out", default="local/embed_compare.json")
    args = p.parse_args()

    rows = load_day(args.gold_root, args.date)
    if args.max_articles and len(rows) > args.max_articles:
        rows = rows.sample(args.max_articles, random_state=SEED).reset_index(drop=True)
    print(f"{args.date}: 기사 {len(rows)}건")
    confirmed, randoms = duplicate_and_random_pairs(rows, args.pair_cap)
    print(f"확정 중복 쌍 {len(confirmed)} / 무작위 쌍 {len(randoms)}")

    texts = [build_input(t, c, args.body_chars) for t, c in zip(rows.title, rows.content_clean)]
    index_of = {a: i for i, a in enumerate(rows.article_id)}

    pilot = None
    if args.pilot:
        pilot = load_pilot(args.pilot, args.pilot_labels, args.body_chars, args.pilot_agree)
        print(f"파일럿 {len(pilot['ids'])}건 / 합의 이슈 {pilot['n_issues']}개 (기준 {args.pilot_agree}/6)")

    report = {"date": args.date, "articles": len(rows), "confirmed_pairs": len(confirmed),
              "threads": args.threads, "body_chars": args.body_chars,
              "pilot_issues": pilot["n_issues"] if pilot else None, "models": {}}
    for name in args.models:
        config = replace(MODELS[name], threads=args.threads, batch_size=args.batch_size,
                         body_chars=args.body_chars)
        model = load_model(config)
        vectors, docs_per_s = encode(model, config, texts)
        dup = pair_cosines(vectors, index_of, confirmed)
        rnd = pair_cosines(vectors, index_of, randoms)
        entry = {
            "hf_id": config.model_name,
            "dim": int(vectors.shape[1]),
            "docs_per_s": docs_per_s,
            "dup_cos_p10": round(float(np.percentile(dup, 10)), 4) if dup else None,
            "random_cos_p90": round(float(np.percentile(rnd, 90)), 4) if rnd else None,
        }
        # 간격 = 중복 쌍 하위 10% 와 무작위 쌍 상위 10% 사이. 양수여야 문턱을 그을 수 있다
        if dup and rnd:
            entry["separation"] = round(entry["dup_cos_p10"] - entry["random_cos_p90"], 4)
        if pilot:
            pilot_vectors, _ = encode(model, config, pilot["texts"])
            entry["pilot_ari"] = round(pilot_ari(pilot_vectors, pilot["gold"]), 4)
        report["models"][name] = entry
        print(f"{name}: {json.dumps(entry, ensure_ascii=False)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
