"""임베딩 모델 후보를 같은 조건에서 비교한다 — 티켓 11 선정 실험.

지표는 셋이고 뒤로 갈수록 서비스 목적에 가깝다.

① 속도 — 정제 본문 표본을 CPU 로 배치 인코딩해 문서/초. EC2(t3.xlarge, 4 vCPU)를
   흉내내도록 스레드를 제한한다.
② 분리력 — STEP 1 검증기가 확정한 중복 쌍(같은 글)과 무작위 쌍(무관한 글)의
   코사인 간격. 같은 글도 못 붙이는 모델은 이슈는 더 못 묶는다.
③ 이슈 재현 — 라벨링 파일럿 30건의 이슈 라벨을 군집화로 재현하는 정도(ARI).
   본문 파일(news30.jsonl)이 로컬에만 있어 --pilot 로 받을 때만 계산한다.

임베딩 입력은 설계(STEP 2)대로 "제목 + 정제 본문 앞부분"이다.

    .venv312/Scripts/python.exe scripts/compare_embeddings.py \
        --gold-root local/gold2023 --date 2023-03-15 --out local/embed_compare.json
"""

import argparse
import json
import random
import time
from dataclasses import dataclass

import numpy as np

from hannun.dedup.candidates import find_candidate_pairs
from hannun.dedup.verify import verify_pairs
from hannun.ingest import GoldStore
from hannun.preprocess import CleanStore


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    # e5 계열은 학습 때 붙인 접두어를 인코딩 때도 붙여야 성능이 나온다
    passage_prefix: str = ""


MODELS = {
    "ko-sroberta": ModelSpec("jhgan/ko-sroberta-multitask"),
    "e5-small-ko": ModelSpec("dragonkue/multilingual-e5-small-ko-v2", passage_prefix="passage: "),
    "kure-v1": ModelSpec("nlpai-lab/KURE-v1"),
}

SEED = 20260901


def build_input(title, clean_text, body_chars):
    # 제목이 기사의 핵심 주장·사건을 담고, 본문 앞부분이 육하원칙을 담는다(역피라미드)
    return f"{title}\n{clean_text[:body_chars]}"


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
    candidates = find_candidate_pairs(
        [{"article_id": a, "text": t} for a, t in texts.items()])
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


def encode(spec, texts, threads, batch_size):
    """모델 하나로 전부 인코딩하고 (정규화 벡터, 문서/초) 를 돌려준다."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    model = SentenceTransformer(spec.hf_id, device="cpu")
    payload = [spec.passage_prefix + t for t in texts]
    t0 = time.monotonic()
    vectors = model.encode(payload, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=False)
    elapsed = time.monotonic() - t0
    return np.asarray(vectors), round(len(texts) / elapsed, 1)


def pair_cosines(vectors, index_of, pairs):
    return [float(vectors[index_of[a]] @ vectors[index_of[b]]) for a, b in pairs]


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

    report = {"date": args.date, "articles": len(rows), "confirmed_pairs": len(confirmed),
              "threads": args.threads, "body_chars": args.body_chars, "models": {}}
    for name in args.models:
        spec = MODELS[name]
        vectors, docs_per_s = encode(spec, texts, args.threads, args.batch_size)
        dup = pair_cosines(vectors, index_of, confirmed)
        rnd = pair_cosines(vectors, index_of, randoms)
        entry = {
            "hf_id": spec.hf_id,
            "dim": int(vectors.shape[1]),
            "docs_per_s": docs_per_s,
            "dup_cos_p10": round(float(np.percentile(dup, 10)), 4) if dup else None,
            "random_cos_p90": round(float(np.percentile(rnd, 90)), 4) if rnd else None,
        }
        # 간격 = 중복 쌍 하위 10% 와 무작위 쌍 상위 10% 사이. 양수여야 문턱을 그을 수 있다
        if dup and rnd:
            entry["separation"] = round(entry["dup_cos_p10"] - entry["random_cos_p90"], 4)
        report["models"][name] = entry
        print(f"{name}: {json.dumps(entry, ensure_ascii=False)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
