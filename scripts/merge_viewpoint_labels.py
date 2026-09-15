"""5인 세부견해 라벨링 결과를 합쳐 골든셋을 만든다.

원칙:
- O = 같은 최종 견해
- X = 다른 최종 견해
- ? = 판단 유보
- 5명 중 O 또는 X가 4표 이상이면 gold 확정
- 그 외(3:2, ? 다수 등)는 ambiguous 로 분리

예:
python scripts/merge_viewpoint_labels.py \
  --pairs local/viewpoint_pairs.csv \
  --labels \
    local/viewpoint_pairs_labeler1.csv \
    local/viewpoint_pairs_labeler2.csv \
    local/viewpoint_pairs_labeler3.csv \
    local/viewpoint_pairs_labeler4.csv \
    local/viewpoint_pairs_labeler5.csv \
  --gold-out local/viewpoint_gold.csv \
  --ambiguous-out local/viewpoint_ambiguous.csv
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


VALID = {"O", "X", "?", ""}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--gold-out", required=True)
    parser.add_argument("--ambiguous-out", required=True)
    parser.add_argument("--min-agreement", type=int, default=4)
    args = parser.parse_args()

    pair_rows = read_csv(args.pairs)

    if not pair_rows:
        raise SystemExit("pairs CSV가 비어 있습니다.")

    required_pairs = {
        "pair_id",
        "issue_id",
        "event_name",
        "stance",
        "article_id_a",
        "title_a",
        "evidence_a",
        "article_id_b",
        "title_b",
        "evidence_b",
    }

    missing = required_pairs - set(pair_rows[0])
    if missing:
        raise SystemExit(
            "pairs CSV 필수 컬럼 없음: " + ", ".join(sorted(missing))
        )

    pair_map = {r["pair_id"]: r for r in pair_rows}

    if len(pair_map) != len(pair_rows):
        raise SystemExit("pairs CSV에 중복 pair_id가 있습니다.")

    votes = {pair_id: {} for pair_id in pair_map}

    seen_labelers = set()

    for label_path in args.labels:
        rows = read_csv(label_path)

        if not rows:
            raise SystemExit(f"빈 라벨 파일: {label_path}")

        required_labels = {
            "pair_id",
            "issue_id",
            "article_id_a",
            "article_id_b",
            "labeler",
            "verdict",
        }

        missing = required_labels - set(rows[0])
        if missing:
            raise SystemExit(
                f"{label_path} 필수 컬럼 없음: "
                + ", ".join(sorted(missing))
            )

        file_labelers = {
            r["labeler"].strip()
            for r in rows
            if r["labeler"].strip()
        }

        if len(file_labelers) != 1:
            raise SystemExit(
                f"{label_path}: labeler 번호가 하나로 고정되어 있지 않습니다."
            )

        labeler = next(iter(file_labelers))

        if labeler in seen_labelers:
            raise SystemExit(
                f"같은 라벨러가 중복 제출되었습니다: {labeler}"
            )

        seen_labelers.add(labeler)

        row_pair_ids = set()

        for r in rows:
            pair_id = r["pair_id"].strip()

            if pair_id not in pair_map:
                raise SystemExit(
                    f"{label_path}: 알 수 없는 pair_id {pair_id}"
                )

            if pair_id in row_pair_ids:
                raise SystemExit(
                    f"{label_path}: pair_id 중복 {pair_id}"
                )

            row_pair_ids.add(pair_id)

            source = pair_map[pair_id]

            if (
                r["issue_id"] != source["issue_id"]
                or r["article_id_a"] != source["article_id_a"]
                or r["article_id_b"] != source["article_id_b"]
            ):
                raise SystemExit(
                    f"{label_path}: {pair_id}의 기사/이슈 ID가 원본과 다릅니다."
                )

            verdict = r["verdict"].strip().upper()

            if verdict not in VALID:
                raise SystemExit(
                    f"{label_path}: {pair_id} 잘못된 판정값 {verdict!r}"
                )

            votes[pair_id][labeler] = verdict

        expected_pair_ids = set(pair_map)
        missing_pair_ids = expected_pair_ids - row_pair_ids

        if missing_pair_ids:
            raise SystemExit(
                f"{label_path}: 제출되지 않은 문항이 있습니다: "
                + ", ".join(sorted(missing_pair_ids))
            )

    if len(seen_labelers) != len(args.labels):
        raise SystemExit("라벨러 파일 수와 실제 라벨러 수가 다릅니다.")

    extra_cols = [
        "vote_o",
        "vote_x",
        "vote_question",
        "vote_blank",
        "agreement",
        "gold_verdict",
    ]

    fieldnames = list(pair_rows[0].keys()) + extra_cols

    gold_rows = []
    ambiguous_rows = []

    for pair_id, source in pair_map.items():
        pair_votes = votes[pair_id]

        counter = Counter(pair_votes.values())

        o = counter["O"]
        x = counter["X"]
        q = counter["?"]
        blank = counter[""]

        winner = ""
        agreement = max(o, x)

        if o >= args.min_agreement:
            winner = "O"
        elif x >= args.min_agreement:
            winner = "X"

        out = dict(source)
        out.update(
            {
                "vote_o": o,
                "vote_x": x,
                "vote_question": q,
                "vote_blank": blank,
                "agreement": agreement,
                "gold_verdict": winner,
            }
        )

        if winner:
            gold_rows.append(out)
        else:
            ambiguous_rows.append(out)

    def write(path, rows):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(
            path,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write(args.gold_out, gold_rows)
    write(args.ambiguous_out, ambiguous_rows)

    total = len(pair_rows)

    unanimous = sum(
        1
        for pair_id in pair_map
        if max(
            Counter(votes[pair_id].values())["O"],
            Counter(votes[pair_id].values())["X"],
        ) == len(args.labels)
    )

    print("===== VIEWPOINT GOLD SUMMARY =====")
    print("labelers:", sorted(seen_labelers))
    print("total pairs:", total)
    print("gold pairs:", len(gold_rows))
    print("ambiguous pairs:", len(ambiguous_rows))
    print("unanimous pairs:", unanimous)

    if total:
        print(
            "gold coverage:",
            f"{len(gold_rows) / total * 100:.1f}%"
        )

    print("gold:", args.gold_out)
    print("ambiguous:", args.ambiguous_out)


if __name__ == "__main__":
    main()
