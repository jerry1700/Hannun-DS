"""세부 견해 골든셋과 현재 관점 그룹 알고리즘을 비교한다."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from hannun.stance.viewpoint_group import generate_viewpoint_group_labels


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--errors-out")
    args = parser.parse_args()

    rows = read_csv(args.gold)

    if not rows:
        raise SystemExit("골든셋 CSV가 비어 있습니다.")

    required = {
        "pair_id",
        "issue_id",
        "event_name",
        "stance",
        "article_id_a",
        "evidence_a",
        "article_id_b",
        "evidence_b",
        "gold_verdict",
    }

    missing = required - set(rows[0])
    if missing:
        raise SystemExit(
            "골든셋 필수 컬럼 없음: " + ", ".join(sorted(missing))
        )

    groups = defaultdict(dict)

    for row in rows:
        key = (
            row["issue_id"],
            row["event_name"],
            row["stance"],
        )

        for suffix in ("a", "b"):
            article_id = row[f"article_id_{suffix}"]
            evidence = row[f"evidence_{suffix}"]

            previous = groups[key].get(article_id)

            if previous is not None and previous != evidence:
                raise SystemExit(
                    f"{article_id}: pair별 evidence가 서로 다릅니다."
                )

            groups[key][article_id] = evidence

    predicted_labels = {}

    for (issue_id, event_name, stance), articles in groups.items():
        inputs = [
            {
                "article_id": article_id,
                "content": evidence,
                "stance": stance,
            }
            for article_id, evidence in articles.items()
        ]

        labels = generate_viewpoint_group_labels(
            inputs,
            target=event_name,
        )

        for article_id, label in labels.items():
            predicted_labels[
                (issue_id, stance, article_id)
            ] = label

    confusion = Counter()
    error_rows = []

    for row in rows:
        gold = row["gold_verdict"].strip().upper()

        if gold not in {"O", "X"}:
            raise SystemExit(
                f"{row['pair_id']}: 잘못된 gold_verdict {gold!r}"
            )

        key_a = (
            row["issue_id"],
            row["stance"],
            row["article_id_a"],
        )
        key_b = (
            row["issue_id"],
            row["stance"],
            row["article_id_b"],
        )

        label_a = predicted_labels[key_a]
        label_b = predicted_labels[key_b]
        predicted = "O" if label_a == label_b else "X"

        confusion[(gold, predicted)] += 1

        if predicted != gold:
            error = dict(row)
            error["predicted_verdict"] = predicted
            error["predicted_label_a"] = label_a
            error["predicted_label_b"] = label_b
            error_rows.append(error)

    total = len(rows)
    correct = confusion[("O", "O")] + confusion[("X", "X")]

    print("===== VIEWPOINT GOLD SCORE =====")
    print("total:", total)
    print("correct:", correct)
    print("errors:", len(error_rows))
    print("accuracy:", f"{safe_div(correct, total) * 100:.1f}%")
    print()
    print("confusion matrix")
    print("gold O -> predicted O:", confusion[("O", "O")])
    print("gold O -> predicted X:", confusion[("O", "X")])
    print("gold X -> predicted O:", confusion[("X", "O")])
    print("gold X -> predicted X:", confusion[("X", "X")])

    for verdict in ("O", "X"):
        tp = confusion[(verdict, verdict)]
        fp = sum(
            confusion[(other, verdict)]
            for other in ("O", "X")
            if other != verdict
        )
        fn = sum(
            confusion[(verdict, other)]
            for other in ("O", "X")
            if other != verdict
        )

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(
            2 * precision * recall,
            precision + recall,
        )

        print()
        print(f"{verdict} precision:", f"{precision:.3f}")
        print(f"{verdict} recall:", f"{recall:.3f}")
        print(f"{verdict} f1:", f"{f1:.3f}")

    if args.errors_out:
        error_path = Path(args.errors_out)
        error_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(rows[0]) + [
            "predicted_verdict",
            "predicted_label_a",
            "predicted_label_b",
        ]

        with open(
            error_path,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_rows)

        print()
        print("errors out:", error_path)


if __name__ == "__main__":
    main()
