"""골드셋 채점 — 사람 라벨을 파이프라인 산출과 대조한다. 채점·시트 생성·스윕 스크립트가 같이 쓴다.

라벨은 두 형식이다. 쌍 CSV(article_id_a, article_id_b, 판정 O/X)는 중복·이슈 판정에 쓰고,
이슈 번호 CSV(article_id, labeler, cluster)는 군집 ARI 에 쓴다. 라벨러마다 이슈 번호가
제각각이라 "두 기사를 같은 이슈로 묶은 라벨러 수"가 min_agree 이상인 쌍을 잇고 연결
요소를 합의 정답으로 삼는다(라벨링 파일럿 방식 그대로, 티켓 104).
"""

import csv

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def load_pairs(path):
    """쌍 라벨 CSV 를 (article_id_a, article_id_b, 같은가) 목록으로. 판정 컬럼은 '판정' 접두로 찾고 O·X 만 남긴다."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    verdict_col = next(c for c in rows[0] if c.startswith("판정"))
    pairs = []
    for row in rows:
        verdict = row[verdict_col].strip().upper()
        if verdict in ("O", "X"):
            pairs.append((row["article_id_a"], row["article_id_b"], verdict == "O"))
    return pairs


def load_labels(path):
    """이슈 번호 CSV 를 labeler → {article_id: cluster} 로."""
    by_labeler = {}
    # utf-8-sig: 엑셀을 거친 CSV 는 BOM 이 붙어 첫 키가 '﻿article_id' 가 된다
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            by_labeler.setdefault(row["labeler"], {})[row["article_id"]] = row["cluster"]
    return by_labeler


def consensus_components(by_labeler: dict, min_agree: int):
    """쌍 득표 min_agree 이상을 잇고 연결 요소를 합의 이슈로 만든다. article_id → 합의 이슈 번호."""
    ids = sorted({a for clusters in by_labeler.values() for a in clusters})
    together = {}
    for clusters in by_labeler.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if a in clusters and b in clusters and clusters[a] == clusters[b]:
                    together[(a, b)] = together.get((a, b), 0) + 1

    neighbors = {a: set() for a in ids}
    for (a, b), votes in together.items():
        if votes >= min_agree:
            neighbors[a].add(b)
            neighbors[b].add(a)

    gold, visited = {}, set()
    label = 0
    for a in ids:
        if a in visited:
            continue
        stack = [a]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            gold[node] = label
            stack.extend(neighbors[node] - visited)
        label += 1
    return gold


def labeler_agreement(by_labeler: dict):
    """라벨러 간 평균 쌍별 ARI — 골드셋 자체의 신뢰도 지표. 겹치는 기사가 없으면 None."""
    names = sorted(by_labeler)
    scores = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            common = sorted(set(by_labeler[x]) & set(by_labeler[y]))
            if len(common) >= 2:
                scores.append(adjusted_rand_score([by_labeler[x][a] for a in common],
                                                  [by_labeler[y][a] for a in common]))
    return round(sum(scores) / len(scores), 4) if scores else None


def score_pairs(pairs: list, label_of: dict, vector_of: dict, boundary_sim: float = 0.75):
    """쌍 라벨을 파이프라인 상태와 대조해 층별 지표를 낸다.

    층(같은 이슈 예측·경계·무작위)은 시트에 없고 여기서 파이프라인 상태로 재유도한다 —
    라벨러가 층을 모르는 채 판정해야 편향이 없다. precision 은 같은 이슈로 묶은 쌍 중
    사람도 O 인 비율, oversplit_rate 는 다른 이슈로 갈렸지만 유사도가 boundary_sim 이상인
    쌍 중 사람이 O 인 비율. recall_in_sample 은 층화 표본 안 기준이라 절대값이 아니라
    설정 비교용이다. label_of 는 article_id → issue_local, vector_of 는 article_id → 정규화 벡터.
    """
    known = [(a, b, o) for a, b, o in pairs
             if a in label_of and b in label_of and a in vector_of and b in vector_of]

    strata = {"machine_same": [], "boundary": [], "random": []}
    for a, b, human_same in known:
        machine_same = label_of[a] == label_of[b] and label_of[a] >= 0
        sim = float(vector_of[a] @ vector_of[b])
        if machine_same:
            strata["machine_same"].append(human_same)
        elif sim >= boundary_sim:
            strata["boundary"].append(human_same)
        else:
            strata["random"].append(human_same)

    human_o = [(a, b) for a, b, o in known if o]
    caught = sum(1 for a, b in human_o if label_of[a] == label_of[b] and label_of[a] >= 0)
    return {
        "labeled_pairs": len(pairs),
        "scored_pairs": len(known),
        "same_issue_predicted": {"n": len(strata["machine_same"]),
                                 "precision": _rate(strata["machine_same"])},
        "boundary": {"n": len(strata["boundary"]), "oversplit_rate": _rate(strata["boundary"])},
        "random": {"n": len(strata["random"]), "human_same_rate": _rate(strata["random"])},
        "recall_in_sample": round(caught / len(human_o), 4) if human_o else None,
    }


def representative_of(dedup_frame: pd.DataFrame):
    """dedup 테이블(article_id, duplicate_of)에서 article_id → 그룹 대표. 대표와 단독은 자기 자신."""
    # 결측 duplicate_of 는 pandas 에서 None 이 아니라 NaN 으로 온다 — is None 검사는 뚫린다
    return {a: (a if pd.isna(rep) else rep)
            for a, rep in zip(dedup_frame.article_id, dedup_frame.duplicate_of)}


def _rate(items):
    return round(sum(items) / len(items), 4) if items else None
