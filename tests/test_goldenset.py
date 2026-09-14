import numpy as np
import pandas as pd

from hannun.quality.goldenset import (consensus_components, labeler_agreement, load_labels, load_pairs,
                                      representative_of, score_pairs)


def write_csv(path, header, rows):
    path.write_text("﻿" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_load_pairs_keeps_only_o_and_x_verdicts(tmp_path):
    path = write_csv(tmp_path / "pairs.csv", "article_id_a,article_id_b,판정(O=같은이슈,X=다른이슈)",
                     ["a,b,O", "a,c, x ", "b,c,", "c,d,?"])
    assert load_pairs(path) == [("a", "b", True), ("a", "c", False)]


def test_load_labels_groups_by_labeler_despite_bom(tmp_path):
    path = write_csv(tmp_path / "labels.csv", "article_id,labeler,cluster", ["a,1,x", "b,1,x", "a,2,y"])
    assert load_labels(path) == {"1": {"a": "x", "b": "x"}, "2": {"a": "y"}}


def test_consensus_components_joins_pairs_with_enough_votes():
    by_labeler = {
        "1": {"a": "1", "b": "1", "c": "2"},
        "2": {"a": "1", "b": "1", "c": "1"},
        "3": {"a": "1", "b": "2", "c": "3"},
    }
    gold = consensus_components(by_labeler, min_agree=2)
    assert gold["a"] == gold["b"]        # 두 명이 같이 묶었다
    assert gold["c"] != gold["a"]        # c 는 한 명(2번)만 a 와 묶었다 — 합의 미달


def test_labeler_agreement_is_one_for_identical_labelers():
    by_labeler = {"1": {"a": "x", "b": "x", "c": "y"}, "2": {"a": "p", "b": "p", "c": "q"}}
    assert labeler_agreement(by_labeler) == 1.0


def test_labeler_agreement_is_none_without_overlap():
    assert labeler_agreement({"1": {"a": "x"}, "2": {"b": "y"}}) is None


def test_score_pairs_splits_strata_by_pipeline_state():
    label_of = {"a": 0, "b": 0, "c": 1, "d": -1}
    vector_of = {
        "a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]),
        "c": np.array([0.9, 0.436]), "d": np.array([0.0, 1.0]),
    }
    pairs = [("a", "b", True), ("a", "c", True), ("a", "d", False), ("a", "zz", True)]
    scores = score_pairs(pairs, label_of, vector_of, boundary_sim=0.75)

    assert scores["scored_pairs"] == 3                       # zz 는 파이프라인에 없어 제외
    assert scores["same_issue_predicted"] == {"n": 1, "precision": 1.0}
    assert scores["boundary"] == {"n": 1, "oversplit_rate": 1.0}   # a·c 는 갈렸지만 유사도 0.9
    assert scores["random"] == {"n": 1, "human_same_rate": 0.0}
    assert scores["recall_in_sample"] == 0.5                 # 사람 O 두 쌍 중 기계도 묶은 건 a·b


def test_representative_of_maps_folded_to_representative_and_others_to_self():
    frame = pd.DataFrame({"article_id": ["a", "b", "c"], "duplicate_of": [None, "a", None]})
    assert representative_of(frame) == {"a": "a", "b": "a", "c": "c"}
