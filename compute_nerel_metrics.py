import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from nerel_utils import (
    parse_model_output_json,
    extract_entities,
    calc_metrics_per_tag,
    calc_precision_recall_f1,
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def multiset_equal(
    gold_entities: List[Tuple[str, str]],
    pred_entities: List[Tuple[str, str]],
) -> bool:
    return Counter(gold_entities) == Counter(pred_entities)


def aggregate_micro_counts(
    gold_all: List[List[Tuple[str, str]]],
    pred_all: List[List[Tuple[str, str]]],
) -> Tuple[int, int, int]:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gold_entities, pred_entities in zip(gold_all, pred_all):
        stats = calc_metrics_per_tag(gold_entities, pred_entities)
        for tp, fp, fn in stats.values():
            total_tp += tp
            total_fp += fp
            total_fn += fn

    return total_tp, total_fp, total_fn


def aggregate_macro_metrics(
    gold_all: List[List[Tuple[str, str]]],
    pred_all: List[List[Tuple[str, str]]],
) -> Tuple[float, float, float]:

    per_tag_tp = defaultdict(int)
    per_tag_fp = defaultdict(int)
    per_tag_fn = defaultdict(int)

    for gold_entities, pred_entities in zip(gold_all, pred_all):
        stats = calc_metrics_per_tag(gold_entities, pred_entities)
        for tag, (tp, fp, fn) in stats.items():
            per_tag_tp[tag] += tp
            per_tag_fp[tag] += fp
            per_tag_fn[tag] += fn

    precisions = []
    recalls = []
    f1s = []

    all_tags = set(per_tag_tp) | set(per_tag_fp) | set(per_tag_fn)

    for tag in all_tags:
        tp = per_tag_tp[tag]
        fp = per_tag_fp[tag]
        fn = per_tag_fn[tag]

        precision, recall, f1 = calc_precision_recall_f1(tp, fp, fn)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return macro_precision, macro_recall, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    dataset = load_jsonl(args.dataset)
    preds = load_jsonl(args.preds)

    if len(dataset) != len(preds):
        raise ValueError(
            f"Dataset and preds size mismatch: {len(dataset)} vs {len(preds)}"
        )

    gold_all = []
    pred_all = []

    format_valid = 0
    exact_match = 0

    for sample, pred_row in zip(dataset, preds):
        gold_entities = extract_entities(sample["gold_batch"])

        response = pred_row.get("response", "")
        pred_json = parse_model_output_json(response)

        if pred_json is not None:
            format_valid += 1
            pred_entities = extract_entities(pred_json)
        else:
            pred_entities = []

        if multiset_equal(gold_entities, pred_entities):
            exact_match += 1

        gold_all.append(gold_entities)
        pred_all.append(pred_entities)

    total_tp, total_fp, total_fn = aggregate_micro_counts(gold_all, pred_all)
    micro_precision, micro_recall, micro_f1 = calc_precision_recall_f1(
        total_tp, total_fp, total_fn
    )

    macro_precision, macro_recall, macro_f1 = aggregate_macro_metrics(
        gold_all, pred_all
    )

    metrics = {
        "num_samples": len(dataset),
        "format_valid_rate": format_valid / len(dataset) if dataset else 0.0,
        "exact_match_rate": exact_match / len(dataset) if dataset else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to: {args.out}")


if __name__ == "__main__":
    main()
