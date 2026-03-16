import json
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Any, Dict


ALL_TAGS = {
    "AGE",
    "AWARD",
    "CITY",
    "COUNTRY",
    "CRIME",
    "DATE",
    "DISEASE",
    "DISTRICT",
    "EVENT",
    "FACILITY",
    "FAMILY",
    "IDEOLOGY",
    "LANGUAGE",
    "LAW",
    "LOCATION",
    "MONEY",
    "NATIONALITY",
    "NUMBER",
    "ORDINAL",
    "ORGANIZATION",
    "PENALTY",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "PROFESSION",
    "RELIGION",
    "STATE_OR_PROVINCE",
    "TIME",
    "WORK_OF_ART",
}

THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)


def strip_empty_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def parse_model_output_json(text: str):
    text = strip_empty_think(text).strip()

    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except Exception:
        pass

    m = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else None
    except Exception:
        return None


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def is_valid_entity_item(item: Any) -> bool:
    if not isinstance(item, list) or len(item) != 2:
        return False

    tag, text = item
    if not isinstance(tag, str) or not isinstance(text, str):
        return False

    tag = tag.strip()
    text = normalize_text(text)

    if tag not in ALL_TAGS:
        return False
    if not text:
        return False

    return True


def schema_validity(data: Any) -> float:
    if not isinstance(data, list):
        return 0.0

    if len(data) == 0:
        return 1.0

    valid_items = sum(1 for item in data if is_valid_entity_item(item))
    return valid_items / len(data)


def extract_entities(items: Any) -> List[Tuple[str, str]]:
    if not isinstance(items, list):
        return []

    entities: List[Tuple[str, str]] = []
    for item in items:
        if not is_valid_entity_item(item):
            continue
        tag, text = item
        entities.append((tag.strip(), normalize_text(text)))
    return entities


def calc_metrics_per_tag(
    gold_entities: List[Tuple[str, str]],
    pred_entities: List[Tuple[str, str]],
) -> Dict[str, Tuple[int, int, int]]:
    gold_by_tag = defaultdict(Counter)
    pred_by_tag = defaultdict(Counter)

    for tag, text in gold_entities:
        gold_by_tag[tag][text] += 1

    for tag, text in pred_entities:
        pred_by_tag[tag][text] += 1

    stats = {}
    all_tags = set(gold_by_tag.keys()) | set(pred_by_tag.keys())

    for tag in all_tags:
        gold_counter = gold_by_tag.get(tag, Counter())
        pred_counter = pred_by_tag.get(tag, Counter())

        tp = sum((gold_counter & pred_counter).values())
        fp = sum((pred_counter - gold_counter).values())
        fn = sum((gold_counter - pred_counter).values())

        stats[tag] = (tp, fp, fn)

    return stats


def calc_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def calc_macro_f1(
    gold_entities: List[Tuple[str, str]],
    pred_entities: List[Tuple[str, str]],
) -> float:
    if not gold_entities and not pred_entities:
        return 1.0

    stats = calc_metrics_per_tag(gold_entities, pred_entities)
    if not stats:
        return 0.0

    f1_values = []
    for _, (tp, fp, fn) in stats.items():
        _, _, f1 = calc_precision_recall_f1(tp, fp, fn)
        f1_values.append(f1)

    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def lcs_length(seq1: List[Tuple[str, str]], seq2: List[Tuple[str, str]]) -> int:
    n = len(seq1)
    m = len(seq2)

    if n == 0 or m == 0:
        return 0

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        a = seq1[i - 1]
        for j in range(1, m + 1):
            b = seq2[j - 1]
            if a == b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]


def calc_order_score(
    gold_entities: List[Tuple[str, str]],
    pred_entities: List[Tuple[str, str]],
) -> float:
    if not gold_entities and not pred_entities:
        return 1.0

    if not gold_entities:
        return 0.0

    return lcs_length(gold_entities, pred_entities) / len(gold_entities)
