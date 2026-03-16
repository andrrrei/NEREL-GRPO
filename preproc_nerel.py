import argparse
import json
import os
import random
import re
import yaml
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datasets import Dataset, load_from_disk

ENTITY_RE = re.compile(
    r"(?P<id>\w+)\s+(?P<tag>\w+)\s+(?P<spans>\d+\s+\d+(?:;\d+\s+\d+)*)\s+(?P<text>.+)",
    flags=re.UNICODE,
)


@dataclass
class Entity:
    tag: str
    begin: int
    end: int
    text: str


def parse_entity(line: str) -> Tuple[Optional[List[Entity]], bool]:
    m = ENTITY_RE.match(line.replace("\t", " ").strip())
    if not m:
        return None, False
    tag = m.group("tag")
    spans_str = m.group("spans")
    text = m.group("text")
    if ";" in spans_str:
        return None, True
    begin, end = map(int, spans_str.split())
    return [Entity(tag=tag, begin=begin, end=end, text=text)], False


def process_entities(sample: dict) -> Optional[List[Entity]]:
    out: List[Entity] = []
    for line in sample.get("entities", []):
        entities, is_discont = parse_entity(line)
        if is_discont:
            return None
        if entities is None:
            return None
        out.extend(entities)
    out.sort(key=lambda x: (x.begin, -x.end))
    return out


def split_sample_to_segments(text: str, split_by: str) -> List[Tuple[str, int, int]]:
    parts = text.split(split_by) if split_by else [text]
    segs: List[Tuple[str, int, int]] = []
    offset = 0
    for i, part in enumerate(parts):
        seg_begin = offset
        seg_end = offset + len(part)
        segs.append((part, seg_begin, seg_end))
        offset = seg_end + (len(split_by) if i != len(parts) - 1 else 0)
    return segs


def sample_to_sentence_examples(
    sample: dict, do_split: bool, split_by: str
) -> List[dict]:

    entities = process_entities(sample)
    if entities is None:
        return []

    text = sample.get("text", "")
    segs = (
        split_sample_to_segments(text, split_by) if do_split else [(text, 0, len(text))]
    )

    idx = 0
    examples = []

    for seg_text, seg_b, seg_e in segs:
        seg_entities = []

        while idx < len(entities) and entities[idx].begin < seg_e:
            if entities[idx].begin < seg_b:
                idx += 1
                continue

            if entities[idx].end <= seg_e:
                seg_entities.append(
                    {
                        "tag": entities[idx].tag,
                        "begin": entities[idx].begin - seg_b,
                        "end": entities[idx].end - seg_b,
                        "text": entities[idx].text,
                    }
                )

            idx += 1

        if not seg_entities:
            continue

        examples.append({"query": seg_text, "entities": seg_entities})

    return examples


def select_fraction(ds: Dataset, fraction: float, seed: int) -> Dataset:
    if fraction >= 1.0:
        return ds
    n = len(ds)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    return ds.select(idxs[:k])


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["nerel_preproc"]

    os.makedirs(config["out_dir"], exist_ok=True)

    ds = load_from_disk(config["local_dataset_path"])

    split_fractions = {
        "train": config["fraction_train"],
        "dev": config["fraction_dev"],
        "test": config["fraction_test"],
    }

    stats = {}

    for split_name in ["train", "dev", "test"]:

        split_ds = select_fraction(
            ds[split_name], split_fractions[split_name], config["seed"]
        )
        out_path = os.path.join(config["out_dir"], f"{split_name}.jsonl")
        kept = 0
        dropped_discont = 0
        dropped_empty = 0
        total_docs = len(split_ds)

        with open(out_path, "w", encoding="utf-8") as f:
            for doc in split_ds:
                examples = sample_to_sentence_examples(
                    doc,
                    do_split=config["do_split"],
                    split_by=config["split_by"],
                )
                if examples == []:
                    entities = process_entities(doc)
                    if entities is None:
                        dropped_discont += 1
                    else:
                        dropped_empty += 1
                    continue

                for ex in examples:
                    record = {
                        "text": ex["query"],
                        "entities": ex["entities"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1

        stats[split_name] = {
            "docs_in_split": total_docs,
            "examples_written": kept,
            "dropped_docs_discont": dropped_discont,
            "dropped_docs_empty_or_too_short": dropped_empty,
        }

    with open(
        os.path.join(config["out_dir"], "stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
