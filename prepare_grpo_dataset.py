import argparse
import json
import os
import yaml
from grpo_prompts import NEREL_JSON_INSTRUCTION_SYSTEM, NEREL_JSON_INSTRUCTION_USER


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_gold_batch(entities):
    return [[e["tag"], e["text"]] for e in entities]


def build_messages(text, add_no_think=False, no_think_suffix="/no_think"):
    user_text = NEREL_JSON_INSTRUCTION_USER.format(text=text)

    if add_no_think:
        user_text = f"{user_text} {no_think_suffix}"

    return [
        {"role": "system", "content": NEREL_JSON_INSTRUCTION_SYSTEM},
        {"role": "user", "content": user_text},
    ]


def transform_split(rows, add_no_think, no_think_suffix):
    out = []
    for row in rows:
        text = row["text"]
        entities = row["entities"]
        gold_batch = build_gold_batch(entities)
        if not gold_batch:
            continue
        out.append(
            {
                "messages": build_messages(text, add_no_think, no_think_suffix),
                "gold_batch": gold_batch,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["grpo_data_prep"]

    input_dir = config["input_dir"]
    out_dir = config["out_dir"]
    add_no_think = config.get("add_no_think", False)
    no_think_suffix = config.get("no_think_suffix", "/no_think")

    os.makedirs(out_dir, exist_ok=True)

    for split_name in ["train", "dev", "test"]:
        in_path = os.path.join(input_dir, f"{split_name}.jsonl")
        if not os.path.exists(in_path):
            continue

        rows = load_jsonl(in_path)
        out_rows = transform_split(
            rows,
            add_no_think=add_no_think,
            no_think_suffix=no_think_suffix,
        )

        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        save_jsonl(out_rows, out_path)


if __name__ == "__main__":
    main()
