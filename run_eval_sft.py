import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def quote(v):
    return shlex.quote(str(v))


def run_cmd(cmd, env=None):
    process = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in process.stdout:
        sys.stdout.write(line)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def find_last_checkpoint(model_output_dir: str) -> str:
    if not os.path.isdir(model_output_dir):
        raise FileNotFoundError(f"Model output dir not found: {model_output_dir}")

    candidates = []
    for root, dirs, _files in os.walk(model_output_dir):
        for name in dirs:
            m = re.fullmatch(r"checkpoint-(\d+)", name)
            if m:
                step = int(m.group(1))
                candidates.append((step, os.path.join(root, name)))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* found in: {model_output_dir}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def choose_model_path(
    run_dir: str,
    eval_cfg: Dict[str, Any],
    checkpoint_path_arg: str,
) -> str:
    checkpoint_path_arg = checkpoint_path_arg.strip()
    if checkpoint_path_arg:
        return checkpoint_path_arg

    if eval_cfg.get("use_merged_model", True):
        merged_dir = os.path.join(run_dir, "model_merged")
        if not os.path.isdir(merged_dir):
            raise FileNotFoundError(f"Merged model dir not found: {merged_dir}")
        return merged_dir

    model_output_dir = os.path.join(run_dir, "model_output")
    return find_last_checkpoint(model_output_dir)


def strip_gold_assistant(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages:
        raise ValueError("Empty messages in eval sample")

    if messages[-1]["role"] == "assistant":
        return messages[:-1]

    return messages


def build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    prompt_messages = strip_gold_assistant(messages)
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def batch_iter(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def generate_predictions(
    model,
    tokenizer,
    rows,
    batch_size,
    max_new_tokens,
    temperature,
    top_p,
    do_sample,
):
    prompts = [build_prompt(tokenizer, row["messages"]) for row in rows]
    predictions = []

    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx, prompt_batch in enumerate(batch_iter(prompts, batch_size), start=1):
        print(
            f"[eval] batch {batch_idx}/{total_batches} | batch_size={len(prompt_batch)}",
            flush=True,
        )

        enc = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        generate_kwargs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = model.generate(**generate_kwargs)

        input_lengths = enc["attention_mask"].sum(dim=1).tolist()

        for i in range(len(prompt_batch)):
            gen_ids = out[i][input_lengths[i] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            predictions.append({"response": text})

        print(f"[eval] finished batch {batch_idx}/{total_batches}", flush=True)

    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, default="")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["sft_train"]
    eval_cfg = config["sft_eval"]

    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    eval_dir = os.path.join(run_dir, "eval_test")
    os.makedirs(eval_dir, exist_ok=True)

    preds_path = os.path.join(eval_dir, "predictions.jsonl")
    metrics_path = os.path.join(eval_dir, "metrics.json")
    infer_log = os.path.join(eval_dir, "infer.log")
    metrics_log = os.path.join(eval_dir, "metrics.log")
    summary_path = os.path.join(eval_dir, "summary.json")

    model_path = choose_model_path(
        run_dir=run_dir,
        eval_cfg=eval_cfg,
        checkpoint_path_arg=args.checkpoint_path,
    )

    dataset_path = eval_cfg["dataset"]
    rows = load_jsonl(dataset_path)

    if not rows:
        raise ValueError(f"Empty evaluation dataset: {dataset_path}")

    torch_dtype = resolve_torch_dtype(train_cfg.get("torch_dtype", "bfloat16"))
    tokenizer_name = train_cfg.get("tokenizer_name", train_cfg["model"])

    tokenizer_source = model_path if os.path.isdir(model_path) else tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        trust_remote_code=True,
    )

    if "bos_token" in train_cfg:
        tokenizer.bos_token = train_cfg["bos_token"]
    if "eos_token" in train_cfg:
        tokenizer.eos_token = train_cfg["eos_token"]
    if "pad_token" in train_cfg:
        tokenizer.pad_token = train_cfg["pad_token"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "trust_remote_code": True,
    }

    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    attn_implementation = train_cfg.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    batch_size = int(eval_cfg.get("batch_size", 1))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 256))
    temperature = float(eval_cfg.get("temperature", 0.0))
    top_p = float(eval_cfg.get("top_p", 1.0))
    do_sample = bool(eval_cfg.get("do_sample", temperature > 0.0))

    with open(infer_log, "w", encoding="utf-8") as f:
        f.write(f"model_path={model_path}\n")
        f.write(f"dataset_path={dataset_path}\n")
        f.write(f"predictions_path={preds_path}\n")
        f.write(f"num_samples={len(rows)}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"max_new_tokens={max_new_tokens}\n")
        f.write(f"temperature={temperature}\n")
        f.write(f"top_p={top_p}\n")
        f.write(f"do_sample={do_sample}\n")

    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    save_jsonl(predictions, preds_path)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    python_bin = sys.executable
    metrics_cmd_parts = [
        f"{quote(python_bin)} compute_nerel_metrics.py",
        f"--preds {quote(preds_path)}",
        f"--dataset {quote(dataset_path)}",
        f"--out {quote(metrics_path)}",
    ]
    metrics_cmd = " ".join(metrics_cmd_parts)

    with open(metrics_log, "w", encoding="utf-8") as f:
        f.write(f"predictions_path={preds_path}\n")
        f.write(f"metrics_path={metrics_path}\n")
        f.write(f"cmd={metrics_cmd}\n\n")

    run_cmd(f"{metrics_cmd} 2>&1 | tee -a {quote(metrics_log)}", env=env)

    summary = {
        "model_path": model_path,
        "predictions_path": preds_path,
        "metrics_path": metrics_path,
        "num_examples": len(rows),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
