import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import yaml


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


def find_last_checkpoint(model_output_dir: str) -> str:
    if not os.path.isdir(model_output_dir):
        raise FileNotFoundError(f"Model output dir not found: {model_output_dir}")

    run_dirs = []
    for name in os.listdir(model_output_dir):
        path = os.path.join(model_output_dir, name)
        if os.path.isdir(path):
            run_dirs.append(path)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {model_output_dir}")

    run_dirs.sort()
    last_run = run_dirs[-1]
    candidates = []
    for name in os.listdir(last_run):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m:
            step = int(m.group(1))
            candidates.append((step, os.path.join(last_run, name)))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* found in: {last_run}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, default="")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["grpo_train"]
    eval_cfg = config["grpo_eval"]

    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    model_output_dir = os.path.join(run_dir, "model_output")
    checkpoint_path = args.checkpoint_path.strip() or find_last_checkpoint(
        model_output_dir
    )
    #     logger.info("Using checkpoint: %s", checkpoint_path)

    eval_dir = os.path.join(run_dir, "eval_test")
    os.makedirs(eval_dir, exist_ok=True)

    preds_path = os.path.join(eval_dir, "predictions.jsonl")
    metrics_path = os.path.join(eval_dir, "metrics.json")
    infer_log = os.path.join(eval_dir, "infer.log")
    metrics_log = os.path.join(eval_dir, "metrics.log")
    summary_path = os.path.join(eval_dir, "summary.json")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(train_cfg["cuda_visible_devices"])
    env["PYTHONUNBUFFERED"] = "1"
    env["RUN_LOG_DIR"] = run_dir

    python_bin = sys.executable
    env_prefix = os.path.dirname(os.path.dirname(python_bin))
    env["LD_LIBRARY_PATH"] = f"{env_prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")

    infer_args = {
        "model": train_cfg["model"],
        "adapters": checkpoint_path,
        "val_dataset": eval_cfg["dataset"],
        "infer_backend": eval_cfg["infer_backend"],
        "max_new_tokens": eval_cfg["max_new_tokens"],
        "temperature": eval_cfg["temperature"],
        "top_p": eval_cfg["top_p"],
        "result_path": preds_path,
    }

    infer_cmd_parts = [f"{quote(python_bin)} -u -m swift.cli.infer"]
    for k, v in infer_args.items():
        infer_cmd_parts.append(f"--{k} {quote(v)}")
    infer_cmd = " ".join(infer_cmd_parts)

    metrics_cmd_parts = [
        f"{quote(python_bin)} compute_nerel_metrics.py",
        f"--preds {quote(preds_path)}",
        f"--dataset {quote(eval_cfg['dataset'])}",
        f"--out {quote(metrics_path)}",
    ]
    metrics_cmd = " ".join(metrics_cmd_parts)

    with open(infer_log, "w", encoding="utf-8") as f:
        f.write(f"checkpoint_path={checkpoint_path}\n")
        f.write(f"predictions_path={preds_path}\n")
        f.write(f"cmd={infer_cmd}\n\n")

    run_cmd(f"{infer_cmd} 2>&1 | tee -a {quote(infer_log)}", env=env)

    with open(metrics_log, "w", encoding="utf-8") as f:
        f.write(f"predictions_path={preds_path}\n")
        f.write(f"metrics_path={metrics_path}\n")
        f.write(f"cmd={metrics_cmd}\n\n")

    run_cmd(f"{metrics_cmd} 2>&1 | tee -a {quote(metrics_log)}", env=env)

    summary = {
        "checkpoint_path": checkpoint_path,
        "predictions_path": preds_path,
        "metrics_path": metrics_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
