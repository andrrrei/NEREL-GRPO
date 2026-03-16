import argparse
import ast
import os
import matplotlib.pyplot as plt


def try_parse_record(line: str):
    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
        return None
    try:
        obj = ast.literal_eval(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    run_dir = args.run_dir
    train_log = os.path.join(run_dir, "train.log")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(train_log):
        raise FileNotFoundError(f"train.log not found: {train_log}")

    steps = []
    losses = []
    rewards = []
    grad_norms = []

    with open(train_log, "r", encoding="utf-8") as f:
        for line in f:
            rec = try_parse_record(line)
            if rec is None:
                continue
            if "loss" not in rec:
                continue

            step = rec.get("global_step/max_steps", "")
            if isinstance(step, str) and "/" in step:
                try:
                    step = int(step.split("/")[0])
                except Exception:
                    step = len(steps) + 1
            else:
                step = len(steps) + 1

            steps.append(step)
            losses.append(rec.get("loss"))
            rewards.append(rec.get("reward"))
            grad_norms.append(rec.get("grad_norm"))

    if not steps:
        raise RuntimeError("No training metric records found in train.log")

    out_path = os.path.join(plots_dir, "training_curves.png")

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(steps, rewards)
    plt.title("Reward")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(steps, losses)
    plt.title("Loss")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(steps, grad_norms)
    plt.title("Grad norm")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
