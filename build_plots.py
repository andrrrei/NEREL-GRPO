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


def extract_step(rec, fallback_step):
    step = rec.get("global_step/max_steps", None)

    if isinstance(step, str) and "/" in step:
        try:
            return int(step.split("/")[0])
        except Exception:
            return fallback_step

    if "step" in rec:
        try:
            return int(rec["step"])
        except Exception:
            return fallback_step

    if "global_step" in rec:
        try:
            return int(rec["global_step"])
        except Exception:
            return fallback_step

    return fallback_step


def plot_series(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.grid(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--train_mode", type=str, required=True, choices=["grpo", "sft"])
    args = ap.parse_args()

    run_dir = args.run_dir
    train_mode = args.train_mode

    train_log = os.path.join(run_dir, "train.log")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(train_log):
        raise FileNotFoundError(f"train.log not found: {train_log}")

    train_steps = []
    train_losses = []
    train_rewards = []
    train_grad_norms = []
    eval_steps = []
    eval_losses = []
    lr_steps = []
    learning_rates = []

    train_counter = 0
    eval_counter = 0
    lr_counter = 0

    with open(train_log, "r", encoding="utf-8") as f:
        for line in f:
            rec = try_parse_record(line)
            if rec is None:
                continue

            has_loss = "loss" in rec
            has_eval_loss = "eval_loss" in rec
            has_lr = "learning_rate" in rec

            if has_loss:
                train_counter += 1
                step = extract_step(rec, train_counter)
                train_steps.append(step)
                train_losses.append(rec.get("loss"))
                train_rewards.append(rec.get("reward"))
                train_grad_norms.append(rec.get("grad_norm"))

            if has_eval_loss:
                eval_counter += 1
                step = extract_step(rec, eval_counter)
                eval_steps.append(step)
                eval_losses.append(rec.get("eval_loss"))

            if has_lr:
                lr_counter += 1
                step = extract_step(rec, lr_counter)
                lr_steps.append(step)
                learning_rates.append(rec.get("learning_rate"))

    if train_mode == "grpo":
        if not train_steps:
            raise RuntimeError("No GRPO training metric records found in train.log")

        out_path = os.path.join(plots_dir, "training_curves.png")
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        plot_series(axes[0], train_steps, train_rewards, "Reward")
        plot_series(axes[1], train_steps, train_losses, "Loss")
        plot_series(axes[2], train_steps, train_grad_norms, "Grad norm")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")
        return

    available = []
    if train_steps and any(v is not None for v in train_losses):
        available.append(("Train loss", train_steps, train_losses))
    if eval_steps and any(v is not None for v in eval_losses):
        available.append(("Eval loss", eval_steps, eval_losses))
    if train_steps and any(v is not None for v in train_grad_norms):
        available.append(("Grad norm", train_steps, train_grad_norms))

    if not available:
        raise RuntimeError("No SFT metric records found in train.log")

    out_path = os.path.join(plots_dir, "training_curves.png")
    _, axes = plt.subplots(len(available), 1, figsize=(12, 3 * len(available)))

    if len(available) == 1:
        axes = [axes]

    for ax, (title, xs, ys) in zip(axes, available):
        plot_series(ax, xs, ys, title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
