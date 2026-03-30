import argparse
import datetime
import logging
import os
import subprocess
import sys
import yaml


def setup_logger(log_path):
    logger = logging.getLogger("queue")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def run_cmd(cmd, logger):
    logger.info("RUN: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        sys.stdout.write(line)
    process.wait()
    return process.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=str, required=True)
    args = ap.parse_args()

    with open(args.queue, "r", encoding="utf-8") as f:
        queue_cfg = yaml.safe_load(f)

    experiments = queue_cfg.get("experiments", [])
    stop_on_error = queue_cfg.get("stop_on_error", True)

    if not experiments:
        raise ValueError("No experiments found")

    queue_log_dir = "queue_runs"
    os.makedirs(queue_log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(queue_log_dir, f"queue_{timestamp}.log")
    logger = setup_logger(log_path)

    logger.info("Queue started")
    logger.info("Experiments count: %d", len(experiments))
    logger.info("stop_on_error=%s", stop_on_error)

    python_bin = sys.executable
    results = []

    for idx, exp in enumerate(experiments, start=1):
        if isinstance(exp, str):
            config_path = exp
            mode = "grpo"
        elif isinstance(exp, dict):
            config_path = exp.get("config")
            mode = exp.get("train_mode", "grpo")
        else:
            logger.error("Invalid experiment spec at index %d: %r", idx, exp)
            results.append(
                {
                    "config": None,
                    "train_mode": None,
                    "status": "invalid_spec",
                    "return_code": None,
                }
            )
            if stop_on_error:
                break
            continue

        logger.info(
            "Experiment %d/%d started: config=%s mode=%s",
            idx,
            len(experiments),
            config_path,
            mode,
        )

        if not config_path or not os.path.exists(config_path):
            logger.error("Config not found: %s", config_path)
            results.append(
                {
                    "config": config_path,
                    "train_mode": mode,
                    "status": "missing_config",
                    "return_code": None,
                }
            )
            if stop_on_error:
                break
            continue

        cmd = [
            python_bin,
            "run_pipeline.py",
            "--config",
            config_path,
            "--train_mode",
            mode,
        ]
        return_code = run_cmd(cmd, logger)

        if return_code == 0:
            logger.info(
                "Experiment finished successfully: config=%s mode=%s", config_path, mode
            )
            results.append(
                {
                    "config": config_path,
                    "mode": mode,
                    "status": "ok",
                    "return_code": 0,
                }
            )
        else:
            logger.error(
                "Experiment failed: config=%s mode=%s code=%s",
                config_path,
                mode,
                return_code,
            )
            results.append(
                {
                    "config": config_path,
                    "mode": mode,
                    "status": "failed",
                    "return_code": return_code,
                }
            )
            if stop_on_error:
                logger.info("Stopping queue")
                break

    summary_path = os.path.join(queue_log_dir, f"queue_{timestamp}_summary.yaml")
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"results": results}, f, allow_unicode=True, sort_keys=False)

    logger.info("Queue finished")


if __name__ == "__main__":
    main()
