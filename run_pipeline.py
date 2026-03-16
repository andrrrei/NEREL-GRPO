import argparse
import datetime
import logging
import os
import shlex
import subprocess
import sys
import yaml


def setup_logger(log_path):
    logger = logging.getLogger("pipeline")
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


def run_cmd(cmd, logger, log_file, env=None):
    logger.info("RUN: %s", cmd)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Running command:\n{cmd}\n\n")
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
            f.write(line)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Run failed with code {process.returncode}")


def quote(v):
    return shlex.quote(str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg = config["run"]
    train_cfg = config["grpo_train"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'{run_cfg["experiment_name"]}_{timestamp}'
    run_dir = os.path.join(run_cfg["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, "pipeline.log"))
    logger.info("Starting run: %s", run_name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(train_cfg["cuda_visible_devices"])
    env["PYTHONUNBUFFERED"] = "1"
    env["RUN_LOG_DIR"] = run_dir

    python_bin = sys.executable

    preproc_cmd = (
        f"{quote(python_bin)} preproc_nerel.py " f"--config {quote(args.config)}"
    )
    grpo_prep_cmd = (
        f"{quote(python_bin)} prepare_grpo_dataset.py " f"--config {quote(args.config)}"
    )
    train_cmd = (
        f"{quote(python_bin)} run_train_grpo.py "
        f"--config {quote(args.config)} "
        f"--run_dir {quote(run_dir)}"
    )
    eval_cmd = (
        f"{quote(python_bin)} run_eval_grpo.py "
        f"--config {quote(args.config)} "
        f"--run_dir {quote(run_dir)}"
    )
    plot_cmd = f"{quote(python_bin)} build_plots.py " f"--run_dir {quote(run_dir)}"

    preproc_log = os.path.join(run_dir, "preproc.log")
    grpo_prep_log = os.path.join(run_dir, "grpo_data_prep.log")
    train_stage_log = os.path.join(run_dir, "train_stage.log")
    eval_log = os.path.join(run_dir, "eval.log")
    plot_log = os.path.join(run_dir, "plots.log")

    try:
        logger.info("Stage 1/5: preprocessing NEREL")
        run_cmd(preproc_cmd, logger, preproc_log, env=env)

        logger.info("Stage 2/5: preparing GRPO dataset")
        run_cmd(grpo_prep_cmd, logger, grpo_prep_log, env=env)

        logger.info("Stage 3/5: training GRPO")
        run_cmd(train_cmd, logger, train_stage_log, env=env)

        logger.info("Stage 4/5: eval last checkpoint on test split")
        run_cmd(eval_cmd, logger, eval_log, env=env)

        logger.info("Stage 5/5: plotting training curves")
        run_cmd(plot_cmd, logger, plot_log, env=env)

        logger.info("Run finished successfully: %s", run_name)

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()
