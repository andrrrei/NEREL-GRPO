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
    ap.add_argument(
        "--train_mode",
        type=str,
        required=True,
        choices=["grpo", "sft"],
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg = config["run"]
    train_mode = args.train_mode

    if train_mode == "grpo":
        train_cfg = config["grpo_train"]
    else:
        train_cfg = config["sft_train"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'{run_cfg["experiment_name"]}_{train_mode}_{timestamp}'
    run_dir = os.path.join(run_cfg["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, "pipeline.log"))
    logger.info("Starting run: %s", run_name)
    logger.info("Train mode: %s", train_mode)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["RUN_LOG_DIR"] = run_dir

    if "cuda_visible_devices" in train_cfg:
        env["CUDA_VISIBLE_DEVICES"] = str(train_cfg["cuda_visible_devices"])

    python_bin = sys.executable

    preproc_cmd = (
        f"{quote(python_bin)} preproc_nerel.py " f"--config {quote(args.config)}"
    )

    if train_mode == "grpo":
        data_prep_cmd = (
            f"{quote(python_bin)} prepare_grpo_dataset.py "
            f"--config {quote(args.config)}"
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
        data_prep_log = os.path.join(run_dir, "grpo_data_prep.log")
    else:
        data_prep_cmd = (
            f"{quote(python_bin)} prepare_sft_dataset.py "
            f"--config {quote(args.config)}"
        )
        train_cmd = (
            f"{quote(python_bin)} run_train_sft.py "
            f"--config {quote(args.config)} "
            f"--run_dir {quote(run_dir)}"
        )
        eval_cmd = (
            f"{quote(python_bin)} run_eval_sft.py "
            f"--config {quote(args.config)} "
            f"--run_dir {quote(run_dir)}"
        )
        data_prep_log = os.path.join(run_dir, "sft_data_prep.log")

    plot_cmd = (
        f"{quote(python_bin)} build_plots.py "
        f"--run_dir {quote(run_dir)} "
        f"--train_mode {quote(train_mode)}"
    )

    preproc_log = os.path.join(run_dir, "preproc.log")
    train_stage_log = os.path.join(run_dir, "train_stage.log")
    eval_log = os.path.join(run_dir, "eval.log")
    plot_log = os.path.join(run_dir, "plots.log")

    try:
        logger.info("Stage 1/5: preprocessing NEREL")
        run_cmd(preproc_cmd, logger, preproc_log, env=env)

        logger.info("Stage 2/5: preparing %s dataset", train_mode.upper())
        run_cmd(data_prep_cmd, logger, data_prep_log, env=env)

        logger.info("Stage 3/5: training %s", train_mode.upper())
        run_cmd(train_cmd, logger, train_stage_log, env=env)

        logger.info("Stage 4/5: evaluating on test split")
        run_cmd(eval_cmd, logger, eval_log, env=env)

        logger.info("Stage 5/5: plotting training curves")
        run_cmd(plot_cmd, logger, plot_log, env=env)

        logger.info("Run finished successfully: %s", run_name)

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()
