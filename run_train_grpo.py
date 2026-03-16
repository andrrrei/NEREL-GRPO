import argparse
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["grpo_train"]

    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    train_log = os.path.join(run_dir, "train.log")
    output_dir = os.path.join(run_dir, "model_output")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(train_cfg["cuda_visible_devices"])
    env["PYTHONUNBUFFERED"] = "1"
    env["RUN_LOG_DIR"] = run_dir

    python_bin = sys.executable
    env_prefix = os.path.dirname(os.path.dirname(python_bin))
    env["LD_LIBRARY_PATH"] = f"{env_prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")

    train_args = {
        "rlhf_type": "grpo",
        "model": train_cfg["model"],
        "dataset": train_cfg["dataset"],
        "external_plugins": train_cfg["external_plugins"],
        "reward_funcs": train_cfg["reward_funcs"],
        "train_type": train_cfg["train_type"],
        "torch_dtype": train_cfg["torch_dtype"],
        "learning_rate": train_cfg["learning_rate"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "lora_rank": train_cfg["lora_rank"],
        "lora_alpha": train_cfg["lora_alpha"],
        "target_modules": train_cfg["target_modules"],
        "num_generations": train_cfg["num_generations"],
        "generation_batch_size": train_cfg["generation_batch_size"],
        "temperature": train_cfg["temperature"],
        "top_p": train_cfg["top_p"],
        "max_length": train_cfg["max_length"],
        "max_completion_length": train_cfg["max_completion_length"],
        "use_vllm": str(train_cfg["use_vllm"]).lower(),
        "vllm_mode": train_cfg["vllm_mode"],
        "vllm_gpu_memory_utilization": train_cfg["vllm_gpu_memory_utilization"],
        "vllm_max_model_len": train_cfg["vllm_max_model_len"],
        "vllm_tensor_parallel_size": train_cfg["vllm_tensor_parallel_size"],
        "logging_steps": train_cfg["logging_steps"],
        "save_steps": train_cfg["save_steps"],
        "save_total_limit": train_cfg["save_total_limit"],
        "report_to": train_cfg["report_to"],
        "sleep_level": train_cfg["sleep_level"],
        "output_dir": output_dir,
    }

    train_cmd_parts = [f"{quote(python_bin)} -u -m swift.cli.rlhf"]
    for k, v in train_args.items():
        train_cmd_parts.append(f"--{k} {quote(v)}")
    train_cmd = " ".join(train_cmd_parts)

    with open(train_log, "w", encoding="utf-8") as f:
        f.write(f"run_dir={run_dir}\n")
        f.write(f"output_dir={output_dir}\n")
        f.write(f"cmd={train_cmd}\n\n")

    run_cmd(f"{train_cmd} 2>&1 | tee -a {quote(train_log)}", env=env)


if __name__ == "__main__":
    main()
