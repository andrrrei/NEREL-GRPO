import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import torch
import yaml
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


os.environ["WANDB_DISABLED"] = "true"


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("sft_train")
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


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


class SFTChatDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        max_tokens_count: int,
        only_target_loss: bool = True,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.only_target_loss = only_target_loss

    def __len__(self) -> int:
        return len(self.records)

    def _render_messages(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        record = self.records[idx]
        messages = record["messages"]

        full_text = self._render_messages(messages)

        full_enc = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_tokens_count,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = input_ids.copy()

        if self.only_target_loss:
            prompt_messages = messages[:-1]
            prompt_text = self._render_messages(prompt_messages)

            prompt_enc = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_tokens_count,
            )
            prompt_len = len(prompt_enc["input_ids"])
            prompt_len = min(prompt_len, len(labels))

            for i in range(prompt_len):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class SFTDataCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)

        if (
            self.pad_to_multiple_of is not None
            and max_len % self.pad_to_multiple_of != 0
        ):
            max_len = (
                (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        pad_id = self.tokenizer.pad_token_id
        for feat in features:
            seq_len = len(feat["input_ids"])
            pad_len = max_len - seq_len

            batch_input_ids.append(feat["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feat["attention_mask"] + [0] * pad_len)
            batch_labels.append(feat["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class TrainMetricsCallback(TrainerCallback):
    def __init__(self, train_log_path: str):
        self.train_log_path = train_log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        record = dict(logs)

        if state.global_step is not None:
            record["step"] = state.global_step

        if state.max_steps is not None and state.max_steps > 0:
            record["global_step/max_steps"] = f"{state.global_step}/{state.max_steps}"

        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(str(record) + "\n")


class SFTTrainerRunner:
    def __init__(self, config_path: str, run_dir: str):
        self.config_path = config_path
        self.run_dir = run_dir

        self.full_config = None
        self.config = None

        self.output_dir = None
        self.merged_dir = None

        self.training_args = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.data_collator = None
        self.model = None
        self.trainer = None

        self.logger = None
        self.train_log_path = None
        self.stage_log_path = None

    def load_config(self) -> None:
        self.train_log_path = os.path.join(self.run_dir, "train.log")
        self.stage_log_path = os.path.join(self.run_dir, "train_stage.log")
        self.logger = setup_logger(self.stage_log_path)

        with open(self.train_log_path, "w", encoding="utf-8") as f:
            f.write("")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.full_config = yaml.safe_load(f)

        self.config = self.full_config["sft_train"]

        output_subdir = self.config.get("output_subdir", "model_output")
        merged_subdir = self.config.get("merged_subdir", "model_merged")

        self.output_dir = os.path.join(self.run_dir, output_subdir)
        self.merged_dir = os.path.join(self.run_dir, merged_subdir)

        ensure_dir(self.output_dir)
        ensure_dir(self.merged_dir)

        self.logger.info("Loaded SFT config from: %s", self.config_path)
        self.logger.info("Run dir: %s", self.run_dir)
        self.logger.info("Model output dir: %s", self.output_dir)
        self.logger.info("Merged model dir: %s", self.merged_dir)

    def init_training_arguments(self) -> None:
        trainer_config = dict(self.config.get("trainer", {}))
        trainer_config["output_dir"] = self.output_dir

        parser = HfArgumentParser((TrainingArguments,))
        self.training_args = parser.parse_dict(trainer_config)[0]

    def init_tokenizer(self) -> None:
        tokenizer_name = self.config.get("tokenizer_name", self.config["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            trust_remote_code=True,
        )

        if "bos_token" in self.config:
            self.tokenizer.bos_token = self.config["bos_token"]
        if "eos_token" in self.config:
            self.tokenizer.eos_token = self.config["eos_token"]
        if "pad_token" in self.config:
            self.tokenizer.pad_token = self.config["pad_token"]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.bos_token is not None:
            self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.bos_token
            )
        if self.tokenizer.eos_token is not None:
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.eos_token
            )
        if self.tokenizer.pad_token is not None:
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )

        self.tokenizer.padding_side = "right"

    def prepare_datasets(self) -> None:
        train_records = load_jsonl(self.config["train_file"])
        val_records = load_jsonl(self.config["val_file"])

        self.train_dataset = SFTChatDataset(
            records=train_records,
            tokenizer=self.tokenizer,
            max_tokens_count=self.config["max_tokens_count"],
            only_target_loss=self.config.get("only_target_loss", True),
        )

        self.val_dataset = SFTChatDataset(
            records=val_records,
            tokenizer=self.tokenizer,
            max_tokens_count=self.config["max_tokens_count"],
            only_target_loss=self.config.get("only_target_loss", True),
        )

        self.logger.info("Train dataset size: %d", len(self.train_dataset))
        self.logger.info("Validation dataset size: %d", len(self.val_dataset))

    def init_data_collator(self) -> None:
        self.data_collator = SFTDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )

    def init_model(self) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        load_in_4bit = bool(self.config.get("load_in_4bit", False))
        load_in_8bit = bool(self.config.get("load_in_8bit", False))
        torch_dtype = resolve_torch_dtype(self.config.get("torch_dtype", "bfloat16"))

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs = {
            "pretrained_model_name_or_path": self.config["model"],
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        attn_implementation = self.config.get("attn_implementation")
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = {"": local_rank}
        elif int(os.environ.get("WORLD_SIZE", "1")) > 1:
            model_kwargs["device_map"] = {"": local_rank}

        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        if load_in_4bit or load_in_8bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.get(
                    "gradient_checkpointing", False
                ),
            )
        elif self.config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        if self.config.get("lora"):
            lora_cfg = dict(self.config["lora"])
            if "task_type" not in lora_cfg:
                lora_cfg["task_type"] = "CAUSAL_LM"

            modules_to_save = list(lora_cfg.get("modules_to_save", []))

            if (
                getattr(self.model.config, "tie_word_embeddings", False)
                and "lm_head" in modules_to_save
                and "embed_tokens" in modules_to_save
            ):
                lora_cfg["modules_to_save"] = ["lm_head"]
                print(
                    "ATTENTION!!! modules_to_save adjusted to ['lm_head'] due to tied embeddings"
                )

            lora_config_obj = LoraConfig(**lora_cfg)
            self.model = get_peft_model(self.model, lora_config_obj)

            final_modules_to_save = lora_cfg.get("modules_to_save", [])
            if (
                getattr(self.model.config, "tie_word_embeddings", False)
                and "lm_head" in final_modules_to_save
            ):
                self.model.base_model.model.model.embed_tokens.weight = (
                    self.model.base_model.model.lm_head.modules_to_save[
                        "default"
                    ].weight
                )

        self.logger.info("Model initialized from: %s", self.config["model"])
        self.logger.info(
            "LoRA enabled: %s | 4bit: %s | 8bit: %s | gradient_checkpointing: %s",
            bool(self.config.get("lora")),
            bool(self.config.get("load_in_4bit", False)),
            bool(self.config.get("load_in_8bit", False)),
            bool(self.config.get("gradient_checkpointing", False)),
        )

    def setup_trainer(self) -> None:
        metrics_callback = TrainMetricsCallback(self.train_log_path)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            callbacks=[metrics_callback],
        )

        if len(getattr(self.trainer, "label_names", [])) == 0:
            self.trainer.label_names.append("labels")

        self.logger.info("Trainer initialized")

    def save_merged_model(self) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return

        model_to_save = self.trainer.model
        try:
            model_to_save = model_to_save.merge_and_unload()
            self.logger.info("Successfully merged LoRA weights into base model")
        except Exception as e:
            self.logger.warning(
                "merge_and_unload failed, saving current model as-is: %s", e
            )

        model_to_save.eval()
        model_to_save.save_pretrained(self.merged_dir)
        self.tokenizer.save_pretrained(self.merged_dir)

        self.logger.info("Merged model saved to: %s", self.merged_dir)

    def train(self) -> None:
        self.load_config()
        self.init_training_arguments()
        self.init_tokenizer()
        self.prepare_datasets()
        self.init_data_collator()
        self.init_model()
        self.setup_trainer()

        self.logger.info("Starting SFT training")

        try:
            self.trainer.train()
            self.trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info("Trainer model saved to: %s", self.output_dir)
        finally:
            self.save_merged_model()

        self.logger.info("SFT training finished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    runner = SFTTrainerRunner(
        config_path=args.config,
        run_dir=args.run_dir,
    )
    runner.train()


if __name__ == "__main__":
    main()
