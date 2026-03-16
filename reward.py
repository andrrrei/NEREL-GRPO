import logging
from typing import List, Any
from swift.rewards.orm import ORM, orms
import os
from nerel_utils import (
    parse_model_output_json,
    schema_validity,
    extract_entities,
    calc_macro_f1,
    calc_order_score,
)

run_log_dir = os.getenv("RUN_LOG_DIR", ".")
log_path = os.path.join(run_log_dir, "reward.log")

logger = logging.getLogger("reward")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class NerelReward(ORM):
    W_JSON_PARSE = 0.05
    W_SCHEMA = 0.05
    W_MACRO_F1 = 0.85
    W_ORDER = 0.05

    def __call__(
        self,
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        gold_batch: List[Any] = kwargs.get("gold_batch", [])
        rewards: List[float] = []

        for pred_text, gold in zip(completions, gold_batch):
            #             logger.info("COMPLETION: %s", pred_text[:300])

            pred_json = parse_model_output_json(pred_text)

            json_parse_score = 1.0 if pred_json is not None else 0.0
            if pred_json is None:
                rewards.append(0.0)
                logger.info(
                    "json_parse=0.0000 schema=0.0000 macro_f1=0.0000 order=0.0000 reward=0.0000"
                )
                continue

            schema_score = schema_validity(pred_json)

            gold_entities = extract_entities(gold)
            pred_entities = extract_entities(pred_json)
            macro_f1 = calc_macro_f1(gold_entities, pred_entities)

            order_score = calc_order_score(gold_entities, pred_entities)

            reward = (
                self.W_JSON_PARSE * json_parse_score
                + self.W_SCHEMA * schema_score
                + self.W_MACRO_F1 * macro_f1
                + self.W_ORDER * order_score
            )

            logger.info(
                "json_parse=%.4f schema=%.4f macro_f1=%.4f order=%.4f reward=%.4f",
                json_parse_score,
                schema_score,
                macro_f1,
                order_score,
                reward,
            )

            rewards.append(float(reward))

        return rewards


orms["nerel_reward"] = NerelReward
