# NEREL SFT + GRPO Training Pipeline

This repository contains a full training and evaluation pipeline for fine-tuning LLMs on the [**NEREL**](https://huggingface.co/datasets/iluvvatar/NEREL) dataset using GRPO and SFT.

The goal of the project is to develop a model that can extract named entities from Russian text and output them in a structured JSON format. This project was created as part of an exploration of reinforcement learning techniques for aligning LLMs with the Russian language.

Training is performed using **GRPO (Group Relative Policy Optimization)** with a custom reward function designed specifically for entity extraction tasks.

The repository implements the full workflow:

- preprocessing the NEREL dataset
- SFT training
- GRPO training
- model evaluation
- metric computation
- training visualization
- experiment queueing

## Repository overview

Core pipeline scripts:

- `run_pipeline.py`  
  Main entry point of the project. Runs the full experiment pipeline: preprocessing → dataset preparation → training → evaluation → plotting.

- `run_train_grpo.py`  
  Launches GRPO reinforcement learning training using `swift.cli.rlhf`.

- `run_eval_grpo.py`  
  Runs inference on the test dataset using the trained model and computes evaluation metrics.

- `run_queue.py`  
  Runs multiple experiments sequentially using a list of configuration files.

- `run_train_sft.py`  
  Launches supervised fine-tuning using standard next-token prediction loss.

- `run_eval_sft.py`  
  Runs evaluation for SFT models.

Dataset preparation:

- `preproc_nerel.py`  
  Converts the raw NEREL dataset into a simplified JSONL format suitable for model training.

- `prepare_grpo_dataset.py`  
  Converts the preprocessed dataset into the message-based format required for GRPO training.

- `prepare_sft_dataset.py`  
Converts the dataset into a standard instruction-response format for supervised fine-tuning.

Evaluation:

- `compute_nerel_metrics.py`  
  Computes evaluation metrics from model predictions.

Training utilities:

- `reward.py`  
  Custom reward function used during GRPO training.

- `nerel_utils.py`  
  Shared utility functions for entity parsing and metric computation.

Visualization:

- `build_plots.py`  
  Builds training curves (reward, loss, gradient norm) from training logs.

Configuration:

- `grpo_config.yaml`  
  Central configuration file controlling all stages of the pipeline.

Environment setup:

- `environment.yml`  
  Micromamba environment specification.

- `requirements_lock.txt`  
  Exact Python dependency versions for reproducibility.


## Pipeline architecture

Each stage is implemented as an independent script but can also be executed together using `run_pipeline.py`.

### 1. Dataset preprocessing

Script: `preproc_nerel.py`


The raw NEREL dataset contains entity annotations with character offsets and may include discontinuous entities.

This stage performs the following transformations:

- parses the original NEREL annotations
- removes discontinuous entities
- optionally splits long documents into segments
- converts entity offsets into segment-local offsets
- converts the dataset into a simplified JSONL format

The following files are produced:

- train.jsonl
- dev.jsonl
- test.jsonl

Each record in the dataset contains:

- the text segment
- a list of entity annotations

Each entity annotation contains:

- tag
- entity text

Parts of the dataset preprocessing logic were adapted from the [repository](https://github.com/RefalMachine/llmtf_open/tree/main/llmtf/tasks/ner).




### 2. GRPO dataset preparation

Script: `prepare_grpo_dataset.py`

This stage converts the processed dataset into the format required for GRPO training.

Each example becomes a chat-style prompt consisting of:

- a system instruction
- a user prompt containing the text
- the list of gold entities

The following files are produced:

- train.jsonl
- dev.jsonl
- test.jsonl

These files are directly consumed by the GRPO trainer.


### 3. Supervised fine-tuning (SFT)

Script: `run_train_sft.py`

This stage performs standard supervised fine-tuning using the processed dataset.

The model is trained to generate the correct list of entities given an input text.

Key characteristics:

- next-token prediction objective
- LoRA fine-tuning
- deterministic training signal
- used to compare with GRPO stage


### 4. GRPO training

Script: `run_train_grpo.py`

Training is performed using the `swift` RLHF framework.

Key components of the training setup:

- LoRA fine-tuning
- GRPO reinforcement learning algorithm
- vLLM inference backend for rollout generation
- custom reward function

Each run directory contains:

- pipeline.log
- train.log
- model_output directory

The model_output directory contains model checkpoints produced during training.


### 5. Evaluation

Script: `run_eval_grpo.py`

After training finishes, the pipeline runs an evaluation stage on the test dataset.

This stage performs three main steps:

1. locate the latest model checkpoint produced during training  
2. run model inference on the test dataset  
3. compute evaluation metrics  

Inference is executed using the `swift.cli.infer` interface with the trained LoRA adapter.


The result directory contains:

- predictions.jsonl  
- metrics.json  
- infer.log  

predictions.jsonl contains the raw model responses for each dataset example.

metrics.json contains aggregated evaluation metrics computed from the predictions.



### 6. Metrics

Metrics are computed by the script `compute_nerel_metrics.py`.

The evaluation script parses model responses, extracts predicted entities, and compares them with the gold annotations.

The following metrics are reported:

- **format valid rate**  
  Fraction of model responses that can be successfully parsed into the expected entity format.

- **exact match rate**  
  Fraction of examples where the predicted entity list exactly matches the gold entity list.

- **micro precision, micro recall, micro f1**  
  Metrics computed across all entity predictions in the dataset.

- **macro precision, macro recall, macro f1**  
  Metrics computed separately for each entity tag and then averaged across tags.
  

## Reward function

The GRPO stage uses a custom reward function implemented in `reward.py`.

The reward function evaluates the quality of model predictions by comparing predicted entities with the gold annotations.

The reward is composed of several components:

- **json parse score**  
  Indicates whether the model output can be successfully parsed as a JSON list of entity pairs.

- **schema score**  
  Measures how many predicted items follow the expected schema.  
  Each predicted entity must be a pair containing a valid entity tag and entity text.

- **macro f1**  
  Measures entity extraction quality.  
  The F1 score is computed separately for each entity tag and then averaged across all tags as mentioned in [RuNNE-2022 Shared Task: Recognizing Nested Named Entities](https://arxiv.org/abs/2205.11159).

- **order score**  
  Measures how similar the predicted entity sequence is to the gold entity sequence.  
  This score is computed using the longest common subsequence (LCS) between the gold and predicted entity lists.

The final reward is computed as a weighted sum of these components.

The current weights are:

- JSON parse score weight: 0.05  
- schema validity weight: 0.05  
- macro F1 weight: 0.85  
- order similarity weight: 0.05  

The reward values are logged during training.

Shared utility functions used by the reward function are implemented in `nerel_utils.py`.

These utilities are also used by the evaluation scripts to ensure that reward computation and evaluation metrics use the same parsing logic.



## Running experiments

All experiments are controlled using a configuration file:

`grpo_config.yaml`

This configuration file contains several sections controlling different stages of the pipeline.

- run  
  General experiment settings such as experiment name and run directory.

- nerel_preproc  
  Parameters controlling dataset preprocessing.

- grpo_data_prep  
  Parameters controlling GRPO dataset construction.

- grpo_train  
  Hyperparameters and settings for reinforcement learning training.

- grpo_eval  
  Settings controlling the evaluation stage.



### Running the full pipeline

The entire experiment pipeline can be executed using a single command:

```bash
python run_pipeline.py --config grpo_config.yaml --mode grpo
```

or

```bash
python run_pipeline.py --config sft_config.yaml --mode sft
```

Each experiment produces a new run directory inside the `runs` folder.



### Running multiple experiments

Multiple experiments can be scheduled using `run_queue.py`.

Each experiment in the queue can specify a training mode:

- grpo — reinforcement learning
- sft — supervised fine-tuning

Example:

experiments:
  - config: grpo_config.yaml
    mode: grpo
  - config: sft_config.yaml
    mode: sft
    
Experiments are executed sequentially.

## Training visualization

Training dynamics can be analyzed using the script `build_plots.py`.

This script reads the training log file produced during GRPO training. The training log contains periodic metric snapshots produced by the trainer, including:

- reward
- loss
- gradient norm
- learning rate
- generation statistics

The script extracts these values and generates training curves.

The resulting plot includes three graphs:

- Reward curve  
- Loss curve  
- Gradient norm  



## Reproducibility

The project environment is managed using **micromamba**.

To recreate the environment on a clean machine:

1. install micromamba
2. create the environment using `environment.yml`
3. install exact Python package versions from `requirements_lock.txt`



## Future improvements

Although the pipeline is fully functional, several improvements can be made in future iterations.

#### Checkpoint selection using the dev set

Currently evaluation is performed using the latest training checkpoint.  
A more robust approach would evaluate multiple checkpoints on the development set and select the best-performing model.

#### Reward weight calibration

The current reward weights should be treated as an initial heuristic configuration rather than a carefully optimized design.

At the current stage of the project, the weights were chosen manually to provide a reasonable training signal and to make the training pipeline operational.  
They have not yet been systematically tuned through ablation studies or hyperparameter search.

