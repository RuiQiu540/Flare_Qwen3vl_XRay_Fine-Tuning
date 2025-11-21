# FLARE Task 5 X-ray Report Generation (Qwen3-VL-4B, Unsloth)

This repository contains scripts to prepare the FLARE Task 5 X-ray dataset, run supervised fine-tuning (SFT) and GRPO-based reinforcement learning (RL) on **Qwen3-VL-4B-Instruct**, and evaluate models with GREEN-based metrics.

## 0. Python environment and dependencies

The `requirements.txt` file in this repository is a *full freeze* of the
environment used on the Compute Canada Alliance cluster. It contains many
cluster-specific wheels (e.g., packages with a `+computecanada` suffix and
a local `pyarrow @ file://...` path), as well as a large number of
transitive dependencies that were pulled in automatically when installing
major libraries such as PyTorch, Transformers, Unsloth, RadGraph, etc.

For reproducibility, **you do not need to install every single package in
that freeze individually**. It is sufficient to:

1. Create a new environment using `env/unsloth_flare.yml`, or
2. Install the packages listed in `env/minimal_requirements.txt`.

These files list the *direct dependencies* that are actually required to
run the data preparation, SFT, RL, and evaluation scripts. All other small
packages that appear in `requirements.txt` are automatically pulled in by
these main libraries and do not need to be pinned or managed manually.

```bash
conda env create -f env/unsloth_flare.yml
conda activate unsloth_flare
```

## 1. Dataset

We use the FLARE Task 5 X-ray dataset released by the FLARE-MedFM team.

* **Training set (X-ray):**
  `FLARE-Task5-MLLM-2D / training / Xray`
  [https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-2D/tree/main/training/Xray](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-2D/tree/main/training/Xray)

* **Validation set (IU X-Ray subset):**
  `FLARE-Task5-MLLM-2D / validation-public / Xray / IU_XRay`
  [https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-2D/tree/main/validation-public/Xray/IU_XRay](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-2D/tree/main/validation-public/Xray/IU_XRay)

> **Access note:**
> These subsets are hosted on Hugging Face as part of the FLARE-MedFM initiative and are **gated**.
> You need to be logged in to Hugging Face and accept the dataset terms in order to download them.
> The authors of this repository do **not** redistribute the raw images or reports;
> please obtain the data directly from the original source and follow its license and usage policy.

### Local directory layout

After downloading the dataset from Hugging Face, we assume the following
directory structure:

```text
/path/to/FLARE_Task5/
  training/Xray/
    images/
      <study_id>/<series_id>/*.png
    reports/
      <study_id>.txt
  validation-public/Xray/IU_XRay/
    images/
      <study_id>/<series_id>/*.png
    reports/
      <study_id>.txt
```

## 2. Data preprocessing

We first convert the raw FLARE Task 5 X-ray data into JSONL files that can be
directly consumed by the SFT and RL scripts.

The script `prepare_flare_task5_jsonl.py` expects the following arguments:

* `--data_root`
  Root directory of the FLARE Task 5 data. Under this root, we assume the
  following structure (after unzipping):

  ```text
  <data_root>/
    train/
      training/Xray/IU_XRay/...
      training/Xray/boneresorption/...
      training/Xray/chestdr/...
      training/Xray/dental/...
      training/Xray/periapical/...
    val/
      validation-public/Xray/IU_XRay/...
  ```

```bash
cd /path/to/this/repo

python scripts/prepare_flare_task5_jsonl.py \
  --data_root    /path/to/FLARE_Task5 \
  --train_output data/train_sft.jsonl \
  --val_output   data/val_sft.jsonl
```

## 3. Supervised Fine-tuning (SFT)

We perform supervised fine-tuning (SFT) of **Qwen3-VL-4B-Instruct** using
the Unsloth vision SFT pipeline implemented in `train_qwen3vl_sft.py`.

### Script and key arguments

The main arguments of `train_qwen3vl_sft.py` are:

* `--train_jsonl`, `--val_jsonl`
  Paths to the training and validation JSONL files produced in the
  preprocessing step (see Section 2).

* `--image_root`
  Root directory of the FLARE Task 5 data. Image paths referenced in the
  JSONL file are resolved relative to this root (Unsloth internally
  handles the multimodal messages).

* `--model_name`
  Base VLM to finetune. We use `unsloth/Qwen3-VL-4B-Instruct`.

* `--output_dir`
  Directory where SFT checkpoints and the final LoRA adapter are saved.

* `--max_seq_length`
  Maximum total sequence length (vision + text). For FLARE Task 5 we use
  a large value (e.g. 12000) to accommodate long vision sequences.

* Optimization & logging:
  `--per_device_train_batch_size`, `--per_device_eval_batch_size`,
  `--gradient_accumulation_steps`, `--learning_rate`, `--warmup_ratio`,
  `--weight_decay`, `--logging_steps`, `--save_steps`, `--eval_steps`,
  `--max_steps`.

* Early stopping (patience):
  `--early_stopping_patience` (int, optional). If set to a positive
  integer, an `EarlyStoppingCallback` based on `eval_loss` is enabled:
  when validation loss does not improve for `early_stopping_patience`
  consecutive evaluation intervals, training is stopped early and the
  best checkpoint is kept.

### Example SFT command

```bash
python scripts/train_qwen3vl_sft.py \
  --train_jsonl data/train_sft.jsonl \
  --val_jsonl   data/val_sft.jsonl \
  --image_root  /path/to/FLARE_Task5 \
  --output_dir  checkpoints/SFT_full \
  --model_name  unsloth/Qwen3-VL-4B-Instruct \
  --max_seq_length 12000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 100 \
  --max_steps 1500 \
  --early_stopping_patience 5 \
  --wandb_project flare_task5_sft \
  --wandb_run_name qwen3vl_sft_full \
  --report_to wandb
```

## 4. Reinforcement Learning (GRPO)

After SFT, we apply reinforcement learning using the GRPO algorithm
(`trl.GRPOTrainer`) on top of the SFT LoRA checkpoint, implemented in
`rl_qwen3vl.py`.

### Script and key arguments

The CLI arguments are:

* `--train_jsonl`
  Path to the RL training JSONL (we reuse `train_sft.jsonl`, but the
  script reshapes it into GRPO format internally).

* `--sft_checkpoint`
  Path to the SFT checkpoint (e.g. `checkpoints/SFT_full/checkpoint-550`).

* `--output_dir`
  Directory to save the RL-finetuned model and processors.

* `--max_steps`
  Maximum number of GRPO training steps.

* `--batch_size`, `--grad_accum`
  Per-device batch size and gradient accumulation steps for GRPO.

* `--lr`
  Learning rate for GRPO.

* `--max_seq_length` / `--max_completion_length`
  Maximum prompt length (vision + text) and maximum generated completion
  length used by GRPO. We typically keep `max_seq_length` consistent with
  SFT (e.g. 12000) and limit `max_completion_length` (e.g. 256) to
  control runtime and prevent very long generations.

* Logging:
  `--wandb_project`, `--wandb_run_name`, `--report_to` (e.g. `wandb`).

The GRPO configuration enforces:

* `num_generations=2` per prompt;
* BF16 where supported (`bf16=is_bf16_supported()`);
* Checkpointing every `save_steps=100`;
* `optim="adamw_torch"` to avoid bitsandbytes.

### Reward function design

The reward is implemented as `combined_reward` inside `rl_qwen3vl.py`.
It combines multiple components:

1. **Formatting reward (`format_reward_func`)**
   Encourages a structured report with separate **Findings** and
   **Impression** sections, penalizes extremely short outputs and obvious
   debugging artefacts, and discourages copy-paste identical sections.

2. **ROUGE-L reward (`rouge_reward_func`)**
   Measures lexical overlap between the generated report and the
   reference report.

3. **Clinical factuality reward (`clinical_reward_func`)**
   Uses **F1-RadGraph** to score entity/relation correctness (RG_ER) for
   chest X-ray style reports. Multichoice and numeric questions are
   explicitly skipped to avoid misusing RadGraph for non-radiology text.

4. **Task-specific logic in `combined_reward`:**

   * **Multiple-choice questions (A/B/C/D/E/F)**
     The reference label is a single letter; if the model output
     contains the correct option as a standalone token, the reward is
     set to **1.0** (full credit). Wrong or missing options receive a
     small penalty (-0.2).
   * **Numeric questions (e.g. ABR%)**
     The reference is parsed as a float; the model’s first numeric value
     is extracted and the reward depends on the absolute error (small
     error → high reward, large error → penalty), with additional
     penalty for clearly invalid ranges.
   * **Free-text chest X-ray reports**
     The reward is a weighted combination of:
     `0.7 * clinical_reward (RadGraph) + 0.2 * format_reward + 0.1 * ROUGE-L`.

Because different datasets and question types contribute different reward
patterns (e.g., discrete 1.0/-0.2 for MC vs. continuous RadGraph scores
for reports), **reward curves are expected to be noisy and oscillatory**,
even when training is healthy.

### Example RL command

```bash
python scripts/rl_qwen3vl.py \
  --train_jsonl    data/train_sft.jsonl \
  --sft_checkpoint checkpoints/SFT_full/checkpoint-550 \
  --output_dir     checkpoints/RL_grpo_bs5_steps800_len12k \
  --max_steps 800 \
  --batch_size 5 \
  --grad_accum 1 \
  --lr 5e-6 \
  --max_seq_length 12000 \
  --max_completion_length 256 \
  --wandb_project flare_task5_rl \
  --wandb_run_name qwen3vl_rl_bs5_len12k \
  --report_to wandb
```

All hyper-parameters for the main experiment are also stored in `configs/sft_flare_task5.yaml`.

## 5 Inference

For qualitative inspection, we provide `make_val_sheet.py`, which:

* Runs inference on a JSONL dataset with a given checkpoint, and
* Writes an Excel file containing:

  * `ImageName` (derived from the image path),
  * `Question` (user text),
  * `Answer` (ground-truth assistant text),
  * `RL` (the model’s generated output).

This is useful for manual review of SFT/RL behavior.

#### Script arguments

`make_val_sheet.py` exposes the following CLI arguments:

```bash
python scripts/make_val_sheet.py \
  --jsonl PATH_TO_JSONL \
  --ckpt_dir CKPT_DIR \
  --max_seq_length MAX_SEQ_LEN \
  --max_new_tokens MAX_NEW_TOKENS \
  --num_samples NUM_SAMPLES \
  --output_xlsx OUTPUT_XLSX
```

* `--jsonl` (str, required)
  Path to the JSONL file to sample from (e.g. `data/val_sft.jsonl`).

* `--ckpt_dir` (str, required)
  Checkpoint directory for the model to be evaluated
  (e.g. `checkpoints/RL_grpo_bs5_steps800_len12k/checkpoint-700`).

* `--max_seq_length` (int, default: 2048)
  Maximum sequence length used when loading the model via
  `FastVisionModel.from_pretrained`. For FLARE Task 5, we typically
  match the SFT/RL setting (e.g. 12000).

* `--max_new_tokens` (int, default: 256)
  Maximum number of tokens to generate per example.

* `--num_samples` (int, default: 50)
  Maximum number of examples to include in the Excel sheet.
  If `--num_samples <= 0`, the script uses all samples in the JSONL.

* `--output_xlsx` (str, required)
  Path to the output Excel file (e.g. `results/val_examples_rl.xlsx`).

Internally, the script:

* Loads samples from the given JSONL file.
* Loads the model (and tokenizer/image processor) from `ckpt_dir`
  using `FastVisionModel.from_pretrained`.
* For each sample:

  * Builds the prompt from all messages up to the last assistant turn,
  * Uses `UnslothVisionDataCollator` to create a batch,
  * Calls `model.generate(...)` to obtain the model output,
  * Extracts:

    * `ImageName` (from the image path),
    * `Question` (user content),
    * `Answer` (ground-truth assistant content),
    * `RL` (generated output).
* Saves all rows into an Excel file with columns:

  * `ImageName`
  * `Question`
  * `Answer`
  * `RL`

#### Example command

To generate a sheet with 200 validation examples for a given RL checkpoint:

```bash
python scripts/make_val_sheet.py \
  --jsonl        data/val_sft.jsonl \
  --ckpt_dir     checkpoints/RL_grpo_bs5_steps800_len12k/checkpoint-700 \
  --max_seq_length 12000 \
  --max_new_tokens 256 \
  --num_samples 200 \
  --output_xlsx results/val_examples_rl.xlsx
```

This will create `results/val_examples_rl.xlsx`, which you can open in
Excel to manually compare the ground-truth reports vs. model outputs for
each image/question pair.

## 6. Evaluation

### GREEN-based quantitative evaluation

For quantitative evaluation, we use the **GREEN score** implementation
(`metrics.py`) adapted from the FLARE25 Qwen2.5VL repository by **Leo Yin**.

We gratefully acknowledge Leo Yin and the FLARE-MedFM team for making
their implementation available:

* Original repository:
  [https://github.com/medfm-flare/FLARE25-QWen2.5VL/tree/main](https://github.com/medfm-flare/FLARE25-QWen2.5VL/tree/main)

The script `eval_green_val.py` runs the model on a validation JSONL file
and computes:

* GREEN metrics (overall + component scores),
* BLEU score,
* Clinical efficacy score.

#### Configuration

`eval_green_val.py` is configured via a small block at the top of the script:

```python
# ======== Config: modify these for your environment ========
CKPT_DIR = "Qwen/Qwen3-VL-4B-Instruct"
VAL_JSONL = "/path/to/data/val_sft.jsonl"
IMAGE_ROOT = "/path/to/FLARE_Task5"
MAX_NEW_TOKENS = 256
MAX_SAMPLES = None   # e.g., 100 for a quick debug run; None for full val
# ============================================================
```

**Config fields**

* **`CKPT_DIR`**
  Path to the model checkpoint to evaluate. This can be:

  * the base model (e.g. `Qwen/Qwen3-VL-4B-Instruct`), or
  * an SFT / RL checkpoint directory produced by this project
    (e.g. `checkpoints/SFT_full/checkpoint-550` or
    `checkpoints/RL_grpo_bs5_steps800_len12k/checkpoint-700`).

* **`VAL_JSONL`**
  Path to the validation JSONL file (e.g. `data/val_sft.jsonl`).

* **`IMAGE_ROOT`**
  Root directory for images; the image fields in the JSONL file are
  resolved relative to this directory.

* **`MAX_NEW_TOKENS`**
  Maximum number of tokens to generate per example.

* **`MAX_SAMPLES`**
  If set to an integer, only the first `MAX_SAMPLES` examples from
  `VAL_JSONL` are evaluated (useful for quick debugging). If `None`,
  the full validation set is evaluated.

Internally, the script:

1. Loads the model and processor from `CKPT_DIR`.

2. Iterates over each example in `VAL_JSONL`, reconstructs the chat-style
   prompt + image, and generates a report.

3. Accumulates model predictions and reference reports.

4. Calls `metrics.calculate_green_score`, `metrics.calculate_bleu_score`,
   and `metrics.calculate_clinical_efficacy_score` to compute metrics.

#### Running evaluation

After editing the config block, run:

```bash
python scripts/eval_green_val.py
```

At the end, the script prints something like:

```text
===== GREEN (Qwen2.5) metrics on validation =====
overall_mean: ...
overall_std: ...
entity_matching_mean: ...
...
BLEU score: ...
Clinical efficacy score: ...
```

These are the GREEN component-wise statistics and auxiliary metrics for
the selected checkpoint on the validation set.

## Citation

If you use this repository or report results based on it, please cite the FLARE Task 5 dataset:

> FLARE-MedFM Team. *FLARE-Task5-MLLM-2D (Xray)*.  
> Training split: https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-2D/tree/main/training/Xray  
> Validation split (IU X-Ray): https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-2D/tree/main/validation-public/Xray/IU_XRay  

All experiments in this repository were run on the **Alliance (Compute Canada) Fir cluster**.


## Acknowledgments

- **Dataset & Benchmark** – We thank the **FLARE-MedFM** team for releasing the FLARE-Task5-MLLM-2D X-ray dataset and benchmark.
- **GREEN metrics implementation** – We use and adapt scripts provided by **Leo Yin** from the FLARE25 QWen2.5VL repository:  
  https://github.com/medfm-flare/FLARE25-QWen2.5VL/tree/main
- **Compute resources** – This work was conducted on the **Alliance (Compute Canada) Fir cluster**, whose support and infrastructure are gratefully acknowledged.
- **Scientific guidance** – This project was guided by **Dr. Jun Ma**  
  (https://www.linkedin.com/in/jun-ma-867b34224/?originalSubdomain=ca).
- **Idea sharing & discussions** – We thank **Ryan Khalloqi** for sharing ideas and helpful discussions related to this project:  
  https://www.linkedin.com/in/ryan-w-khalloqi/
