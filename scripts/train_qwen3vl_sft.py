#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback


def load_jsonl_dataset(jsonl_path: str):
    """
    read jsonl，return Python list[dict]，each element is a sample：
    {
      "messages": [...],  # Qwen3-VL 的 chat 格式
    }
    """
    data = []
    n_total = 0
    n_bad = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"[warn] analyze {jsonl_path} line {n_total} fail, skip: {e}")
                n_bad += 1

    print(f"[data] loaded {len(data)} samples from {jsonl_path}, bad lines = {n_bad}")
    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="/home/ruiqiu/scratch/FLARE_Task5/train_sft.jsonl",
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        default="/home/ruiqiu/scratch/FLARE_Task5/val_sft.jsonl",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/home/ruiqiu/scratch/FLARE_Task5",
        help="dont need this parameter，image path is within messages，Unsloth will do",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="flare_task5_sft",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="qwen3vl_sft_debug",
        help="wandb run name",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["none", "wandb"],
        help="opload to wandb or not",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ruiqiu/scratch/FLARE_Task5/SFT_checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen3-VL-4B-Instruct",
        help="avoid 4bit/bnb depencency，full precision + LoRA",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=0.1,
        help="keep small in debug",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="verifying phase per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help=">0 prior to use max_steps；= -1 then use num_train_epochs",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help=(
            "if it is positive integer，use EarlyStoppingCallback，"
            "when eval_loss stop decrease for this number, stop"
            "if this is None then it will not stop early"
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project

    print("=== Loading datasets from JSONL (pure Python list) ===")
    train_dataset = load_jsonl_dataset(args.train_jsonl)
    eval_dataset = load_jsonl_dataset(args.val_jsonl)
    print(f"[data] train: {len(train_dataset)} examples, val: {len(eval_dataset)} examples")

    print("=== Torch & CUDA check ===")
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    ok = torch.cuda.is_available()
    print("torch.cuda.is_available():", ok)
    if ok:
        print("GPU count:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: torch.cuda.is_available() is False.")
        print("        use sbatch to submit")

    print("=== Loading Qwen3-VL base model via Unsloth (no 4bit) ===")
    model_and_processors = FastVisionModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,          
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )

    # robust to return (model, tokenizer, image_processor) / (model, tokenizer)
    if isinstance(model_and_processors, tuple):
        if len(model_and_processors) == 3:
            model, tokenizer, image_processor = model_and_processors
        elif len(model_and_processors) == 2:
            model, tokenizer = model_and_processors
            image_processor = None
        else:
            model = model_and_processors[0]
            tokenizer = model_and_processors[1] if len(model_and_processors) > 1 else None
            image_processor = None
    else:
        model = model_and_processors
        tokenizer = None
        image_processor = None

    # apply LoRA adapter：text + vision together
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # apply training（Unsloth will do patch）
    FastVisionModel.for_training(model)

    # === training parameter：apply SFTConfig + UnslothVisionDataCollator ===
    # max_steps and num_train_epochs choose one（prior to use max_steps）
    if args.max_steps and args.max_steps > 0:
        max_steps = args.max_steps
        num_train_epochs = None
    else:
        max_steps = -1
        num_train_epochs = args.num_train_epochs

    print("=== Building SFTConfig ===")
    use_early_stopping = (
        args.early_stopping_patience is not None and args.early_stopping_patience > 0
    )

    sft_config_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        max_seq_length=args.max_seq_length,

        #  wandb related
        report_to=args.report_to,
        run_name=args.wandb_run_name,

        # force to use pure torch AdamW，avoid bitsandbytes optimizer
        optim="adamw_torch",

        # vision only：let Unsloth deal messages structure
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,

        # config for early stopping / best model
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # mixed precision：according to official recommanded，apply is_bf16_supported to judge
    sft_config_kwargs["bf16"] = is_bf16_supported()
    sft_config_kwargs["fp16"] = not is_bf16_supported()
    # DataLoader multi processor to accelerate
    sft_config_kwargs["dataloader_num_workers"] = 4

    if max_steps is not None and max_steps > 0:
        sft_config_kwargs["max_steps"] = max_steps
    if num_train_epochs is not None:
        sft_config_kwargs["num_train_epochs"] = num_train_epochs

    training_args = SFTConfig(**sft_config_kwargs)

    print("=== Building callbacks (EarlyStopping) ===")
    callbacks = None
    if use_early_stopping:
        # based on eval_loss early stop；threshold 0.0 means only smaller loss will be counted as progress
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        ]
        print(f"[early stopping] enabled with patience = {args.early_stopping_patience}")
    else:
        print("[early stopping] disabled (no patience or <=0)")

    print("=== Building SFTTrainer ===")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=training_args,
        callbacks=callbacks,
    )

    print("=== Start training ===")
    trainer.train()

    print("=== Saving final adapter & processors ===")
    trainer.save_model(args.output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_dir)
    if image_processor is not None:
        try:
            image_processor.save_pretrained(args.output_dir)
        except Exception as e:
            print(f"[warn] Failed to save image_processor: {e}")

    print(f"[done] SFT finished. Checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

