#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback  # ★ 新增：用于 patience 早停


def load_jsonl_dataset(jsonl_path: str):
    """
    直接读 jsonl，返回 Python list[dict]，每个元素是一条样本：
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
                print(f"[warn] 解析 {jsonl_path} 第 {n_total} 行失败, 跳过: {e}")
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
        help="目前暂时不用这个参数，图像路径已经写在 messages 里，由 Unsloth 自己处理。",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="flare_task5_sft",
        help="wandb 项目名称，用于收集本次 SFT 训练曲线。",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="qwen3vl_sft_debug",
        help="wandb run 的名字，方便区分不同实验。",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["none", "wandb"],
        help="是否把日志上报到 wandb。",
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
        help="避免 4bit/bnb 依赖，先用全精度 + LoRA。",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=0.1,
        help="正式跑的时候可以改大；debug 阶段只看能不能跑起来。",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    # ★ 新增：验证集 batch size 配置
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="验证阶段 per-device batch size。",
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
        default=10,  # ★ debug：只跑 10 步，确认环境 OK
        help=">0 则优先使用 max_steps；= -1 则使用 num_train_epochs。",
    )
    # ★ 新增：patience 早停配置（基于 eval_loss）
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help=(
            "如果设置为正整数，则启用 EarlyStoppingCallback，"
            "当 eval_loss 连续这么多次 eval 都没有下降时提前停止训练。"
            "如果为 None 或 <=0 则关闭早停。"
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 如果启用 wandb，设置项目名环境变量
    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        # team / entity 可以在 sbatch 里通过 WANDB_ENTITY 设置

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
        print("         如果这是在 login 节点上跑的，请忽略；真正训练时请在 GPU 结点上用 sbatch 提交。")

    print("=== Loading Qwen3-VL base model via Unsloth (no 4bit) ===")
    model_and_processors = FastVisionModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,          # ★ 禁用 4bit，避免 bitsandbytes 权重
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",  # 减少显存 / 支持长序列
    )

    # 兼容返回 (model, tokenizer, image_processor) / (model, tokenizer)
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

    # 挂 LoRA 适配器：语言 + 视觉 一起训
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

    # 启用训练模式（Unsloth 会做内部 patch）
    FastVisionModel.for_training(model)

    # === 训练参数：用 SFTConfig + UnslothVisionDataCollator ===
    # max_steps 与 num_train_epochs 二选一（优先使用 max_steps）
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
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # ★ 使用配置的 val batch size
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

        # ★ wandb 相关
        report_to=args.report_to,
        run_name=args.wandb_run_name,

        # ★ 强制用纯 torch AdamW，避免 bitsandbytes optimizer
        optim="adamw_torch",

        # vision 专用：让 Unsloth 自己处理 messages 结构
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,

        # ★ 为 early stopping / best model 做配置
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # 混合精度：按官方推荐，用 is_bf16_supported 判断
    sft_config_kwargs["bf16"] = is_bf16_supported()
    sft_config_kwargs["fp16"] = not is_bf16_supported()
    # DataLoader 多进程，加速一些预处理
    sft_config_kwargs["dataloader_num_workers"] = 4

    if max_steps is not None and max_steps > 0:
        sft_config_kwargs["max_steps"] = max_steps
    if num_train_epochs is not None:
        sft_config_kwargs["num_train_epochs"] = num_train_epochs

    training_args = SFTConfig(**sft_config_kwargs)

    print("=== Building callbacks (EarlyStopping) ===")
    callbacks = None
    if use_early_stopping:
        # 基于 eval_loss 的早停；阈值 0.0 表示只要有更小的 loss 就认为有提升
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

