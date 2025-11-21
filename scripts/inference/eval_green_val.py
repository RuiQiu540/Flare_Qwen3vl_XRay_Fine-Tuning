#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用 SFT 后的 Qwen3-VL 模型在 val_sft.jsonl 上生成报告，
然后用 Qwen2.5 的 GREEN metric 计算在验证集上的性能。

运行：
    python eval_green_val.py
"""

import json
from pathlib import Path

from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import metrics  # QWen2.5VL 的 metrics.py

# ======== 配置区：按你的环境改 ========
CKPT_DIR = "Qwen/Qwen3-VL-4B-Instruct"
VAL_JSONL = "/home/ruiqiu/scratch/FLARE_Task5/val_sft.jsonl"
IMAGE_ROOT = "/home/ruiqiu/scratch/FLARE_Task5"  # 根目录，jsonl 里的 image 是相对这个目录的路径
MAX_NEW_TOKENS = 256
MAX_SAMPLES = None          # 调试用；比如先设 100 只跑 100 条，看没问题后改成 None 跑全量
# ================================


def extract_prompt_and_label(sample):
    """
    从一条 jsonl 样本中抽出：
    - prompt_messages：用户侧的 messages（去掉最后一条 assistant）
    - gt_report：最后一条 assistant 的文本 label
    - image_rel_path：第一条 user message 里的 image 相对路径
    假设结构类似：
    {
      "messages": [
        {"role": "user", "content": [{"type": "image", "image": "train/...png"},
                                     {"type": "text", "text": "...prompt..."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "...report..."}]}
      ]
    }
    """
    messages = sample["messages"]
    assert len(messages) >= 2, "每条样本至少应该包含 user + assistant 两条消息"

    # 最后一条消息假设是 ground truth 的 assistant
    last_msg = messages[-1]
    assert last_msg["role"] == "assistant", "目前假设最后一条 message 是 assistant（label）"

    # 拼接 assistant 文本作为 gt_report
    gt_parts = []
    for c in last_msg.get("content", []):
        if c.get("type") == "text":
            gt_parts.append(c.get("text", ""))
    gt_report = " ".join(gt_parts).strip()

    # 找到第一条 user message 里的 image 路径
    image_rel_path = None
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for c in msg.get("content", []):
            if c.get("type") == "image":
                image_rel_path = c.get("image")
                break
        if image_rel_path is not None:
            break

    if image_rel_path is None:
        raise ValueError("在 messages 里没有找到 image 字段。")

    # prompt_messages = 去掉最后一条 assistant 的所有 messages
    prompt_messages = messages[:-1]

    return prompt_messages, gt_report, image_rel_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    print("[info] Loading model & processor...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT_DIR,
        torch_dtype=torch.bfloat16,
    ).to(device)
    processor = AutoProcessor.from_pretrained(CKPT_DIR)
    model.eval()

    preds = []
    refs = []

    print(f"[info] Loading validation data from: {VAL_JSONL}")
    with open(VAL_JSONL, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if MAX_SAMPLES is not None:
        lines = lines[:MAX_SAMPLES]
        print(f"[info] DEBUG 模式，仅使用前 {len(lines)} 条样本。")

    print(f"[info] Running inference on {len(lines)} validation examples...")

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)

        prompt_messages, gt_report, image_rel_path = extract_prompt_and_label(sample)

        # 打开图像
        image_path = Path(IMAGE_ROOT) / image_rel_path
        image = Image.open(image_path).convert("RGB")

        # 用 PIL 图像替换 messages 里的路径字符串，让 processor 直接吃 PIL
        for msg in prompt_messages:
            if msg.get("role") != "user":
                continue
            for c in msg.get("content", []):
                if c.get("type") == "image":
                    c["image"] = image  # 改成 PIL.Image，而不是原来的 "train/xxx.png"

        # 直接让 processor.apply_chat_template 同时处理文本+图像
        inputs = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,              # 直接返回 token + vision features
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Qwen3-VL 官方示例里会去掉 token_type_ids
        inputs.pop("token_type_ids", None)

        # 把所有张量搬到 GPU 上
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,        # 评估建议关闭采样
                temperature=0.0,
                top_p=1.0,
            )

        pred = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0].strip()

        preds.append(pred)
        refs.append(gt_report)

        if idx % 50 == 0:
            print(f"[info] processed {idx} samples")

    print("[info] Computing metrics on validation set...")

    # 使用 Qwen2.5 的 GREEN + BLEU + clinical efficacy
    green_results = metrics.calculate_green_score(preds, refs)
    bleu = metrics.calculate_bleu_score(preds, refs)
    ce = metrics.calculate_clinical_efficacy_score(preds, refs)

    print("===== GREEN (Qwen2.5) metrics on validation =====")
    for k, v in green_results.items():
        print(f"{k}: {v}")

    print(f"\nBLEU score: {bleu}")
    print(f"Clinical efficacy score: {ce}")


if __name__ == "__main__":
    main()

