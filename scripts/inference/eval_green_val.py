#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
from pathlib import Path

from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import metrics  # QWen2.5VL 的 metrics.py

# config
CKPT_DIR = "Qwen/Qwen3-VL-4B-Instruct"
VAL_JSONL = "/home/ruiqiu/scratch/FLARE_Task5/val_sft.jsonl"
IMAGE_ROOT = "/home/ruiqiu/scratch/FLARE_Task5"
MAX_NEW_TOKENS = 256
MAX_SAMPLES = None         


def extract_prompt_and_label(sample):

    messages = sample["messages"]
    assert len(messages) >= 2, "every sample should contain 2 messages user + assistant "

    # assuming the last message is ground truth assistant
    last_msg = messages[-1]
    assert last_msg["role"] == "assistant", "assume last message is assistant（label）"

    # merge assistant text as gt_report
    gt_parts = []
    for c in last_msg.get("content", []):
        if c.get("type") == "text":
            gt_parts.append(c.get("text", ""))
    gt_report = " ".join(gt_parts).strip()

    # find an image path in user message
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
        raise ValueError("can't find messages in image。")


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
        print(f"[info] DEBUG mode，only use {len(lines)} samples")

    print(f"[info] Running inference on {len(lines)} validation examples...")

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)

        prompt_messages, gt_report, image_rel_path = extract_prompt_and_label(sample)

        # load image
        image_path = Path(IMAGE_ROOT) / image_rel_path
        image = Image.open(image_path).convert("RGB")

        # use PIL image to replace path string in messages，let processor go PIL
        for msg in prompt_messages:
            if msg.get("role") != "user":
                continue
            for c in msg.get("content", []):
                if c.get("type") == "image":
                    c["image"] = image 


        inputs = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,             
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Qwen3-VL official sample may not have token_type_ids
        inputs.pop("token_type_ids", None)

        # move all tensors to GPU 
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
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

