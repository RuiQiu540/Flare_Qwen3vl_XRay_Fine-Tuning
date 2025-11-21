#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from copy import deepcopy

import torch
import pandas as pd
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator


def load_jsonl_dataset(jsonl_path: str):
    """和 train_qwen3vl_sft.py / infer_qwen3vl_tiny.py 保持一致，直接读 jsonl。"""
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


def extract_text_from_content(content):
    """
    把 Qwen3-VL 的 content 结构（可能是 str 或 list[dict{type,text}]）抽成纯文本。
    和 infer_qwen3vl_tiny.py 里面保持一致。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and "text" in c:
                    parts.append(c["text"])
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(p for p in parts if p)
    return str(content)


def extract_image_names_from_messages(messages):
    """
    从 messages 里找到所有 image 的路径，取 basename，多个用分号拼接.
    例如: "img1.jpg; img2.jpg"
    """
    names = []
    for m in messages:
        content = m.get("content", [])
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image":
                    img_path = c.get("image", "")
                    if img_path:
                        names.append(os.path.basename(img_path))
    return "; ".join(names)


def build_prompt_and_target_from_messages(messages):
    """
    从 messages 里拆：
    - prompt_messages: 所有 assistant 之前的轮次（一般就是 user/system）
    - target_text: 最后一个 assistant 的文本（label）
    """
    if not messages:
        return [], ""

    # 找最后一个 assistant，当作 label
    target_text = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            target_text = extract_text_from_content(m.get("content", ""))
            break

    # prompt = 第一个 assistant 之前的所有 messages
    prompt_messages = []
    for m in messages:
        if m.get("role") == "assistant":
            break
        prompt_messages.append(m)

    if not prompt_messages:
        prompt_messages = messages  # 兜底

    return prompt_messages, target_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="validation jsonl，比如 /home/ruiqiu/scratch/FLARE_Task5/val_sft.jsonl",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="要推理的模型 checkpoint 目录，比如 RL 的 checkpoint-700",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="最多抽多少个样本写进表，如果 <=0 就用全部",
    )
    parser.add_argument(
        "--output_xlsx",
        type=str,
        required=True,
        help="输出 Excel 路径，比如 /home/.../val_examples_rl.xlsx",
    )
    return parser.parse_args()


def load_model_and_tokenizer(ckpt_dir: str, max_seq_length: int = 2048):
    """
    和 infer_qwen3vl_tiny.py 一样，用 FastVisionModel.from_pretrained.
    """
    print(f"[model] loading from: {ckpt_dir}")
    model_and_processors = FastVisionModel.from_pretrained(
        ckpt_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )

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

    model.eval()
    return model, tokenizer, image_processor


@torch.no_grad()
def generate_answer_for_sample(sample, model, tokenizer, device, max_new_tokens, max_seq_length):
    """
    给一个 sample，用当前模型生成回答（只看 prompt，不看 label）。
    返回 Question, GT Answer, 以及模型输出。
    """
    data_collator = UnslothVisionDataCollator(model, tokenizer)

    messages = sample.get("messages", [])
    prompt_messages, target_text = build_prompt_and_target_from_messages(messages)

    # 抽 user 的文本当 Question
    prompt_texts = []
    for m in prompt_messages:
        if m.get("role") == "user":
            prompt_texts.append(extract_text_from_content(m.get("content", "")))
    prompt_text = "\n".join(p for p in prompt_texts if p)

    # 只保留 prompt 的 messages 送进 collator
    infer_sample = deepcopy(sample)
    infer_sample["messages"] = prompt_messages

    batch = data_collator([infer_sample])
    batch.pop("labels", None)
    batch = {k: v.to(device) for k, v in batch.items()}

    input_ids = batch["input_ids"]

    generated_ids = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    if generated_ids.shape[1] > input_ids.shape[1]:
        new_tokens = generated_ids[:, input_ids.shape[1]:]
    else:
        new_tokens = generated_ids

    if tokenizer is not None:
        output_text = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0]
    else:
        output_text = "<no tokenizer available>"

    return prompt_text, target_text, output_text


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] using: {device}")

    # 读数据
    data = load_jsonl_dataset(args.jsonl)
    if args.num_samples > 0:
        data = data[:args.num_samples]
    print(f"[info] num_samples to process = {len(data)}")

    # 只加载一个模型（比如 RL checkpoint）
    model, tokenizer, _ = load_model_and_tokenizer(
        args.ckpt_dir,
        max_seq_length=args.max_seq_length,
    )
    model.to(device)

    rows = []

    for idx, sample in enumerate(data):
        messages = sample.get("messages", [])
        image_name = extract_image_names_from_messages(messages)

        q_text, ans_text, rl_out = generate_answer_for_sample(
            sample,
            model,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_length,
        )

        rows.append({
            "ImageName": image_name,
            "Question": q_text,
            "Answer": ans_text,
            "RL": rl_out,
        })

        if (idx + 1) % 10 == 0:
            print(f"[info] processed {idx + 1} / {len(data)}")

    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(args.output_xlsx)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_excel(args.output_xlsx, index=False)
    print(f"[done] saved sheet to: {args.output_xlsx}")


if __name__ == "__main__":
    main()

