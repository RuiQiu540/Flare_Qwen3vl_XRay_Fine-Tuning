#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_flare_task5_jsonl.py

功能：
1. 在 FLARE_Task5 数据集根目录下，递归解压所有 .zip（尤其是 imagesTr.zip / imagesVal.zip）
2. 从各个 *_questions_train.json / IU_XRay_all_*.json 中提取
   - 图像文件名
   - 问题文本
   - 回答文本
3. 生成适用于 Qwen3-VL-4B 的 SFT 格式 jsonl：
   每行一个样本，包含 messages -> [user(image+text), assistant(text)]

使用示例：
python prepare_flare_task5_jsonl.py \
    --data_root /scratch/FLARE_Task5 \
    --train_output /scratch/FLARE_Task5/train_sft.jsonl \
    --val_output   /scratch/FLARE_Task5/val_sft.jsonl
"""

import argparse
import json
import os
import zipfile
from typing import List, Dict, Any, Optional
import glob


# ------------------------ 工具函数：解压所有 zip ------------------------ #

def unzip_all_zips(data_root: str) -> None:
    """递归解压 data_root 下所有 .zip 文件.
    对于 imagesTr.zip / imagesVal.zip，默认解压到同级目录下的 imagesTr / imagesVal.
    若目标目录存在且非空，则跳过。
    """
    print(f"[unzip] 扫描并解压 {data_root} 下的 .zip 文件...")
    for root, dirs, files in os.walk(data_root):
        for fname in files:
            if not fname.lower().endswith(".zip"):
                continue
            zip_path = os.path.join(root, fname)
            # 去掉 .zip
            base = fname[:-4]
            dest_dir = os.path.join(root, base)
            if os.path.isdir(dest_dir) and os.listdir(dest_dir):
                print(f"[unzip] 已存在非空目录，跳过: {dest_dir}")
                continue
            print(f"[unzip] 解压 {zip_path} -> {dest_dir}")
            os.makedirs(dest_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
    print("[unzip] 解压完成。")


# ------------------------ 工具函数：从 json 里取样本列表 ------------------------ #

def load_samples_from_json(json_path: str) -> List[Dict[str, Any]]:
    """根据不同结构尝试从 json 文件中取出样本列表."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        # 常见几种字段名
        for key in ["data", "samples", "annotations", "items", "questions"]:
            v = raw.get(key, None)
            if isinstance(v, list):
                return v

    raise ValueError(f"无法从 {json_path} 中推断样本列表结构，请手动检查该 json 格式。")


# ------------------------ 工具函数：猜字段名 ------------------------ #

def find_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """在字典 d 中根据候选列表 candidates 猜一个最匹配的 key."""
    if not isinstance(d, dict):
        return None

    keys = list(d.keys())
    lower_map = {k.lower(): k for k in keys}

    # 1. 先尝试完全匹配（大小写都试）
    for c in candidates:
        if c in d:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    # 2. 再尝试子串匹配（例如 "question_text" 里包含 "question"）
    for k in keys:
        kl = k.lower()
        for c in candidates:
            if c.lower() in kl:
                return k

    return None

def extract_image_names(ex: Dict[str, Any]) -> List[str]:
    """从单条样本 ex 中猜测图像文件名列表（支持一张或多张图像）."""
    img_key_candidates = [
        "image", "image_path", "image_name", "img_name",
        "img", "imageid", "image_id", "file_name", "filename", "path", "imagename"
    ]
    key = find_key(ex, img_key_candidates)
    value = ex.get(key) if key is not None else None

    # 1) 直接是字符串：单张图
    if isinstance(value, str):
        return [value]

    # 2) 是 list：多张图，比如 ["a.png", "b.png"]
    if isinstance(value, list):
        names: List[str] = []
        for v in value:
            if isinstance(v, str):
                names.append(v)
            elif isinstance(v, dict):
                # 例如 {"file_name": "xxx.png"}
                k2 = find_key(v, img_key_candidates)
                if k2 is not None and isinstance(v[k2], str):
                    names.append(v[k2])
        if names:
            return names

    # 3) 是 dict：里面再包了一层
    if isinstance(value, dict):
        k2 = find_key(value, img_key_candidates)
        v2 = value.get(k2) if k2 is not None else None
        if isinstance(v2, str):
            return [v2]
        if isinstance(v2, list):
            names: List[str] = []
            for v in v2:
                if isinstance(v, str):
                    names.append(v)
                elif isinstance(v, dict):
                    k3 = find_key(v, img_key_candidates)
                    if k3 is not None and isinstance(v[k3], str):
                        names.append(v[k3])
            if names:
                return names

    raise ValueError(f"无法从样本中推断图像字段: {ex}")



def extract_text_field(ex: Dict[str, Any], kind: str) -> str:
    """从样本中提取 question / answer."""
    if kind == "question":
        cand = ["question", "Question", "instruction", "query", "q", "prompt"]
    elif kind == "answer":
        cand = ["answer", "Answer", "gt_answer", "response", "label",
                "A", "target", "output", "text"]
    else:
        raise ValueError(f"kind 必须是 question 或 answer, 现在是 {kind}")

    key = find_key(ex, cand)
    value = ex.get(key) if key is not None else None

    # 1) 字符串，直接用
    if isinstance(value, str):
        return value

    # 2) 回归任务：数值型答案（int / float / bool）
    if kind == "answer" and isinstance(value, (int, float, bool)):
        # 可以按需格式化，比如保留 4 位小数：
        # return f"{float(value):.4f}"
        return str(value)
        # 3) list：多句 / 多标签
    if isinstance(value, list):
        parts = []
        for v in value:
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                tkey = find_key(v, ["text", "Text", "answer", "Answer"])
                if tkey is not None and isinstance(v[tkey], str):
                    parts.append(v[tkey])
        if parts:
            return " ".join(parts)

    raise ValueError(f"无法从样本中提取 {kind}: {ex}")



def resolve_image_path(img_dir: str, image_name: str) -> Optional[str]:
    """根据 image_name 和 img_dir 推断图像的绝对路径，尽量兼容."""
    # 已经是绝对路径就直接用
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name

    # 直接拼接
    candidate = os.path.join(img_dir, image_name)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # 有些 json 里可能已经带了子目录，比如 "imagesTr/xxx.jpg"
    candidate2 = os.path.join(img_dir, os.path.basename(image_name))
    if os.path.exists(candidate2):
        return os.path.abspath(candidate2)

    # 最后兜底，用 glob 在 img_dir 下递归搜索
    matches = glob.glob(os.path.join(img_dir, "**", os.path.basename(image_name)), recursive=True)
    if matches:
        return os.path.abspath(matches[0])

    return None


# ------------------------ 主转换函数 ------------------------ #

def convert_dataset(
    json_path: str,
    img_dir: str,
    out_f,
    split: str,
    dataset_name: str,
) -> None:
    """把某个子数据集 (一个 json + 一个图像目录) 转成 Qwen jsonl 写入 out_f."""
    print(f"[convert] 处理 {split} 集: {dataset_name}")
    print(f"          json: {json_path}")
    print(f"          img_dir: {img_dir}")

    if not os.path.isfile(json_path):
        print(f"[warn] 找不到 json 文件，跳过: {json_path}")
        return
    if not os.path.isdir(img_dir):
        print(f"[warn] 找不到图像目录，跳过: {img_dir}")
        return

    samples = load_samples_from_json(json_path)
    print(f"[convert] 读入样本数: {len(samples)}")

    n_ok = 0
    n_skip = 0
    for ex in samples:
        try:
            # 支持一条样本多张图
            image_names = extract_image_names(ex)
            question = extract_text_field(ex, "question")
            answer = extract_text_field(ex, "answer")

            img_paths: List[str] = []
            for name in image_names:
                p = resolve_image_path(img_dir, name)
                if p is not None:
                    img_paths.append(p)

            if not img_paths:
                print(f"[warn] 找不到任何对应图像: {image_names} (json={json_path})，该样本跳过")
                n_skip += 1
                continue

            # 构造 Qwen3-VL 的 content
            # img_paths 可能长度为 1（单图）或 >1（多图），统一写成多个 {"type": "image"}
            user_content = []
            for p in img_paths:
                user_content.append({"type": "image", "image": p})  # ★ 用 "image" 字段

            # 再加上文字问题
            user_content.append({"type": "text", "text": question})

            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer},
                        ],
                    },
                ]
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1


        except Exception as e:
            print(f"[warn] 处理单条样本时出错，跳过: {e}")
            n_skip += 1


    print(f"[convert] {dataset_name} 转换完成: 写入 {n_ok} 条, 跳过 {n_skip} 条。")


# ------------------------ main ------------------------ #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="FLARE_Task5 根目录，例如 /scratch/FLARE_Task5")
    parser.add_argument("--train_output", type=str, required=True,
                        help="训练集输出 jsonl 路径")
    parser.add_argument("--val_output", type=str, required=True,
                        help="验证集输出 jsonl 路径")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)

    # 1. 解压所有 .zip
    unzip_all_zips(data_root)

    # 2. 定义各子数据集（相对 data_root 的路径）
    DATASETS = [
        # ---- train ----
        {
            "split": "train",
            "name": "IU_XRay_train",
            "json_rel": "train/training/Xray/IU_XRay/IU_XRay_all_train.json",
            "img_dir_rel": "train/training/Xray/IU_XRay/imagesTr",
        },
        {
            "split": "train",
            "name": "boneresorption",
            "json_rel": "train/training/Xray/boneresorption/boneresorption_questions_train.json",
            "img_dir_rel": "train/training/Xray/boneresorption/imagesTr",
        },
        {
            "split": "train",
            "name": "chestdr",
            "json_rel": "train/training/Xray/chestdr/chestdr_questions_train.json",
            "img_dir_rel": "train/training/Xray/chestdr/imagesTr",
        },
        {
            "split": "train",
            "name": "dental",
            "json_rel": "train/training/Xray/dental/dental_questions_train.json",
            "img_dir_rel": "train/training/Xray/dental/imagesTr",
        },
        {
            "split": "train",
            "name": "periapical",
            "json_rel": "train/training/Xray/periapical/periapical_questions_train.json",
            "img_dir_rel": "train/training/Xray/periapical/imagesTr",
        },
        # ---- val ----
        {
            "split": "val",
            "name": "IU_XRay_val",
            "json_rel": "val/validation-public/Xray/IU_XRay/IU_XRay_all_val.json",
            "img_dir_rel": "val/validation-public/Xray/IU_XRay/imagesVal",
        },
    ]

    # 3. 打开输出文件
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_output), exist_ok=True)

    train_f = open(args.train_output, "w", encoding="utf-8")
    val_f = open(args.val_output, "w", encoding="utf-8")

    try:
        for ds in DATASETS:
            json_path = os.path.join(data_root, ds["json_rel"])
            img_dir = os.path.join(data_root, ds["img_dir_rel"])

            if ds["split"] == "train":
                out_f = train_f
            else:
                out_f = val_f

            convert_dataset(
                json_path=json_path,
                img_dir=img_dir,
                out_f=out_f,
                split=ds["split"],
                dataset_name=ds["name"],
            )
    finally:
        train_f.close()
        val_f.close()

    print("[done] 所有数据集转换完成。")
    print(f"训练集写入: {args.train_output}")
    print(f"验证集写入: {args.val_output}")


if __name__ == "__main__":
    main()
