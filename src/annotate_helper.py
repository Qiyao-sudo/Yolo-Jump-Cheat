"""标注辅助：生成 LabelImg classes、启动命令、划分 train/val、校验标注。

子命令：
    python src/annotate_helper.py setup     # 生成 classes.txt + 打印 LabelImg 启动命令
    python src/annotate_helper.py split     # 8:2 划分 images+labels 到 train/val
    python src/annotate_helper.py check     # 校验 images/labels 一一对应 + YOLO 格式合法
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images"
LBL_DIR = DATA_DIR / "labels"

CLASSES = ["piece", "platform"]


def setup():
    """生成 LabelImg classes.txt 并打印启动命令。"""
    classes_file = DATA_DIR / "classes.txt"
    classes_file.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    print(f"[OK] 已写 {classes_file}")
    print("\n=== LabelImg 安装与启动 ===")
    print("  pip install labelImg")
    print(f"  labelImg {IMG_DIR} {classes_file}")
    print("  在 LabelImg 里: 左侧切到 'YOLO' 格式，画框标 piece / platform，保存。")
    print("\n=== Roboflow 在线备选 ===")
    print(f"  上传 {IMG_DIR} 下所有图，标注后导出 'YOLOv8' 格式，解压回 {DATA_DIR}。")
    print("  （Roboflow 导出的 data.yaml 别覆盖我们的 dataset.yaml）")


def split(train_ratio: float = 0.8, seed: int = 42):
    """把 data/images/*.{png,jpg} 和对应 data/labels/*.txt 按 train_ratio 划分到 train/val。"""
    train_img = IMG_DIR / "train"
    val_img = IMG_DIR / "val"
    train_lbl = LBL_DIR / "train"
    val_lbl = LBL_DIR / "val"
    for d in (train_img, val_img, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp")
    imgs = sorted(p for p in IMG_DIR.iterdir()
                  if p.suffix.lower() in exts and p.is_file())
    if not imgs:
        print(f"[!] {IMG_DIR} 下没有图片，先采集数据。")
        return

    rng = random.Random(seed)
    rng.shuffle(imgs)
    n_train = int(len(imgs) * train_ratio)
    train_set = imgs[:n_train]
    val_set = imgs[n_train:]

    missing = 0
    for subset_imgs, dst_img, dst_lbl in (
        (train_set, train_img, train_lbl),
        (val_set, val_img, val_lbl),
    ):
        for img in subset_imgs:
            shutil.copy2(img, dst_img / img.name)
            lbl = LBL_DIR / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)
            else:
                missing += 1

    print(f"[OK] train={len(train_set)} val={len(val_set)}")
    if missing:
        print(f"[!] {missing} 张图没有对应标注 .txt，记得先标注再划分。")


def check():
    """校验 images/labels 对应关系与 YOLO 格式合法性。"""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    imgs = sorted(p for p in IMG_DIR.iterdir()
                  if p.suffix.lower() in exts and p.is_file())
    if not imgs:
        print(f"[!] {IMG_DIR} 下没有图片。")
        return

    no_label = 0
    bad_format = 0
    total_boxes = 0
    for img in imgs:
        lbl = LBL_DIR / (img.stem + ".txt")
        if not lbl.exists():
            no_label += 1
            continue
        for ln, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad_format += 1
                print(f"  [格式错误] {lbl.name}:{ln} -> {line!r}")
                continue
            try:
                c = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                bad_format += 1
                print(f"  [格式错误] {lbl.name}:{ln} -> {line!r}")
                continue
            if c not in (0, 1):
                print(f"  [类别警告] {lbl.name}:{ln} class id={c} 不在 [0,1]")
            for v in (cx, cy, w, h):
                if not (0.0 <= v <= 1.0 + 1e-6):
                    print(f"  [越界警告] {lbl.name}:{ln} 值 {v} 不在 [0,1]")
            total_boxes += 1

    print(f"图片总数: {len(imgs)}")
    print(f"缺少标注: {no_label}")
    print(f"格式错误: {bad_format}")
    print(f"标注框总数: {total_boxes}")
    print("[OK] 校验完成" if no_label == 0 and bad_format == 0 else "[!] 存在问题，按上面提示修正。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "split", "check"])
    ap.add_argument("--ratio", type=float, default=0.8)
    args = ap.parse_args()
    if args.cmd == "setup":
        setup()
    elif args.cmd == "split":
        split(args.ratio)
    elif args.cmd == "check":
        check()


if __name__ == "__main__":
    main()
