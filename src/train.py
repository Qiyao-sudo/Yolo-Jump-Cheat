"""YOLOv8 训练入口。

用法：
    python src/train.py                       # 默认 yolov8n, 150 epochs
    python src/train.py --weights yolov8s.pt --epochs 200 --batch 16

权重读到 runs/detect/train/weights/best.pt，会自动复制到 models/best.pt。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "dataset.yaml"
MODELS_DIR = ROOT / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="预训练权重：yolov8n/s/m.pt（n 最快，数据多用 s）")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0=GPU, cpu=CPU")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name="train",
        exist_ok=False,
    )

    best = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if best.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dst = MODELS_DIR / "best.pt"
        shutil.copy2(best, dst)
        print(f"[OK] 训练完成，权重已复制到 {dst}")
    else:
        print(f"[!] 未找到 {best}，请检查训练日志。")


if __name__ == "__main__":
    main()
