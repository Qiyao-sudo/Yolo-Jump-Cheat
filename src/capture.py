"""数据采集：边手动玩边截图，攒训练集。

用法：
    python src/capture.py              # 定时自动采集（默认 0.8s/张）
    python src/capture.py --interval 0.5
    python src/capture.py --manual     # 空格手动截一张

存到 data/images/，文件名 cap_XXXXXX.png。
采集结束后，用 LabelImg / Roboflow 标注 piece 与 platform。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from window import ScreenGrabber, find_game_window, grab_window_frame

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images"


def next_path(dir_: Path, prefix: str = "cap_", ext: str = ".png") -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        p = dir_ / f"{prefix}{idx:06d}{ext}"
        if not p.exists():
            return p
        idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.8, help="定时采集间隔（秒）")
    ap.add_argument("--manual", action="store_true", help="手动模式：按空格截一张")
    args = ap.parse_args()

    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    box = find_game_window(cfg["window_title"])
    if box is None:
        print(f"[!] 没找到窗口「{cfg['window_title']}」，请先打开游戏。")
        return

    grabber = ScreenGrabber()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 窗口 {box.width}x{box.height}，存图到 {IMG_DIR}")
    print("     定时模式：按 Ctrl+C 停止。手动模式：按空格截、q 退出预览。")
    print("     目标 300~500 张，覆盖不同方块/距离/背景。")

    count = 0
    if args.manual:
        while True:
            frame = grab_window_frame(box, grabber, cfg.get("crop_margin"))
            cv2.imshow("capture (space=save, q=quit)", frame)
            key = cv2.waitKey(50) & 0xFF
            if key == ord(" "):
                p = next_path(IMG_DIR)
                cv2.imwrite(str(p), frame)
                count += 1
                print(f"  saved {p.name}  ({count})")
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        try:
            while True:
                frame = grab_window_frame(box, grabber, cfg.get("crop_margin"))
                p = next_path(IMG_DIR)
                cv2.imwrite(str(p), frame)
                count += 1
                print(f"  saved {p.name}  ({count})", end="\r")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"\n[完成] 共采集 {count} 张 -> {IMG_DIR}")


if __name__ == "__main__":
    main()
