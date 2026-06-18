"""按压系数 k 标定工具。

两种用法：
    python src/calibrate.py live       # 实时预览：显示每帧距离与建议按压时长
                                        # 你手动按对应时长，根据过/欠跳调整 k

    python src/calibrate.py adjust --distance 300 --ideal 420
                                        # 已知某距离下理想按压时长，反推 k

调整后的 k/b 写回 config.yaml。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from window import ScreenGrabber, find_game_window, grab_window_frame
from detect import Detector
from distance import select_target

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.yaml"


def load_cfg():
    return yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))


def save_cfg(cfg: dict):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def live():
    import os
    # 切换到项目根目录，让 models/best.pt 的相对路径生效
    os.chdir(ROOT)
    cfg = load_cfg()
    box = find_game_window(cfg["window_title"])
    assert box is not None, "没找到游戏窗口"
    det = Detector(cfg["model"], conf=cfg.get("detect_conf", 0.4))
    grabber = ScreenGrabber()
    k, b = cfg["press"]["k"], cfg["press"]["b"]
    print("实时标定预览。按 +/- 调整 k，s 保存，q 退出。")
    print(f"  当前 k={k} b={b}")
    while True:
        frame = grab_window_frame(box, grabber, cfg.get("crop_margin"))
        d = det.detect(frame)
        target, pa, ta, dist = select_target(d)
        vis = frame.copy()
        if pa and ta:
            cv2.line(vis, pa, ta, (255, 0, 255), 2)
            cv2.circle(vis, pa, 4, (0, 0, 255), -1)
            cv2.circle(vis, ta, 4, (0, 255, 255), -1)
        if dist is not None:
            ms = dist * k + b
            cv2.putText(vis, f"dist={dist:.0f} press={ms:.0f}ms k={k:.3f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("calibrate (+/- k, s save, q quit)", vis)
        key = cv2.waitKey(200) & 0xFF
        if key == ord("+") or key == ord("="):
            k += 0.02
            print(f"  k={k:.3f}")
        elif key == ord("-"):
            k -= 0.02
            print(f"  k={k:.3f}")
        elif key == ord("s"):
            cfg["press"]["k"], cfg["press"]["b"] = round(k, 4), b
            save_cfg(cfg)
            print(f"  [已保存] k={k:.4f} -> {CFG_PATH}")
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


def adjust(distance: float, ideal_ms: float):
    """press = dist*k + b，假设 b 已知（默认用 config 里的 b）反推 k。"""
    cfg = load_cfg()
    b = cfg["press"]["b"]
    k = (ideal_ms - b) / distance
    print(f"反推 k = ({ideal_ms} - {b}) / {distance} = {k:.4f}")
    cfg["press"]["k"] = round(k, 4)
    save_cfg(cfg)
    print(f"[已保存] -> {CFG_PATH}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("live", help="实时预览并 +/- 调 k")
    p2 = sub.add_parser("adjust", help="按已知 (距离, 理想时长) 反推 k")
    p2.add_argument("--distance", type=float, required=True)
    p2.add_argument("--ideal", type=float, required=True, help="理想按压时长 ms")
    args = ap.parse_args()
    if args.cmd == "live":
        live()
    elif args.cmd == "adjust":
        adjust(args.distance, args.ideal)


if __name__ == "__main__":
    main()
