"""YOLO 推理封装：从一帧画面检测棋子(piece)与目标方块(platform)。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# 类别 id（与 data/dataset.yaml 一致）
CLS_PIECE = 0
CLS_PLATFORM = 1


@dataclass
class Box:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bottom(self) -> float:
        return self.y2

    @property
    def top(self) -> float:
        return self.y1

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Detection:
    piece: Optional[Box]
    platforms: List[Box]


class Detector:
    def __init__(self, model_path: str | Path, conf: float = 0.4, device: str = "0"):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.device = device

    def detect(self, frame: "np.ndarray") -> Detection:
        res = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )[0]
        piece = None
        platforms: List[Box] = []
        if res.boxes is None:
            return Detection(piece=piece, platforms=platforms)

        for cls_id, xyxy, c in zip(
            res.boxes.cls.cpu().numpy(),
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
        ):
            box = Box(cls=int(cls_id), conf=float(c),
                      x1=float(xyxy[0]), y1=float(xyxy[1]),
                      x2=float(xyxy[2]), y2=float(xyxy[3]))
            if box.cls == CLS_PIECE:
                # 取置信度最高的棋子
                if piece is None or box.conf > piece.conf:
                    piece = box
            elif box.cls == CLS_PLATFORM:
                platforms.append(box)
        return Detection(piece=piece, platforms=platforms)


def draw(frame, det: Detection, target: Optional[Box] = None,
         anchor_from=None, anchor_to=None) -> "np.ndarray":
    """调试可视化。"""
    import cv2
    out = frame.copy()
    if det.piece:
        b = det.piece
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                      (0, 0, 255), 2)
        cv2.circle(out, (int(b.cx), int(b.bottom)), 4, (0, 0, 255), -1)
        cv2.putText(out, "piece", (int(b.x1), int(b.y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for p in det.platforms:
        cv2.rectangle(out, (int(p.x1), int(p.y1)), (int(p.x2), int(p.y2)),
                      (0, 255, 0), 1)
    if target:
        cv2.rectangle(out, (int(target.x1), int(target.y1)),
                      (int(target.x2), int(target.y2)), (0, 255, 255), 2)
        cv2.circle(out, (int(target.cx), int(target.top)), 4, (0, 255, 255), -1)
    if anchor_from and anchor_to:
        cv2.line(out, anchor_from, anchor_to, (255, 0, 255), 2)
    return out


# ---------- 直接运行：调试检测 + 距离线 ----------
if __name__ == "__main__":
    import os
    import time
    import cv2
    import yaml

    # 切换到项目根目录，让 config.yaml / models/best.pt 的相对路径生效
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)

    from window import ScreenGrabber, find_game_window, grab_window_frame
    from distance import select_target

    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    box = find_game_window(cfg["window_title"])
    assert box is not None, "没找到游戏窗口"

    det_model = Detector(cfg["model"], conf=cfg.get("detect_conf", 0.4))
    grabber = ScreenGrabber()
    print("按 q 退出检测预览。")
    while True:
        frame = grab_window_frame(box, grabber, cfg.get("crop_margin"))
        d = det_model.detect(frame)
        target, piece_anchor, target_anchor, dist = select_target(d)
        vis = draw(frame, d, target,
                   anchor_from=piece_anchor, anchor_to=target_anchor)
        if dist is not None:
            cv2.putText(vis, f"dist={dist:.0f}px  press={(dist*cfg['press']['k']+cfg['press']['b']):.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("detect debug (q to quit)", vis)
        if cv2.waitKey(200) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
