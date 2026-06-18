"""主循环：截图 -> YOLO 检测 -> 选目标算距离 -> pyautogui 长按。

热键：
    F9  启动自动跳跃
    F10 停止
    ESC 退出程序
"""
from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import cv2
import yaml

from window import ScreenGrabber, find_game_window, grab_window_frame, bring_to_front
from detect import Detector, draw
from distance import select_target, press_ms
from control import long_press

ROOT = Path(__file__).resolve().parent.parent


class Bot:
    def __init__(self):
        # 切换到项目根目录，让 models/best.pt 的相对路径生效
        import os
        os.chdir(ROOT)
        cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
        self.cfg = cfg
        box = find_game_window(cfg["window_title"])
        if box is None:
            raise SystemExit(f"[!] 没找到窗口「{cfg['window_title']}」，请先打开游戏。")
        self.box = box
        self.det = Detector(cfg["model"], conf=cfg.get("detect_conf", 0.4))
        self.grabber = ScreenGrabber()
        self.k = cfg["press"]["k"]
        self.b = cfg["press"]["b"]
        self.interval = cfg["loop"]

        # 按压中心：窗口客户区中心（相对屏幕坐标）
        self.press_xy = (box.left + box.width // 2, box.top + box.height // 2)

        self._running = threading.Event()
        self._stop_all = threading.Event()

    # ---- 热键 ----
    def _on_key(self, key):
        from pynput import keyboard
        hot = self.cfg["hotkeys"]
        target = None
        try:
            target = key.char
        except AttributeError:
            target = key.name  # f9/f10/esc...
        if target == hot.get("start"):
            self._running.set()
            print("[BOT] 启动")
        elif target == hot.get("stop"):
            self._running.clear()
            print("[BOT] 停止")
        elif target == hot.get("exit"):
            self._running.clear()
            self._stop_all.set()
            print("[BOT] 退出")

    def _register_hotkeys(self):
        from pynput import keyboard
        self._listener = keyboard.Listener(on_press=self._on_key)
        self._listener.daemon = True
        self._listener.start()

    # ---- 单跳 ----
    def step(self) -> tuple[bool, "np.ndarray | None"]:
        # 每跳前重新获取窗口位置，避免窗口被移动后按压坐标失效
        box = find_game_window(self.cfg["window_title"])
        if box is None:
            print("[warn] 窗口丢失，跳过本次")
            return False, None
        self.box = box
        self.press_xy = (box.left + box.width // 2, box.top + box.height // 2)

        frame = grab_window_frame(self.box, self.grabber, self.cfg.get("crop_margin"))
        det = self.det.detect(frame)
        target, pa, ta, dist = select_target(det)
        # 调试可视化
        vis = draw(frame, det, target, anchor_from=pa, anchor_to=ta)
        if dist is not None:
            ms = press_ms(dist, self.k, self.b)
            cv2.putText(vis, f"dist={dist:.0f}px press={ms:.0f}ms k={self.k} b={self.b}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if det.piece is None or target is None or dist is None:
            return False, vis
        ms = press_ms(dist, self.k, self.b)
        long_press(self.press_xy, ms)
        return True, vis

    # ---- 主循环 ----
    def run(self):
        bring_to_front(self.box.hwnd)
        self._register_hotkeys()
        print("热键: F9 启动 | F10 停止 | ESC 退出")
        print(f"窗口: hwnd={self.box.hwnd} 客户区 {self.box.width}x{self.box.height} @ ({self.box.left},{self.box.top})")
        print(f"按压中心 {self.press_xy}  k={self.k} b={self.b}")
        print("调试窗口显示检测结果（红框=棋子, 黄框=目标, 紫线=距离）")
        while not self._stop_all.is_set():
            if self._running.is_set():
                try:
                    ok, vis = self.step()
                    if not ok:
                        time.sleep(0.3)
                except Exception as e:
                    print(f"[warn] {e}")
                    time.sleep(0.5)
                # 显示这一帧的检测结果
                if vis is not None:
                    cv2.imshow("bot debug (F9=start, F10=stop, ESC=quit)", vis)
                    cv2.waitKey(1)
                # 跳间随机间隔
                time.sleep(random.uniform(self.interval["min_interval"],
                                          self.interval["max_interval"]))
            else:
                # 暂停时也持续显示检测结果，便于观察
                try:
                    frame = grab_window_frame(self.box, self.grabber, self.cfg.get("crop_margin"))
                    det = self.det.detect(frame)
                    target, pa, ta, dist = select_target(det)
                    vis = draw(frame, det, target, anchor_from=pa, anchor_to=ta)
                    status = "RUNNING" if self._running.is_set() else "PAUSED (F9 to start)"
                    cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255) if not self._running.is_set() else (0, 255, 0), 2)
                    if dist is not None:
                        ms = press_ms(dist, self.k, self.b)
                        cv2.putText(vis, f"dist={dist:.0f}px press={ms:.0f}ms",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("bot debug (F9=start, F10=stop, ESC=quit)", vis)
                except Exception:
                    pass
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC 也在 cv2 窗口生效
                    self._running.clear()
                    self._stop_all.set()
                    break
                time.sleep(0.05)
        cv2.destroyAllWindows()
        print("[BOT] 已退出。")


if __name__ == "__main__":
    Bot().run()
