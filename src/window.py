"""窗口定位 + 高速截图（mss）。"""
from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Optional

# 必须在导入 win32gui 之前声明 per-monitor DPI 感知，否则在高 DPI 缩放
# 的显示器上，GetClientRect / ClientToScreen 返回的是逻辑像素坐标，
# 而 mss 按物理像素截图，会导致截图位置和尺寸都不对。
try:
    # Windows 8.1+: PROCESS_PER_MONITOR_DPI_AWARE
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    try:
        # Windows Vista+: SetProcessDPIAware fallback
        ctypes.windll.user32.SetProcessDPIAware()
    except OSError:
        pass

import win32con
import win32gui

try:
    import mss
    import numpy as np

    _HAS_DEPS = True
except Exception:  # pragma: no cover
    _HAS_DEPS = False


@dataclass
class WindowBox:
    hwnd: int
    left: int   # 屏幕坐标
    top: int
    width: int  # 客户区宽（不含标题栏）
    height: int


def find_game_window(title_keyword: str = "跳一跳") -> Optional[WindowBox]:
    """枚举顶层窗口，模糊匹配标题关键字，返回客户区矩形。

    微信小程序 PC 端每个小游戏运行在独立顶层窗口，标题即游戏名。
    """
    # 每项存 (hwnd, title, width, height)，在回调内同步算客户区，
    # 避免 EnumWindows 回调结束后 PyHANDLE 失效（"not a PyHANDLE object"）
    matches: list[tuple[int, str, int, int]] = []

    def _enum(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if title and title_keyword in title:
            cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
            matches.append((int(hwnd), title, cr - cl, cb - ct))
        return True

    win32gui.EnumWindows(_enum, None)
    if not matches:
        return None

    # 取最大客户区窗口（避免误匹配子窗口/托盘提示）
    hwnd, title, w, h = max(matches, key=lambda m: m[2] * m[3])

    # ClientToScreen(hwnd, (x,y)) 返回 (x_screen, y_screen)
    sx, sy = win32gui.ClientToScreen(hwnd, (0, 0))
    return WindowBox(hwnd=hwnd, left=sx, top=sy, width=w, height=h)


def bring_to_front(hwnd: int) -> None:
    """把窗口置前台（pyautogui 前台模拟需要）。"""
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.15)
    except Exception:
        pass


class ScreenGrabber:
    """mss 截图器，截取给定屏幕矩形区域，返回 BGR ndarray。"""

    def __init__(self):
        if not _HAS_DEPS:
            raise RuntimeError("缺少 mss / numpy，请先 pip install -r requirements.txt")
        self._sct = mss.mss()

    def grab(self, left: int, top: int, width: int, height: int) -> "np.ndarray":
        monitor = {"left": left, "top": top, "width": width, "height": height}
        shot = self._sct.grab(monitor)
        # mss -> numpy 得到 BGRA（HxWx4），取前 3 通道为 BGR
        img = np.asarray(shot)[:, :, :3]
        return np.ascontiguousarray(img)


def grab_window_frame(box: WindowBox, grabber: Optional[ScreenGrabber] = None,
                      crop=None) -> "np.ndarray":
    """截取窗口客户区，可选用 crop_margin 裁掉留白。"""
    grabber = grabber or ScreenGrabber()
    frame = grabber.grab(box.left, box.top, box.width, box.height)
    if crop:
        h, w = frame.shape[:2]
        cl = crop.get("left", 0)
        cr = crop.get("right", 0)
        ct = crop.get("top", 0)
        cb = crop.get("bottom", 0)
        x1, x2 = cl, max(cl + 1, w - cr)
        y1, y2 = ct, max(ct + 1, h - cb)
        frame = frame[y1:y2, x1:x2]
    return frame


# ---------- 直接运行：调试窗口定位 ----------
if __name__ == "__main__":
    import yaml
    import cv2

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    box = find_game_window(cfg["window_title"])
    if box is None:
        print(f"[!] 没找到标题含「{cfg['window_title']}」的窗口。请确认游戏已打开。")
        raise SystemExit(1)

    print(f"[OK] 找到窗口 hwnd={box.hwnd} 客户区 {box.width}x{box.height} @ ({box.left},{box.top})")

    grabber = ScreenGrabber()
    print("按 q 退出预览。")
    while True:
        frame = grab_window_frame(box, grabber, cfg.get("crop_margin"))
        cv2.imshow("jump window preview (q to quit)", frame)
        if cv2.waitKey(200) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
