"""按压控制：pyautogui 长按，perf_counter 忙等保证计时精度。

跳一跳里按压位置不影响跳跃距离，只有按压时长影响；所以按在游戏区一个
安全点即可。落点/时长加随机抖动，避免机械化的等间隔。
"""
from __future__ import annotations

import random
import time

import pyautogui

# 防止 pyautogui 自带的 0.1s 延迟干扰计时
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False


def busy_sleep(seconds: float) -> None:
    """忙等到指定秒数。time.sleep 在 Windows 精度 ~15ms，忙等可达 <1ms。"""
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def long_press(center_xy: tuple[int, int], duration_ms: float,
               pos_jitter: int = 6, time_jitter_ms: float = 8.0) -> None:
    """在 center_xy 附近长按 duration_ms 毫秒。

    - pos_jitter: 落点随机抖动像素数。
    - time_jitter_ms: 时长随机抖动毫秒数。
    """
    jx = center_xy[0] + random.randint(-pos_jitter, pos_jitter)
    jy = center_xy[1] + random.randint(-pos_jitter, pos_jitter)
    dur = max(40.0, duration_ms + random.uniform(-time_jitter_ms, time_jitter_ms))

    pyautogui.moveTo(jx, jy)
    pyautogui.mouseDown(jx, jy)
    busy_sleep(dur / 1000.0)
    pyautogui.mouseUp(jx, jy)
