"""核心几何：选择目标方块 + 计算棋子→目标距离 + 按压时长映射。"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from detect import Detection, Box


def select_target(det: Detection) -> Tuple[
    Optional[Box],                 # 目标方块框
    Optional[Tuple[int, int]],     # 棋子锚点（屏幕-相对帧坐标），调试用
    Optional[Tuple[int, int]],     # 目标锚点，调试用
    Optional[float],               # 距离（像素）
]:
    """返回 (target_box, piece_anchor, target_anchor, distance)。

    - 棋子锚点：棋子框底部中心（棋子是小瓶子，底部中心是接触点）。
    - 起点方块：棋子底部锚点落入其框内的 platform。
    - 目标方块：除起点外，离棋子最近的 platform。
    - 目标锚点：目标方块框顶部中心（顶面落点）。
    """
    if det.piece is None or not det.platforms:
        return None, None, None, None

    px, py = det.piece.cx, det.piece.bottom
    piece_anchor = (int(px), int(py))

    # 起点方块 = 棋子锚点所在的 platform（用底部中心判断更稳）
    start = None
    for p in det.platforms:
        if p.contains(px, py):
            start = p
            break

    # 候选 = 除起点外、且在棋子上方（屏幕坐标 y 更小）的方块。
    # 跳一跳里目标方块永远在棋子前方（上方），排除下方的方块避免误选。
    candidates = [p for p in det.platforms if p is not start and p.top < py]
    if not candidates:
        # 极端兜底：若没有严格在棋子上方的，退回到除起点外的全部
        candidates = [p for p in det.platforms if p is not start]
    if not candidates:
        candidates = det.platforms[:]  # 再退一步

    # 目标 = 离棋子锚点最近的候选（欧氏距离，棋子锚点 vs 目标锚点）
    def anchor(p: Box):
        # 目标锚点：方块顶部中心向下移动 1/3 方块宽度（落在顶面中心区域，
        # 而非顶边边缘，更符合实际落点位置）
        offset = (p.x2 - p.x1) / 3
        return (p.cx, p.top + offset)

    target = min(candidates, key=lambda p: math.hypot(p.cx - px, p.top - py))
    tx, ty = anchor(target)
    target_anchor = (int(tx), int(ty))
    dist = math.hypot(tx - px, ty - py)
    return target, piece_anchor, target_anchor, dist


def press_ms(dist: float, k: float, b: float) -> float:
    """按压时长 = 距离 * k + b（毫秒）。"""
    return dist * k + b
