# YOLO 跳一跳外挂（PC 微信小程序）

基于 **YOLOv8 视觉识别** 的微信小游戏「跳一跳」辅助外挂。识别棋子与目标方块 →
算距离 → 模拟长按。

> ⚠️ 仅供学习 YOLO + 自动化的技术练习。游戏窗口需打开并保持前台。

## 工作原理

1. **截图**：mss 高速截取微信小程序游戏窗口的客户区（[src/window.py](src/window.py)）。
2. **检测**：YOLOv8 识别 `piece`（棋子）与 `platform`（方块）（[src/detect.py](src/detect.py)）。
3. **几何**：棋子框底部中心 → 最近目标方块框顶部中心，欧氏像素距离；`press_ms = dist * k + b`（[src/distance.py](src/distance.py)）。
4. **按压**：pyautogui 长按 + `perf_counter` 忙等保证计时精度（[src/control.py](src/control.py)）。
5. **循环**：F9 启动 / F10 停止 / ESC 退出（[src/bot.py](src/bot.py)）。

## 环境准备

- Windows，**NVIDIA GPU**（本项目在 RTX 5070 上验证；CPU 也能跑，慢些）。
- Python 3.10+（已在 3.13 测试）。
- 微信 PC 端打开「跳一跳」小程序。

```bash
# 1) 装 CUDA 版 torch（关键，否则用 CPU）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) 装其余依赖
pip install -r requirements.txt

# 3) 验证 GPU
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## 完整使用流程

### 第 1 步：确认能截到游戏画面
打开微信「跳一跳」窗口，运行：
```bash
cd src
python window.py      # 弹窗显示游戏画面，按 q 退出
```
若两侧有黑边/留白，编辑 [config.yaml](config.yaml) 的 `crop_margin` 裁掉。

### 第 2 步：采集训练数据（300~500 张）
```bash
cd src
python capture.py                 # 定时采集，手动玩几局；Ctrl+C 停止
# 或手动：python capture.py --manual   # 空格截一张
```
覆盖不同方块形状、距离、白天/夜间背景。图存 `data/images/`。

### 第 3 步：标注
```bash
cd src
python annotate_helper.py setup   # 生成 classes.txt + 打印 LabelImg 命令
pip install labelImg
labelImg ../data/images ../data/classes.txt
```
LabelImg 里切换到 **YOLO** 格式，画框标 `piece` / `platform`。标注文件落到 `data/labels/`。
> 在线备选：Roboflow 上传 `data/images/`，导出 YOLOv8 格式，把 labels 解压回 `data/labels/`。

### 第 4 步：校验 + 划分 train/val
```bash
python annotate_helper.py check   # 校验标注格式
python annotate_helper.py split   # 8:2 划分到 images/{train,val} 与 labels/{train,val}
```

### 第 5 步：训练
```bash
python train.py                           # 默认 yolov8n, 150 epochs
# 数据多时：python train.py --weights yolov8s.pt --epochs 200
```
权重自动复制到 `models/best.pt`。

### 第 6 步：调试检测 + 距离
```bash
python detect.py     # 调试窗口：红框=棋子，黄框=目标方块，紫线=距离，按 q 退出
```
确认棋子底部→目标顶部的距离线合理。

### 第 7 步：标定按压系数 k
```bash
python calibrate.py live      # 实时显示 dist 与建议 press；+/- 调 k，s 保存，q 退出
```
观察落点：**过跳（飞过头）→ k 调小；欠跳（没跳到）→ k 调大**。
也可用已知点反推：`python calibrate.py adjust --distance 300 --ideal 420`。

### 第 8 步：开跑
```bash
python bot.py
```
- 把游戏窗口放前台。
- **F9** 启动自动跳跃，**F10** 停止，**ESC** 退出。

## 常见问题

- **找不到窗口**：确认游戏已打开；必要时改 [config.yaml](config.yaml) 的 `window_title`。
- **检测不到棋子/方块**：训练数据不足或类别不准，补充标注重训；也可适当调低 `detect_conf`。
- **过跳/欠跳**：永远是 k 的问题，用 `calibrate.py live` 重标。DPI 缩放或窗口大小变化后需重标。
- **按压时长不准**：已用忙等而非 `sleep`；若仍偏，检查是否有其它程序抢占前台。
- **被风控**：已加随机抖动，但分数过高有风险，仅供学习。

## 项目结构

```
src/
  window.py          定位微信小程序窗口 + mss 截图
  capture.py         数据采集（定时/手动）
  annotate_helper.py 标注辅助：setup / split / check
  train.py           YOLOv8 训练
  detect.py          推理 + 调试可视化
  distance.py        目标选择 + 距离 + 按压时长
  control.py         pyautogui 高精度长按
  calibrate.py       按压系数 k 标定
  bot.py             主循环 + 热键
data/
  images/ labels/ dataset.yaml
models/best.pt
config.yaml
```
