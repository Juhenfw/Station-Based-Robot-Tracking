[English](README.md) | [Indonesia](README-id.md) | [日本語](README-ja.md) | [简体中文](README-zh.md)

---

# 基于站点的机器人追踪 (开源模板)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%99%A5-red)](https://opensource.org/)

本仓库是一个**对开源友好的模板**，用于运行**多摄像头机器人检测 + 检查点（站点）追踪**。

它的设计符合生产环境风格：
- 一个 **编排器 (orchestrator)** 进程负责启动和监控
- 多个 **摄像头工作进程 (camera worker)**（每个摄像头一个）

所有可能包含敏感信息的内容（RTSP URL、凭据、内部数据库设置）都已移至本地 `config.yaml` 中，您**不应提交 (commit)** 该文件。

## 功能特点

- **并行运行 N 个摄像头**（每个摄像头一个操作系统进程）
- 支持的视频源：**网络摄像头 (webcam)**、**视频文件**、**RTSP**
- 通过 **Ultralytics** 进行 YOLO 检测（可选），并自动降级为 **无检测模式 (no-detect mode)**
- 简单的追踪（基于质心） + **检查点进入/退出事件**
- 开源安全的事件存储：**SQLite** 或 **JSONL** (不需要 MySQL)

## 仓库结构

- `configs/config.example.yaml` — 配置示例（可安全提交）
- `configs/config.yaml` — 您的实际配置文件（**切勿提交**）
- `src/holabot_station/run_station.py` — 编排器 (负责生成工作进程)
- `src/holabot_station/camera_worker.py` — 单个摄像头的核心工作脚本
- `src/holabot_station/db.py` — 本地事件接收器
- `src/holabot_station/tracker.py` — 可复用的追踪 + 检查点逻辑
- `src/holabot_station/vision.py` — YOLO 探测器封装

## 快速开始 (Windows)

### 1) 创建并激活虚拟环境

PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) 安装依赖项

```bash
pip install -r requirements.txt
```

完整功能推荐的包：
- `opencv-python`
- `pyyaml`
- `ultralytics` (仅当您需要 YOLO 检测时需要)

### 3) 创建您的配置

复制示例配置：
```powershell
Copy-Item .\configs\config.example.yaml .\configs\config.yaml
```

编辑 `configs/config.yaml`：
- 为了测试，首先将 `cameras[].source` 设置为 `webcam` 或 `file`
- (可选) 将 `model.path` 设置为您的模型文件（示例使用 `model.pt`）

### 4) 确保 Python 可以从 `src/` 导入

如果您没有安装此包，请将 `./src` 添加到 `PYTHONPATH`：
```powershell
$env:PYTHONPATH = "$PWD\src"
```
(如果需要，您可以将其添加到 PowerShell 配置文件中。)

### 5) 运行 (编排器生成摄像头进程)

```powershell
python -m holabot_station.run_station --config configs/config.yaml
```
按 **Ctrl+C** 停止。

## 运行单个摄像头工作进程 (用于调试)

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m holabot_station.camera_worker --config configs/config.yaml --camera cam1
```
在 OpenCV 窗口中按 **q** 停止工作进程。

## 事件输出

事件根据 `storage.type` 写入：
- `sqlite` (默认): `data/events.sqlite`，包含表 `events`
- `jsonl`: `data/events.jsonl` (每行一个 JSON)
- `none`: 丢弃事件

示例事件：
```json
{
  "ts": 1720000000.123,
  "camera": "cam1",
  "track_id": 3,
  "station": "Station_1",
  "type": "enter",
  "meta": {"source_type": "file"}
}
```

## 关于模型的说明 (开源安全性)

此模板即使 **没有模型也能运行**：
- 如果未安装 `ultralytics`，或缺少 `model.path`，工作进程将进入 **无检测模式**。
- 在无检测模式下，应用程序仍会显示视频和检查点，以便您验证摄像头接线和配置。

如果您需要检测：
- 将 YOLO `.pt` 文件放入仓库中（例如 `model.pt`）或您选择的任何路径
- 相应地在 `configs/config.yaml` 中设置 `model.path`

## 故障排除 (Windows)

- **`ModuleNotFoundError: holabot_station`**
  - 确保运行前已执行：`$env:PYTHONPATH = "$PWD\src"`。

- **OpenCV 无法打开摄像头/文件**
  - 尝试先将 `source.type` 切换为 `file` 并提供有效的 `file_path`。

- **YOLO 未运行 / 找不到模型**
  - 安装 ultralytics: `pip install ultralytics`
  - 确保 `model.path` 指向现有的文件。

## 许可证

参见 `LICENSE`。