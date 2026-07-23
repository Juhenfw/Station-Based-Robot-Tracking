[English](README.md) | [Indonesia](README-id.md) | [日本語](README-ja.md) | [简体中文](README-zh.md)

---

# ステーションベースのロボットトラッキング (オープンソース テンプレート)

このリポジトリは、**マルチカメラによるロボット検出とチェックポイント（ステーション）トラッキング**を実行するための**オープンソース向けテンプレート**です。

本番環境に合わせた設計になっており、以下の構成で動作します：
- 1つの **オーケストレーター (orchestrator)** プロセスが起動と監視を担当
- 複数の **カメラワーカー (camera worker)** プロセス（カメラごとに1つ）

機密情報（RTSP URL、認証情報、内部データベース設定など）を含む可能性のあるものはすべてローカルの `config.yaml` に移動されます。これは **コミットしないでください**。

## 主な機能

- **N台のカメラを並列実行**（カメラごとに1つのOSプロセス）
- サポートするソース：**ウェブカメラ**, **動画ファイル**, **RTSP**
- **Ultralytics** を使用したYOLO検出（オプション）、自動的に **非検出モード (no-detect mode)** へのフォールバック機能付き
- シンプルなトラッキング（重心ベース） + **チェックポイントへの進入/退出イベント**
- オープンソースで安全なイベントストレージ：**SQLite** または **JSONL** (MySQLは不要)

## リポジトリ構成

- `configs/config.example.yaml` — 設定例（コミットしても安全）
- `configs/config.yaml` — 実際の設定ファイル（**コミット不可**）
- `src/holabot_station/run_station.py` — オーケストレーター（ワーカーを生成）
- `src/holabot_station/camera_worker.py` — カメラごとのワーカー
- `src/holabot_station/db.py` — ローカルイベントシンク
- `src/holabot_station/tracker.py` — 再利用可能なトラッキング + チェックポイントロジック
- `src/holabot_station/vision.py` — YOLO 検出器ラッパー

## クイックスタート (Windows)

### 1) 仮想環境の作成と有効化

PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) 依存関係のインストール

```bash
pip install -r requirements.txt
```

全機能を利用するための推奨パッケージ：
- `opencv-python`
- `pyyaml`
- `ultralytics` (YOLO検出を行いたい場合のみ必要)

### 3) 設定ファイルの作成

設定例をコピーします：
```powershell
Copy-Item .\configs\config.example.yaml .\configs\config.yaml
```

`configs/config.yaml` を編集：
- テストのため、最初に `cameras[].source` を `webcam` または `file` に設定します。
- 必要に応じて `model.path` をお使いのモデルファイルに設定します（例では `model.pt` を使用）。

### 4) Python が `src/` からインポートできることを確認

このパッケージをインストールしていない場合は、`./src` を `PYTHONPATH` に追加してください：
```powershell
$env:PYTHONPATH = "$PWD\src"
```
(必要に応じてPowerShellプロファイルに追加できます。)

### 5) 実行 (オーケストレーターがカメラワーカーを生成)

```powershell
python -m holabot_station.run_station --config configs/config.yaml
```
停止するには **Ctrl+C** を押します。

## 単一のカメラワーカーの実行 (デバッグ)

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m holabot_station.camera_worker --config configs/config.yaml --camera cam1
```
ワーカーを停止するには、OpenCVウィンドウで **q** を押します。

## イベント出力

イベントは `storage.type` に基づいて書き込まれます：
- `sqlite` (デフォルト): `events` テーブルを持つ `data/events.sqlite`
- `jsonl`: `data/events.jsonl` (1行に1つのJSON)
- `none`: イベントを破棄

イベントの例：
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

## モデルに関する注意事項 (オープンソースの安全性)

このテンプレートは **モデルがなくても動作します**：
- `ultralytics` がインストールされていない場合、または `model.path` が見つからない場合、ワーカーは **非検出モード** に入ります。
- 非検出モードでも、カメラの配線や設定を検証できるように、映像とチェックポイントが表示されます。

検出を行いたい場合：
- リポジトリ内にYOLOの `.pt` ファイル（例：`model.pt`）などを配置します。
- それに応じて `configs/config.yaml` の `model.path` を設定します。

## トラブルシューティング (Windows)

- **`ModuleNotFoundError: holabot_station`**
  - 実行前に `$env:PYTHONPATH = "$PWD\src"` が設定されているか確認してください。

- **OpenCVがカメラ/ファイルを開けない**
  - 最初に `source.type` を `file` に切り替え、有効な `file_path` を提供して試してください。

- **YOLOが実行されない / モデルが見つからない**
  - ultralyticsをインストールします: `pip install ultralytics`
  - `model.path` が既存のファイルを指していることを確認してください。

## ライセンス

`LICENSE` を参照してください。