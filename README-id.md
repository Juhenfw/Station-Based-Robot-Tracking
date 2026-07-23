[English](README.md) | [Indonesia](README-id.md) | [日本語](README-ja.md) | [简体中文](README-zh.md)

---

# Pelacakan Robot Berbasis Stasiun (Templat Open Source)

Repositori ini adalah **templat ramah open-source** untuk menjalankan **deteksi robot multi-kamera + pelacakan pos pemeriksaan (stasiun)**.

Dirancang untuk menyesuaikan pengaturan gaya produksi di mana:
- satu proses **orkestrator** memulai dan memantau
- beberapa proses **pekerja kamera (camera worker)** (satu per kamera)

Segala hal yang berisi informasi sensitif (URL RTSP, kredensial, pengaturan database internal) dipindahkan ke `config.yaml` lokal yang **jangan Anda commit**.

## Fitur

- Jalankan **N kamera secara paralel** (satu proses OS per kamera)
- Sumber yang didukung: **webcam**, **file video**, **RTSP**
- Deteksi YOLO via **Ultralytics** (opsional) dengan fallback otomatis ke **mode tanpa deteksi (no-detect mode)**
- Pelacakan sederhana (berbasis centroid) + **event masuk/keluar pos pemeriksaan**
- Penyimpanan event yang aman untuk open-source: **SQLite** atau **JSONL** (tidak butuh MySQL)

## Struktur Repositori

- `configs/config.example.yaml` — contoh konfigurasi (aman di-commit)
- `configs/config.yaml` — konfigurasi asli Anda (**jangan di-commit**)
- `src/holabot_station/run_station.py` — orkestrator (menjalankan pekerja)
- `src/holabot_station/camera_worker.py` — pekerja per-kamera
- `src/holabot_station/db.py` — sink event lokal
- `src/holabot_station/tracker.py` — logika pelacakan + pos pemeriksaan yang dapat digunakan kembali
- `src/holabot_station/vision.py` — wrapper detektor YOLO

## Mulai Cepat (Windows)

### 1) Buat dan aktifkan virtual environment

PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Instal dependensi

```bash
pip install -r requirements.txt
```

Paket yang disarankan untuk fungsionalitas penuh:
- `opencv-python`
- `pyyaml`
- `ultralytics` (hanya diperlukan jika Anda ingin deteksi YOLO)

### 3) Buat konfigurasi Anda

Salin contoh konfigurasi:
```powershell
Copy-Item .\configs\config.example.yaml .\configs\config.yaml
```

Edit `configs/config.yaml`:
- atur `cameras[].source` ke `webcam` atau `file` terlebih dahulu untuk pengujian
- (opsional) atur `model.path` ke file model Anda (contoh menggunakan `model.pt`)

### 4) Pastikan Python dapat mengimpor dari `src/`

Jika Anda belum menginstal paket ini, tambahkan `./src` ke `PYTHONPATH`:
```powershell
$env:PYTHONPATH = "$PWD\src"
```
(Anda dapat menambahkan ini ke profil PowerShell Anda jika mau.)

### 5) Jalankan (orkestrator meluncurkan pekerja kamera)

```powershell
python -m holabot_station.run_station --config configs/config.yaml
```
Hentikan dengan **Ctrl+C**.

## Menjalankan pekerja kamera tunggal (debug)

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m holabot_station.camera_worker --config configs/config.yaml --camera cam1
```
Tekan **q** di jendela OpenCV untuk menghentikan pekerja.

## Output Event

Event ditulis berdasarkan `storage.type`:
- `sqlite` (default): `data/events.sqlite` dengan tabel `events`
- `jsonl`: `data/events.jsonl` (satu JSON per baris)
- `none`: abaikan event

Contoh event:
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

## Catatan tentang model (keamanan open-source)

Templat ini akan **berjalan meskipun tanpa model**:
- Jika `ultralytics` tidak diinstal, atau `model.path` tidak ada, pekerja memasuki **mode tanpa deteksi (no-detect mode)**.
- Dalam mode ini, aplikasi masih menampilkan video + pos pemeriksaan sehingga Anda dapat memvalidasi kabel kamera dan konfigurasi.

Jika Anda menginginkan deteksi:
- letakkan file YOLO `.pt` di dalam repositori (contoh: `model.pt`) atau path mana pun yang Anda pilih
- atur `model.path` di `configs/config.yaml`

## Pemecahan Masalah (Windows)

- **`ModuleNotFoundError: holabot_station`**
  - Pastikan: `$env:PYTHONPATH = "$PWD\src"` sebelum menjalankan program.

- **OpenCV tidak dapat membuka kamera / file**
  - Coba ubah `source.type` ke `file` terlebih dahulu dan berikan `file_path` yang valid.

- **YOLO tidak berjalan / model tidak ditemukan**
  - Instal ultralytics: `pip install ultralytics`
  - Pastikan `model.path` mengarah ke file yang ada.

## Lisensi

Lihat `LICENSE`.