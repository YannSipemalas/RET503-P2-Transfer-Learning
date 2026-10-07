cat << 'EOF' > README.md
#  RET503 Computer Vision and Deep Learning - Praktikum P2
## Transfer Learning dan Fine-Tuning Model Visi (Klasifikasi Box Merah vs Box Cokelat)

**Program Studi:** Teknologi Rekayasa Robotika - Politeknik Negeri Batam  
**Model Visi:** ResNet-18  
**Hardware Pengujian:** NVIDIA GeForce RTX 4060 Laptop GPU  
**Framework:** PyTorch & Torchvision  

---

##  1. Deskripsi Proyek
Proyek ini mengimplementasikan Transfer Learning menggunakan arsitektur ResNet-18 untuk membedakan dua jenis komponen/objek industri pada robot:
- box_merah (100 citra frame video)
- box_cokelat (100 citra frame video)

Tujuan utama praktikum ini adalah membandingkan 3 pendekatan strategi pelatihan deep learning serta mengukur kecepatan inferensi (latensi) untuk memastikan model layak dideploy pada sistem robot real-time.

---

##  2. Struktur Dataset & Manajemen Data
- Total Citra: 200 citra (100 per kelas)
- Metadata: File metadata.csv mencatat path, label, frame_index, group, width, dan height.
- Pemisahan Data (Data Split): Memakai skrip split.py dengan metode temporal split (menahan segmen frame akhir sebagai data validasi) untuk menghindari data leakage akibat frame berurutan dari video.
  - Data Train: 78 citra per kelas (Total: 156 citra)
  - Data Validation: 18 citra per kelas (Total: 36 citra)
  - Excluded (Guard Band): 4 citra per kelas

---

##  3. Hasil Eksperimen 3 Mode Pelatihan (10 Epoch)

| Mode Pelatihan | Bobot Awal (Weights) | Layer Dilatih | Learning Rate | Akurasi Validasi Terbaik | Converge (Akurasi 100%) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| partial (Fine-Tuning Parsial) | ImageNet Pretrained | layer4 + fc | 1e-4 (layer4) / 1e-3 (fc) | 100.00% | Epoch 1 |
| feature (Feature Extraction) | ImageNet Pretrained | fc saja | 1e-3 | 100.00% | Epoch 4 |
| scratch (Pelatihan dari Nol) | Random (Acak) | Seluruh Layer | 1e-3 | 100.00% | Epoch 4 |

### Analisis Perbandingan Mode:
1. Mode partial (Terbaik): Sangat unggul dan tercepat konvergen. Sejak Epoch 1, akurasi validasi langsung menyentuh 100%. Membuka kunci layer4 membantu model menyesuaikan fitur visual tingkat tinggi dengan karakteristik khusus box industri.
2. Mode feature: Sangat stabil dan hemat komputasi karena hanya melatih layer fc akhir.
3. Mode scratch: Membutuhkan waktu awal lebih lambat (akurasi Epoch 1 & 3 fluktuatif di 50% / setara tebakan acak koin) karena bobot diawali dari nilai acak tanpa bantuan pretrained knowledge dari ImageNet.

---

##  4. Hasil Pengukuran Latensi (Inference Speed)

- Device: cuda (NVIDIA GeForce RTX 4060 Laptop GPU)
- Batch Size: 1
- Median Latensi: 1.247 ms per frame
- Mean Latensi: 1.248 ms per frame
- P95 Latensi: 1.252 ms per frame
- Target Budget Latensi Robot (15 FPS): <= 67 ms
- Kesimpulan Latensi: Model ResNet-18 di GPU RTX 4060 berjalan sangat cepat (~1.25 ms), jauh di bawah batas 67 ms. Model ini sangat siap dideploy pada sistem robot hingga kecepatan ~800 FPS.

---

##  5. Panduan Lengkap Cara Menjalankan Kode (Step-by-Step)

Langkah 5.1: Persiapan Environment & Instalasi Dependencies
Pastikan berada di direktori utama proyek /home/yanzz/Documents/P2 - Transfer Learning. Buat dan aktifkan lingkungan Python virtual (venv), lalu install seluruh pustaka yang diperlukan:

  python3 -m venv venv
  ./venv/bin/pip install torch torchvision pillow matplotlib

Langkah 5.2: Pemisahan Dataset (split.py)
Skrip ini membagi dataset di metadata.csv menjadi subset train dan validation secara chronological/temporal split dengan menambahkan guard band (gap) untuk mencegah data leakage.

  ./venv/bin/python split.py --val-fraction 0.2 --gap 2

Output Terminal:
  box_cokelat: train=78 validation=18 excluded=4
  box_merah: train=78 validation=18 excluded=4

Langkah 5.3: Pelatihan Model 3 Mode (train.py)
Skrip ini akan melatih model ResNet-18 dalam 3 mode eksperimen secara otomatis (feature, partial, scratch) masing-masing selama 10 epoch.

  ./venv/bin/python train.py --epochs 10 --batch-size 16 --seed 42

Penjelasan Flag Parameter:
- --epochs 10: Jumlah putaran pelatihan per mode (default: 10).
- --batch-size 16: Jumlah sampel per batch.
- --mode all: Melatih 3 mode sekaligus (feature, partial, scratch).
- --seed 42: Mengunci random seed agar hasil pelatihan dapat direproduksi (deterministic).

Output Terbentuk:
- File Checkpoint: models/resnet18_feature.pth, models/resnet18_partial.pth, models/resnet18_scratch.pth
- Catatan Log & CSV: results/experiments.csv, results/feature_history.csv, results/partial_history.csv, results/scratch_history.csv
- Grafik Perbandingan: results/accuracy_comparison.png

Langkah 5.4: Pengukuran Kecepatan Inferensi (latency.py)
Mengukur latensi waktu eksekusi tebakan model untuk 1 gambar (batch size = 1) di hardware target.

  ./venv/bin/python latency.py --checkpoint models/resnet18_feature.pth --image box_merah/box_merah__frame_000000000.jpg
  ./venv/bin/python latency.py --checkpoint models/resnet18_partial.pth --image box_merah/box_merah__frame_000000000.jpg
  ./venv/bin/python latency.py --checkpoint models/resnet18_scratch.pth --image box_merah/box_merah__frame_000000000.jpg

Output Terminal:
  cuda: median=1.247 ms, mean=1.248 ms, p95=1.252 ms

Langkah 5.5: Uji Prediksi Foto Baru (predict.py)
Menguji kemampuan model terlatih dalam mengenali gambar box secara langsung:

  ./venv/bin/python predict.py box_cokelat/box_cokelat__frame_000000000.jpg
  ./venv/bin/python predict.py box_merah/box_merah__frame_000000000.jpg

Contoh Output Terminal:
  ========================================
   File Gambar    : box_cokelat__frame_000000000.jpg
   Hasil Prediksi : BOX_COKELAT
   Tingkat Keyakinan (Confidence): 99.85%
  ========================================
