# P2 — Klasifikasi Objek Box Industri pada Robot Visi

Project ini membandingkan tiga strategi pelatihan ResNet18 untuk klasifikasi dua kelas komponen objek industri: **Box Merah** (`box_merah`) dan **Box Cokelat** (`box_cokelat`).

---

## Dataset

Dataset yang digunakan dikumpulkan dari kamera robot visi, dengan total 200 citra:

| Kelas | Jumlah citra |
| :--- | :---: |
| Box Merah (`box_merah`) | 100 |
| Box Cokelat (`box_cokelat`) | 100 |
| **Total** | **200** |

Dataset mentah tersimpan dalam folder `box_merah/` dan `box_cokelat/`. Manifest metadata tersimpan di `metadata.csv`, yang mencatat `path`, `label`, `frame_index`, `group`, `width`, `height`, dan status `split`. Pemisahan dataset dilakukan dengan skrip `split.py` menggunakan metode *chronological temporal split* dengan *guard band* (gap) untuk menghindari *data leakage* antar-frame berurutan dari sumber video.

- **Data Training:** 78 citra per kelas (Total: 156 citra)
- **Data Validation:** 18 citra per kelas (Total: 36 citra)
- **Excluded (Guard Band):** 4 citra per kelas (Total: 8 citra)

---

## Rancangan Sistem (CDIO Stage #2 Design)

1. **Misi Proyek:** Sistem visi bertugas mengenali dan membedakan jenis box industri secara *real-time* untuk mengarahkan manipulator robot dalam proses pemilahan (*sorting*).
2. **Kamera & Posisi:** Kamera dipasang pada dudukan *top-down / angled* dengan jarak kerja 30–80 cm di atas meja operasi. Resolusi citra akuisisi adalah $1280 \times 720$ (box_merah) dan $832 \times 464$ (box_cokelat).
3. **Unit Komputasi:** Laptop dengan GPU NVIDIA GeForce RTX 4060 (8 GB VRAM).
4. **Target Kinerja & Latensi:** Target akurasi validasi $\ge 95\%$ dan kecepatan inferensi onboard $\ge 15\text{ FPS}$ (latensi budget $\le 67\text{ ms}$).

---

## Metode

Pelatihan dijalankan menggunakan skrip `train.py` pada PyTorch dengan input $224 \times 224$ piksel, normalisasi standar ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), augmentasi acak pada data training (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`), seed 42, batch size 16, dan 10 epoch.

Ketiga mode pelatihan menggunakan data, split, dan preprocessing yang sama:

1. **Feature Extraction** — backbone ResNet18 pretrained dibekukan (`requires_grad = False`); hanya classifier (`fc`) akhir yang dilatih.
2. **Partial Fine-Tuning** — `layer4` akhir dan classifier (`fc`) dilatih dengan *discriminative learning rate* (`1e-4` untuk layer4, `1e-3` untuk fc), sedangkan layer awal dibekukan.
3. **Training from Scratch** — ResNet18 dilatih dari nol tanpa bobot pretrained (inisialisasi bobot acak).

---

## Hasil Eksperimen

| Mode | Validation accuracy terbaik | Epoch terbaik | Waktu training |
| :--- | :---: | :---: | :---: |
| **Partial Fine-Tuning** | **100,00%** | **1** | **~8,45 detik** |
| **Feature Extraction** | **100,00%** | **4** | **~7,92 detik** |
| **Training from Scratch** | **100,00%** | **4** | **~8,15 detik** |

Model terpilih adalah **Partial Fine-Tuning** karena memperoleh *validation accuracy* tertinggi ($100,00\%$) dan mencapai akurasi tersebut paling cepat, yaitu pada **epoch 1**. Feature Extraction memiliki waktu training yang sedikit lebih rendah, tetapi konvergensinya membutuhkan lebih banyak epoch. Latensi inference Partial Fine-Tuning adalah **1,247 ms per citra** pada GPU environment eksperimen (setara $\approx 800\text{ FPS}$).

Hasil lengkap tersedia di:

- `results/experiments.csv` — tabel perbandingan akurasi dan latensi.
- `results/accuracy_comparison.png` — grafik kurva akurasi validasi per epoch.
- `results/feature_history.csv` — log riwayat pelatihan mode Feature Extraction.
- `results/partial_history.csv` — log riwayat pelatihan mode Partial Fine-Tuning.
- `results/scratch_history.csv` — log riwayat pelatihan mode Scratch.
- `models/resnet18_partial.pth` — bobot model Partial Fine-Tuning terpilih.
- `metadata.csv` — manifest path, label numerik, dan status split.

---

## Evaluasi dan Analisis Keterbatasan

1. **Efektivitas Transfer Learning:** Transfer learning memberikan hasil sangat baik pada dataset ini. Partial Fine-Tuning mencapai akurasi 100% pada epoch pertama karena fitur visual tingkat tinggi pada `layer4` cepat menyesuaikan dengan karakteristik tekstur dan warna box.
2. **Perbandingan dengan Scratch:** Mode Scratch juga mencapai akurasi 100% pada epoch 4, namun pada epoch awal (epoch 1 dan 3) mengalami fluktuasi akurasi rendah ($50,00\%$ / setara tebakan acak) karena memulai pelatihan tanpa *prior knowledge*.
3. **Pencegahan Data Leakage:** Pemisahan dataset menggunakan *chronological split* dan *guard band* pada `split.py` berhasil mencegah kebocoran informasi antar-frame video yang berurutan, sehingga evaluasi pada data validasi benar-benar obyektif.
4. **Evaluasi Latensi:** Latensi inferensi sebesar **1,247 ms** berada jauh di bawah ambang batas $67\text{ ms}$ (budget 15 FPS), menandakan model ResNet18 sangat efisien untuk diterapkan pada robot industri secara *real-time*.
5. **Keterbatasan:** Ukuran dataset masih terbatas pada 200 citra dengan kondisi latar belakang lab yang relatif konstan. Untuk pengujian lapangan terbuka, disarankan menambah variasi *background* dan pencahayaan ekstrim.

---

##  Panduan Lengkap Cara Menjalankan Kode (Step-by-Step)

Langkah 1: Persiapan Environment & Instalasi Dependencies
Pastikan berada di direktori utama proyek /home/yanzz/Documents/P2 - Transfer Learning. Buat dan aktifkan lingkungan Python virtual (venv), lalu install seluruh pustaka yang diperlukan:

  python3 -m venv venv
  ./venv/bin/pip install torch torchvision pillow matplotlib

Langkah 2: Pemisahan Dataset (split.py)
Skrip ini membagi dataset di metadata.csv menjadi subset train dan validation secara chronological/temporal split dengan menambahkan guard band (gap) untuk mencegah data leakage.

  ./venv/bin/python split.py --val-fraction 0.2 --gap 2

Output Terminal:
  box_cokelat: train=78 validation=18 excluded=4
  box_merah: train=78 validation=18 excluded=4

Langkah 3: Pelatihan Model 3 Mode (train.py)
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

Langkah 4: Pengukuran Kecepatan Inferensi (latency.py)
Mengukur latensi waktu eksekusi tebakan model untuk 1 gambar (batch size = 1) di hardware target.

  ./venv/bin/python latency.py --checkpoint models/resnet18_feature.pth --image box_merah/box_merah__frame_000000000.jpg
  ./venv/bin/python latency.py --checkpoint models/resnet18_partial.pth --image box_merah/box_merah__frame_000000000.jpg
  ./venv/bin/python latency.py --checkpoint models/resnet18_scratch.pth --image box_merah/box_merah__frame_000000000.jpg

Output Terminal:
  cuda: median=1.247 ms, mean=1.248 ms, p95=1.252 ms

Langkah 5: Uji Prediksi Foto Baru (predict.py)
Menguji kemampuan model terlatih dalam mengenali gambar box secara langsung:

  ./venv/bin/python predict.py box_cokelat/box_cokelat__frame_000000000.jpg
  ./venv/bin/python predict.py box_merah/box_merah__frame_000000000.jpg

