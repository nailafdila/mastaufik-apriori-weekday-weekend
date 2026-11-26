# Analisis Pola Pembelian Weekday dan Weekend — Toko Mastaufik Frozen Food (Apriori + OLAP)

Repositori ini berisi dataset dan kode untuk artikel **"Analisis Pola Pembelian Weekday dan Weekend pada Toko Mastaufik Frozen Food Menggunakan Algoritma Apriori"** (Jurnal Restikom).

Penulis: Narsa Elvanisa, Nailah Nur Fadila, Nova Noor Kamala Sari — Teknik Informatika, Universitas Palangka Raya.

## Isi repositori

| Berkas | Keterangan |
|---|---|
| `data/dataset_nota_clean.csv` | Dataset setelah preprocessing (310 baris item, 58 nota) — masukan `analysis.py` |
| `data/dataset_nota_raw.csv` | Data nota sebelum preprocessing (untuk audit; lihat Lampiran 1 artikel) |
| `analysis.py` | Seluruh analisis: ringkasan data, OLAP, uji Mann-Whitney U, Apriori, sensitivitas `min_count`, keterkaitan kategori |
| `results/` | Keluaran skrip (tabel CSV dan gambar PNG) |
| `requirements.txt` | Pustaka Python yang dibutuhkan |

## Kamus kolom (`dataset_nota_clean.csv`)

| Kolom | Isi |
|---|---|
| `No_Nota` | Nomor nota anonim (TRX001–TRX058) |
| `Tanggal` | Tanggal transaksi (YYYY-MM-DD), 2026-08-01 s.d. 2026-09-21 |
| `Nama Produk` | Nama produk sesuai nota |
| `Kategori` | 8 kategori: Sembako, Nugget & Sosis, Bumbu dan saus, Daging dan Ayam, Sayur, Olahan Ikan, Kemasan, Cemilan Gorengan |
| `QTY`, `Harga Satuan`, `Harga Total` | Jumlah, harga satuan, dan harga total (Rupiah) |
| `Nama Produk Clean` | Nama produk setelah standardisasi ejaan (dipakai sebagai item Apriori) |
| `DayName`, `Segment`, `Bulan`, `Minggu` | Hari, segmen (Weekday = Senin–Jumat, Weekend = Sabtu–Minggu), bulan, dan minggu ke-n |

Data tidak memuat identitas pelanggan.

## Cara menjalankan

```bash
pip install -r requirements.txt
python analysis.py
```

Skrip mencetak ringkasan ke layar dan menyimpan tabel serta gambar ke folder `results/`.
Seluruh angka utama pada artikel (58 nota; 43 dan 38 aturan; 45 dan 57 aturan per segmen; Tabel 6; U = 282,5, p = 0,038) dapat direproduksi dengan skrip ini.

Parameter utama: minimum *support* 5% (setara 3 nota pada 58 nota, dan 2 nota pada tiap segmen), minimum *confidence* 30% dan 50%, hanya aturan dengan *lift* > 1. Pustaka: pandas, scipy, mlxtend, matplotlib.

## Etika dan penggunaan data

Data nota diberikan langsung oleh pemilik toko dengan persetujuan untuk keperluan akademik. Informasi sensitif seperti harga pokok dan margin keuntungan tidak diminta dan tidak disertakan.

## Lisensi

- Kode: MIT (lihat `LICENSE`).
- Data: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), sejalan dengan lisensi artikel.

## Cara mengutip

Tambahkan di sini setelah artikel terbit (judul, jurnal, tahun, DOI artikel) dan DOI Zenodo repositori ini bila ada.
