"""
Analisis Pola Pembelian Weekday dan Weekend - Toko Mastaufik Frozen Food
Mereproduksi angka utama naskah: ringkasan dataset, operasi OLAP, uji Mann-Whitney U,
Apriori (keseluruhan dan per segmen), analisis sensitivitas min_count, dan keterkaitan antarkategori.

Jalankan:  python analysis.py
Keluaran:  folder results/ (tabel CSV dan gambar PNG)
"""
import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

DATA = "data/dataset_nota_clean.csv"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

ITEM = "Nama Produk Clean"
df = pd.read_csv(DATA)
HAS_PRICE = "Harga Total" in df.columns   # jika kolom harga dihapus, bagian omset dilewati

# ---------------------------------------------------------------- 1. Ringkasan dataset
nota = df.groupby("No_Nota").agg(Segment=("Segment", "first"), n_item=(ITEM, "size"))
print("== Ringkasan dataset ==")
print("Nota:", len(nota), nota.Segment.value_counts().to_dict())
print("Baris item:", len(df), df.Segment.value_counts().to_dict())
print("Produk unik / kategori:", df[ITEM].nunique(), "/", df.Kategori.nunique())
print("Periode:", df.Tanggal.min(), "s.d.", df.Tanggal.max())
print("Rata-rata item/nota: %.2f" % nota.n_item.mean())
print(nota.groupby("Segment").n_item.agg(["mean", "median"]).round(2))

# ---------------------------------------------------------------- 2. OLAP
if HAS_PRICE:
    print("\n== OLAP ==")
    print("Total penjualan:", int(df["Harga Total"].sum()))
    bulan = df.groupby("Bulan")["Harga Total"].sum()
    print(bulan.to_string())
    seg_kat = df.pivot_table(index="Kategori", columns="Segment", values="Harga Total", aggfunc="sum", fill_value=0)
    share = (seg_kat / seg_kat.sum() * 100).round(1)
    share.to_csv(f"{OUT}/olap_porsi_kategori_per_segmen.csv")
    seg_kat.to_csv(f"{OUT}/olap_omset_kategori_per_segmen.csv")
    print((share).to_string())
    print("Total per segmen:", seg_kat.sum().to_dict())
    wk = df[df.Segment == "Weekend"]
    print("Slice Weekend - kategori tertinggi:",
          wk.groupby("Kategori").agg(omset=("Harga Total", "sum"), unit=("QTY", "sum")).sort_values("omset", ascending=False).head(1).to_dict("index"))
    dice = wk[wk.Kategori.isin(["Nugget & Sosis", "Bumbu dan saus"])].pivot_table(
        index="Kategori", columns="Bulan", values="Harga Total", aggfunc="sum", fill_value=0)
    print("Dice Weekend x 2 kategori:\n", dice.to_string())
    dd = df[df.Kategori == "Nugget & Sosis"].groupby(ITEM).agg(omset=("Harga Total", "sum"), unit=("QTY", "sum")).sort_values("omset", ascending=False)
    print("Drill-down Nugget & Sosis (top 1):\n", dd.head(1).to_string())

# ---------------------------------------------------------------- 3. Uji Mann-Whitney U
print("\n== Mann-Whitney U ==")
wd = nota[nota.Segment == "Weekday"]; we = nota[nota.Segment == "Weekend"]
def mw(a, b, label):
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    u_min = min(u, n1 * n2 - u)
    r = abs(1 - 2 * u_min / (n1 * n2))           # rank-biserial (ukuran efek)
    print(f"{label}: U = {u_min:.1f}, p = {p:.3f}, r = {r:.2f}, n = {n1} dan {n2}")
    return dict(uji=label, U=u_min, p=p, r=r, n_weekday=n1, n_weekend=n2)
res = [mw(wd.n_item, we.n_item, "Jumlah item per nota")]
if HAS_PRICE:
    val = df.groupby("No_Nota")["Harga Total"].sum()
    res.append(mw(val[wd.index], val[we.index], "Nilai transaksi per nota"))
pd.DataFrame(res).to_csv(f"{OUT}/uji_mann_whitney.csv", index=False)

# Gambar 3 (versi per segmen)
fig, ax = plt.subplots(figsize=(6, 3.6))
data = [wd.n_item.values, we.n_item.values]
ax.boxplot(data, tick_labels=["Weekday (n=33)", "Weekend (n=25)"], widths=0.5, showmeans=True)
rng = np.random.default_rng(0)
for i, d in enumerate(data, 1):
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=14, alpha=0.5, color="tab:blue", zorder=3)
ax.set_ylabel("Jumlah item per nota"); ax.set_title("Distribusi jumlah item per nota (basket size)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/gambar3_basket_size_per_segmen.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 4. Apriori
def transaksi(sub, col=ITEM):
    return sub.groupby("No_Nota")[col].apply(lambda s: sorted(set(s))).tolist()

def encode(tx):
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)

def mine(sub, min_count, conf, col=ITEM):
    """Apriori dengan batas minimum jumlah nota absolut (min_count), lalu aturan dengan lift > 1."""
    tx = transaksi(sub, col); N = len(tx); X = encode(tx)
    fi = apriori(X, min_support=(min_count - 0.5) / N, use_colnames=True)
    fi["count"] = (fi.support * N).round().astype(int)
    fi = fi[fi["count"] >= min_count]
    rules = association_rules(fi, metric="confidence", min_threshold=conf)
    rules = rules[rules.lift > 1].copy()
    rules["n_nota"] = (rules.support * N).round().astype(int)
    rules["kombinasi"] = [frozenset(a | c) for a, c in zip(rules.antecedents, rules.consequents)]
    return N, fi, rules

def ceil_count(N, sup=0.05):
    return int(np.ceil(sup * N - 1e-9))

print("\n== Apriori keseluruhan (support 5% = min 3 nota) ==")
for conf in (0.30, 0.50):
    N, fi, r = mine(df, ceil_count(58), conf)
    print(f"confidence {int(conf*100)}%: itemset sering = {len(fi)}, aturan = {len(r)}")
N, fi, r_all = mine(df, ceil_count(58), 0.50)
tab4 = (r_all.assign(kombinasi_txt=r_all.kombinasi.apply(lambda s: " + ".join(sorted(s))))
        .groupby("kombinasi_txt").agg(jumlah_nota=("n_nota", "first"), lift_min=("lift", "min"), lift_max=("lift", "max"))
        .reset_index().sort_values(["jumlah_nota", "lift_max"], ascending=False))
tab4.to_csv(f"{OUT}/apriori_kombinasi_keseluruhan_conf50.csv", index=False)
print(tab4.head(10).to_string(index=False))

print("\n== Apriori per segmen (confidence 50%, lift > 1) ==")
rows = []; seg_rules = {}
for k in (2, 3, 4, 5):
    rec = {"min_nota": k}
    combos = {}
    for seg in ("Weekday", "Weekend"):
        N, fi, r = mine(df[df.Segment == seg], k, 0.50)
        combos[seg] = set(r.kombinasi)
        rec[f"{seg}_aturan"] = len(r); rec[f"{seg}_kombinasi"] = len(combos[seg])
        if k == 2:
            seg_rules[seg] = r
    rec["kombinasi_sama"] = len(combos["Weekday"] & combos["Weekend"])
    rows.append(rec)
    print(rec)
    if k == 2:
        print("   kombinasi sama:", [" + ".join(sorted(s)) for s in combos["Weekday"] & combos["Weekend"]])
sens = pd.DataFrame(rows); sens.to_csv(f"{OUT}/sensitivitas_min_count.csv", index=False)
for seg, r in seg_rules.items():
    out = r.assign(antecedents=r.antecedents.apply(lambda s: " + ".join(sorted(s))),
                   consequents=r.consequents.apply(lambda s: " + ".join(sorted(s)))
                   )[["antecedents", "consequents", "n_nota", "support", "confidence", "lift"]]
    out.sort_values(["n_nota", "lift"], ascending=False).to_csv(f"{OUT}/aturan_{seg.lower()}_mincount2.csv", index=False)

# Lift maksimum = N / n_B (alasan lift tinggi pada item jarang)
print("\n== Contoh lift maksimum (N / n_B) ==")
for n_b in (3, 13):
    print(f"n_B = {n_b}: lift maks = {58 / n_b:.2f}")

# ---------------------------------------------------------------- 5. Keterkaitan antarkategori
print("\n== Keterkaitan antarkategori ==")
kat = df.groupby("No_Nota").Kategori.apply(set)
seg_of = nota.Segment
cats = sorted(df.Kategori.unique())
rows = []
for a, b in combinations(cats, 2):
    has = kat.apply(lambda s: a in s and b in s)
    rows.append({"kategori_1": a, "kategori_2": b, "semua": int(has.sum()),
                 "weekday": int(has[seg_of[has.index] == "Weekday"].sum()),
                 "weekend": int(has[seg_of[has.index] == "Weekend"].sum())})
pairs = pd.DataFrame(rows).sort_values("semua", ascending=False)
pairs.to_csv(f"{OUT}/kategori_pasangan_cooccurrence.csv", index=False)
print(pairs.head(4).to_string(index=False))

# Heatmap co-occurrence (diagonal = jumlah nota yang memuat kategori)
M = pd.DataFrame(0, index=cats, columns=cats)
for s in kat:
    for a in s:
        for b in s:
            M.loc[a, b] += 1
fig, ax = plt.subplots(figsize=(8, 6.8))
off = M.where(~np.eye(len(cats), dtype=bool))
im = ax.imshow(off.astype(float), cmap="YlGnBu")
for i in range(len(cats)):
    for j in range(len(cats)):
        if i == j:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#b3b3b3"))
            ax.text(j, i, f"n={M.iloc[i, j]}", ha="center", va="center", style="italic", fontsize=8)
        else:
            v = M.iloc[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if v > 28 else "black")
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=45, ha="right")
ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
ax.set_title("Co-occurrence antarkategori\n(sel abu-abu = jumlah nota yang memuat kategori tersebut)")
fig.colorbar(im, label="Jumlah nota yang memuat kedua kategori")
fig.tight_layout(); fig.savefig(f"{OUT}/gambar6_heatmap_cooccurrence.png", dpi=200); plt.close(fig)
print("\nSelesai. Hasil tersimpan di folder results/")
