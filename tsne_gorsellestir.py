import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

print("dataset.csv yükleniyor...")
df = pd.read_csv("dataset.csv")

print("Özellikler ölçeklendiriliyor...")
X = df.drop(columns=["filename", "label"])
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Toplam {X.shape[0]} adet veri PCA ile 30 boyuta indirgeniyor...")
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("PCA Sonrası t-SNE ile 2 boyuta indirgeniyor...")
print("Bu işlem verinin büyüklüğüne göre 15-30 saniye sürebilir...")
tsne = TSNE(
    n_components=2, 
    perplexity=40,
    early_exaggeration=50,
    random_state=42
)
X_tsne = tsne.fit_transform(X_pca)

tsne_df = pd.DataFrame(data=X_tsne, columns=["TSNE_Boyut_1", "TSNE_Boyut_2"])
tsne_df["Sınıf"] = y.map({1: "Wake", 0: "Non-Wake"})

print("Grafik çiziliyor...")
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="TSNE_Boyut_1", 
    y="TSNE_Boyut_2", 
    hue="Sınıf", 
    palette={"Wake": "green", "Non-Wake": "red"}, 
    data=tsne_df, 
    alpha=0.6,    # Üst üste binen noktaların görünürlüğünü artırmak için saydamlık artırıldı
    s=20,         # Nokta boyutu küçültüldü
    edgecolor="w",
    linewidth=0.2
)

plt.title("Wake Word 78 Özellik için t-SNE Grafiği")
plt.xlabel("T-SNE Boyut 1")
plt.ylabel("T-SNE Boyut 2")
plt.grid(True, linestyle="--", alpha=0.5)

# Grafiği dosya olarak da kaydedelim (her ihtimale karşı)
output_file = "tsne_plot.png"
plt.savefig(output_file, dpi=300)
print(f"Grafik başarıyla '{output_file}' olarak kaydedildi.")

print("Ekranda gösteriliyor... (Pencereyi kapattığınızda işlem tamamlanacaktır)")
plt.show()
