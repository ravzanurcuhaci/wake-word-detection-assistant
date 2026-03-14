import pandas as pd

df = pd.read_csv("dataset.csv")

print("Veri boyutu:", df.shape)

print("\nLabel dağılımı:")
print(df["label"].value_counts())

print("\nBoş değer sayısı:")
print(df.isnull().sum().sum())