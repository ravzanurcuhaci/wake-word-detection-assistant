import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

print("Veriseti yükleniyor...")
df = pd.read_csv("dataset.csv")

X = df.drop(columns=["filename", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("SVM Modeli Eğitiliyor (Grid Search)...")

# class_weight='balanced' dengesiz veri kümesini ele alır.
# live_detect_gui proba() fonksiyonlarına bağımlı olduğu için probability=True zorunlu.
svc = SVC(probability=True, random_state=42, class_weight='balanced')

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto", 0.01],
    "kernel": ["rbf", "linear"]
}

grid = GridSearchCV(
    svc,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nEn iyi parametreler:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(best_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nYeni Model kaydedildi: svm_model.pkl (SVC)")
print("Scaler kaydedildi: scaler.pkl")