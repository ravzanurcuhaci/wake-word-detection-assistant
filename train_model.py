import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
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

print("Gradient Boosting Modeli Eğitiliyor...")
# Sınıf dengesizliği için örnek ağırlıkları hesaplayalım
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

gb = GradientBoostingClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5]
}

grid = GridSearchCV(
    gb,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train, sample_weight=sample_weights)

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

print("\nYeni Model kaydedildi: svm_model.pkl (Gerçekte Gradient Boosting)")
print("Scaler kaydedildi: scaler.pkl")