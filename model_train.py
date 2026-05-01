# PCOS Detection with Improved Accuracy (Stacking + Hyperparameter Tuning + Model Comparison)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("PCOS_data.csv")  # <-- replace with dataset path

print("Dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Drop patient ID if present
if 'Patient File No.' in df.columns:
    df = df.drop('Patient File No.', axis=1)

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Features & Target
target_col = "PCOS (Y/N)"
X = df.drop(target_col, axis=1)
y = df[target_col]

# -------------------------
# 2. Handle Missing + Imbalance
# -------------------------
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# -------------------------
# 3. Feature Selection
# -------------------------
selector = SelectKBest(score_func=f_classif, k=15)  # select top 15 features
X_new = selector.fit_transform(X_res, y_res)
selected_features = X.columns[selector.get_support()]
print("\nSelected Features:", list(selected_features))

# -------------------------
# 4. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# -------------------------
# 5. Hyperparameter Tuning - Random Forest
# -------------------------
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("\nBest RandomForest Params:", rf_grid.best_params_)

# -------------------------
# 6. Hyperparameter Tuning - XGBoost
# -------------------------
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    xgb_params, cv=5, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print("\nBest XGBoost Params:", xgb_grid.best_params_)

# -------------------------
# 7. Train Individual Models
# -------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["No PCOS","PCOS"],
                yticklabels=["No PCOS","PCOS"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# -------------------------
# 8. Stacking Ensemble
# -------------------------
stacking = StackingClassifier(
    estimators=[
        ('dt', models["Decision Tree"]),
        ('svm', models["SVM"]),
        ('rf', models["Random Forest"]),
        ('xgb', models["XGBoost"])
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
acc = accuracy_score(y_test, y_pred)
results["Stacking Ensemble"] = acc

print("\n===== Stacking Ensemble =====")
print("Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["No PCOS","PCOS"],
            yticklabels=["No PCOS","PCOS"])
plt.title("Confusion Matrix - Stacking Ensemble")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------
# 9. Model Comparison
# -------------------------
print("\nModel Comparison (Accuracy):")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.show()

# -------------------------
# 10. Save Final Model
# -------------------------
with open("pcos_best_model.pkl", "wb") as f:
    pickle.dump(stacking, f)

print("\n✅ Final Stacking Model saved as pcos_best_model.pkl")
