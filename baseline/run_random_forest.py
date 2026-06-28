import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_prep import load_and_prepare
from models.random_forest import build_model, build_model_balanced
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df_main_with_weather = pd.read_excel("main_data_with_weather.xlsx")
X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = load_and_prepare(
    df_main_with_weather
)

rf_model = build_model()
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print("Прогноз на следующий день (Random Forest)")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["OK (>=99.8%)", "Problem (<99.8%)"], digits=4
    )
)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

feature_names = list(X_train.columns)
importances = rf_model.feature_importances_
df_imp = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values("Importance", ascending=False)

print("Важность признаков (Random Forest)")
print(df_imp.head(15).to_string(index=False))

plt.figure(figsize=(10, 8))
top_n = 15
sns.barplot(x="Importance", y="Feature", data=df_imp.head(top_n), palette="viridis")
plt.title(f"Топ {top_n} признаков (Random Forest)", fontsize=14)
plt.xlabel("Gini Importance", fontsize=12)
plt.ylabel("Признак", fontsize=12)
plt.tight_layout()
plt.show()

rf_model = build_model_balanced()
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print("Прогноз на следующий день с балансировкой (Random Forest)")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["OK (>=99.8%)", "Problem (<99.8%)"], digits=4
    )
)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

feature_names = list(X_train.columns)
importances = rf_model.feature_importances_
df_imp = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values("Importance", ascending=False)

print("Важность признаков (Random Forest)")
print(df_imp.head(15).to_string(index=False))

plt.figure(figsize=(10, 8))
top_n = 15
sns.barplot(x="Importance", y="Feature", data=df_imp.head(top_n), palette="viridis")
plt.title(f"Топ {top_n} признаков (Random Forest)", fontsize=14)
plt.xlabel("Gini Importance", fontsize=12)
plt.ylabel("Признак", fontsize=12)
plt.tight_layout()
plt.show()
