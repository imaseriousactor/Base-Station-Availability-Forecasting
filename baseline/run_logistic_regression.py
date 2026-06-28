import pandas as pd
from data_prep import load_and_prepare
from models.logistic_regression import build_model, build_model_balanced, build_scaler
from sklearn.metrics import classification_report, roc_auc_score

df_main_with_weather = pd.read_excel("main_data_with_weather.xlsx")
X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = load_and_prepare(
    df_main_with_weather
)

X_train_copy = X_train.copy()
X_train_copy["target_today"] = train_df["target_today"].values

corr_today = (
    X_train_copy.corr(numeric_only=True)["target_today"]
    .drop("target_today")
    .sort_values(ascending=False)
)

print("Корреляция признаков с сегодняшними target:")
print(corr_today.head(10))

scaler = build_scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = build_model()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Прогноз на следующий день (Logistic Regression)")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(
    classification_report(
        y_test, y_pred, target_names=["OK (>=99.8%)", "Problem (<99.8%)"], digits=4
    )
)

coef_df = pd.DataFrame({"feature": feature_cols, "coef": model.coef_[0]}).sort_values(
    "coef", key=abs, ascending=False
)
print("Топ признаков:")
print(coef_df.head(10).to_string(index=False))

df_model = train_df.copy()
df_model["target_today"] = (df_model["Cell Avail Base Tech (%)"] < 99.8).astype(int)
corr = df_model["target_today"].corr(df_model["target_next_day"])
print(f"Автокорреляция целевой переменной: {corr:.4f}")

model = build_model_balanced()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "=" * 60)
print("Прогноз на следующий день с балансировкой классов (Logistic Regression)")
print("=" * 60)
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(
    classification_report(
        y_test, y_pred, target_names=["OK (>=99.8%)", "Problem (<99.8%)"], digits=4
    )
)

coef_df = pd.DataFrame({"feature": feature_cols, "coef": model.coef_[0]}).sort_values(
    "coef", key=abs, ascending=False
)
print("Топ признаков:")
print(coef_df.head(10).to_string(index=False))
