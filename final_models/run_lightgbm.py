import matplotlib.pyplot as plt
import optuna
import pandas as pd
from cv_utils import run_time_series_cv
from data_prep import load_and_prepare
from models.lightgbm_model import DEFAULT_PARAMS, build_model, suggest_params
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

df_main_with_weather = pd.read_excel("main_data_with_weather.xlsx")
X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = load_and_prepare(
    df_main_with_weather
)

print(f"Колонок до очистки: {X_train.shape[1]}")
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]
print(f"Колонок после очистки: {X_train.shape[1]}")

assert X_train.columns.is_unique, "Есть дубли в X_train!"
assert X_test.columns.is_unique, "Есть дубли в X_test!"

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


def objective(trial):
    params = suggest_params(trial)

    def get_model():
        return build_model(scale_pos_weight, **params)

    cv_res = run_time_series_cv(X_train, y_train, get_model, n_splits=3, verbose=False)
    return cv_res["mean_auc"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params

model_lgb = build_model(scale_pos_weight, **best_params)
model_lgb.fit(X_train, y_train)

y_pred_lgb = model_lgb.predict(X_test)
y_prob_lgb = model_lgb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_lgb))
print(confusion_matrix(y_test, y_pred_lgb))
print("roc-auc ", roc_auc_score(y_test, y_prob_lgb))
print("pr-auc ", average_precision_score(y_test, y_prob_lgb))

study.trials_dataframe().to_csv("optuna_results_lgb.csv")
print(f"Best params: {best_params}")
print(f"Best AUC: {study.best_value}")


model_lgb = build_model(scale_pos_weight, **DEFAULT_PARAMS)
model_lgb.fit(X_train, y_train)

y_pred_lgb = model_lgb.predict(X_test)
y_prob_lgb = model_lgb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_lgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lgb))
print("PR-AUC: ", average_precision_score(y_test, y_prob_lgb))


precision, recall, thresholds = precision_recall_curve(y_test, y_prob_lgb)
pr_auc = average_precision_score(y_test, y_prob_lgb)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.grid(True)
plt.show()


feat_imp_lgb = pd.DataFrame(
    {"feature": X_train.columns, "importance": model_lgb.feature_importances_}
).sort_values("importance", ascending=False)
print(feat_imp_lgb.head(10))
