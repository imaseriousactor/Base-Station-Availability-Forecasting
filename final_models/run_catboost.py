import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve,
)

from data_prep import load_and_prepare
from cv_utils import run_time_series_cv
from models.catboost_model import build_model, suggest_params, DEFAULT_PARAMS

df_main_with_weather = pd.read_excel("main_data_with_weather.xlsx")
X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = load_and_prepare(df_main_with_weather)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


def objective(trial):
    params = suggest_params(trial)

    def get_model():
        return build_model(scale_pos_weight, **params)

    cv_res = run_time_series_cv(X_train, y_train, get_model, n_splits=3, verbose=False)
    return cv_res['mean_auc']


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

best_params = study.best_params

model_cb = build_model(scale_pos_weight, **best_params)
model_cb.fit(X_train, y_train)

y_pred_cb = model_cb.predict(X_test)
y_prob_cb = model_cb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_cb))
print(confusion_matrix(y_test, y_pred_cb))
print("roc-auc ", roc_auc_score(y_test, y_prob_cb))
print("pr-auc ", average_precision_score(y_test, y_prob_cb))

study.trials_dataframe().to_csv("optuna_results_catboost.csv")
print(f"Best params: {best_params}")
print(f"Best AUC: {study.best_value}")


model_cb = build_model(scale_pos_weight, **DEFAULT_PARAMS)
model_cb.fit(X_train, y_train)

y_pred_cb = model_cb.predict(X_test)
y_prob_cb = model_cb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_cb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_cb))
print("PR-AUC: ", average_precision_score(y_test, y_prob_cb))


precision, recall, thresholds = precision_recall_curve(y_test, y_prob_cb)
pr_auc = average_precision_score(y_test, y_prob_cb)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.grid(True)
plt.show()


feat_imp_cb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_cb.feature_importances_
}).sort_values('importance', ascending=False)
print(feat_imp_cb.head(10))
