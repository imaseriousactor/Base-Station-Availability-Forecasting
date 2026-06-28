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
from models.xgboost_model import build_model, suggest_params, DEFAULT_PARAMS

df_main_with_weather = pd.read_excel("main_data_with_weather.xlsx")
X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = load_and_prepare(df_main_with_weather)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


def objective(trial):
    params = suggest_params(trial)

    def get_model():
        return build_model(scale_pos_weight, **params)

    cv_res = run_time_series_cv(X_train, y_train, get_model, n_splits=3, use_scaler=False, verbose=False)
    return cv_res['mean_auc']


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
best_params = study.best_params

model_xgb = build_model(scale_pos_weight, **best_params)
model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)
y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
print("roc-auc ", roc_auc_score(y_test, y_prob_xgb))
print("pr-auc ", average_precision_score(y_test, y_prob_xgb))

print(f"Best params: {best_params}")
print(f"Best AUC: {study.best_value}")


model_xgb = build_model(scale_pos_weight, **DEFAULT_PARAMS)
model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)
y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print("PR-AUC: ", average_precision_score(y_test, y_prob_xgb))


feat_imp_xgb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=False)
print(feat_imp_xgb.head(10))


hist_features = ['Failures_7d', 'Failures_30d', 'Days_Since_Fail',
                  '4G_EMA_3', 'BaseTech_EMA_3', 'Hist_Avg_Avail']
hist_to_drop = [f for f in hist_features if f in X_train.columns]

print(f"удаляем {len(hist_to_drop)} исторических признаков:")
print(f"   {hist_to_drop}")

X_train_abl = X_train.drop(columns=hist_to_drop)
X_test_abl = X_test.drop(columns=hist_to_drop)

model_xgb_abl = build_model(scale_pos_weight, **best_params)
model_xgb_abl.fit(X_train_abl, y_train)

y_pred_abl = model_xgb_abl.predict(X_test_abl)
y_prob_abl = model_xgb_abl.predict_proba(X_test_abl)[:, 1]

auc_full = roc_auc_score(y_test, y_prob_xgb)
auc_abl = roc_auc_score(y_test, y_prob_abl)
pr_full = average_precision_score(y_test, y_prob_xgb)
pr_abl = average_precision_score(y_test, y_prob_abl)

print("Ablation")
print(classification_report(y_test, y_pred_abl, digits=4))
print(f"roc-auc: {auc_abl:.4f}")
print(f"pr-auc:  {pr_abl:.4f}")

