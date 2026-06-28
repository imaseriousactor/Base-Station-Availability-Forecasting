import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from data_prep import load_and_prepare
from models.naive_bayes import build_model

df_main_with_weather = pd.read_excel("main_data_with_weather.xlsx")
X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = load_and_prepare(df_main_with_weather)

model = build_model()
model.fit(X_train, y_train)

y_pred_default = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred_default):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_default))
print("Отчёт по классам:")
print(classification_report(y_test, y_pred_default, target_names=['Нет аварии', 'Авария']))
