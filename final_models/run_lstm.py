import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve,
)

from data_prep import load_and_prepare
from models.lstm import LSTMModel, DiceLoss

df_main_with_weather = pd.read_excel("main_data_with_weather.xlsx")
X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = load_and_prepare(df_main_with_weather)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

X_train_t = torch.tensor(X_train_lstm, dtype=torch.float32)
X_test_t = torch.tensor(X_test_lstm, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t, y_train_t = X_train_t.to(device), y_train_t.to(device)
X_test_t, y_test_t = X_test_t.to(device), y_test_t.to(device)

model = LSTMModel(
    input_size=X_train_lstm.shape[2],
    hidden_size=64,
    num_layers=2
).to(device)

pos_weight = torch.tensor([scale_pos_weight], dtype=torch.float32).to(device)

bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice_loss = DiceLoss()


def criterion(logits, targets):
    return bce_loss(logits, targets) + dice_loss(logits, targets)


optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
model.train()

epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(X_train_t).squeeze()
    loss = criterion(logits, y_train_t)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

baseline_prob = y_train_t.mean().item()
baseline_pred = (torch.full_like(y_test_t, baseline_prob) >= 0.5).int().cpu().numpy()
print("baseline")
print(classification_report(y_test, baseline_pred))

model.eval()

with torch.no_grad():
    logits = model(X_test_t).squeeze()

    y_prob_lstm = torch.sigmoid(logits).cpu().numpy()
    y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

precision, recall, thresholds = precision_recall_curve(y_test, y_prob_lstm)

pr_auc = average_precision_score(y_test, y_prob_lstm)

print("lstm")
print(classification_report(y_test, y_pred_lstm))
print(confusion_matrix(y_test, y_pred_lstm))
print("roc-auc ", roc_auc_score(y_test, y_prob_lstm))
print("pr-auc ", pr_auc)
