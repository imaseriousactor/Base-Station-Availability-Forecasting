import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def run_time_series_cv(X, y, model_factory, n_splits=5, use_scaler=True, verbose=True):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    cv_models = []
    oof_proba = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_tr, y_tr)

        y_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_proba)
        cv_scores.append(score)
        cv_models.append(model)
        oof_proba[val_idx] = y_proba

        if verbose:
            print(
                f"{fold} | ROC-AUC {score:.4f} | Train {len(X_tr)} | Val {len(X_val)}"
            )

    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    if verbose:
        print(f"CV Result: {mean_auc:.4f} ± {std_auc:.4f}")

    return {
        "scores": cv_scores,
        "models": cv_models,
        "oof_proba": oof_proba,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }
