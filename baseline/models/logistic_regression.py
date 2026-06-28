from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_scaler():
    return StandardScaler()


def build_model():
    """Без балансировки классов."""
    return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, solver="lbfgs")


def build_model_balanced():
    """С балансировкой классов."""
    return LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced", solver="lbfgs"
    )
