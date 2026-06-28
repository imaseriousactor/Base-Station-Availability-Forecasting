from sklearn.ensemble import RandomForestClassifier


def build_model():
    """Без балансировки классов."""
    return RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )


def build_model_balanced():
    """С балансировкой классов."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
