import lightgbm as lgb


DEFAULT_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
)


def build_model(scale_pos_weight, **params):
    return lgb.LGBMClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )


def suggest_params(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 30, 2000),
        max_depth=trial.suggest_int("max_depth", 1, 13),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        subsample=trial.suggest_float("subsample", 0.1, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )
