import catboost as cb

DEFAULT_PARAMS = dict(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
)


def build_model(scale_pos_weight, **params):
    return cb.CatBoostClassifier(
        **params, scale_pos_weight=scale_pos_weight, random_seed=42, verbose=0
    )


def suggest_params(trial):
    return dict(
        iterations=trial.suggest_int("iterations", 100, 2000),
        depth=trial.suggest_int("depth", 1, 12),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0, 1),
        random_strength=trial.suggest_float("random_strength", 1e-3, 10, log=True),
    )
