import pandas as pd
from generate_features import (
    add_alarm_features,
    add_cyclical_features,
    add_weather_composite_features,
    create_features,
)

tech_cols = [
    "Cell Avail 2G (%)",
    "Cell Avail 3G (%)",
    "Cell Avail 4G (%)",
    "Cell Avail Base Tech (%)",
]
weather_cols = [
    "TempDelta",
    "MaxWind",
    "MaxInt",
    "Rain",
    "Snow",
    "Sprinkling",
    "Hail",
    "U",
    "Po",
]
cyclical_cols = [
    "day_sin",
    "day_cos",
    "day_month_sin",
    "day_month_cos",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "season_sin",
    "season_cos",
]
interaction_cols = [
    "Storm_Intensity",
    "Ice_Storm_Risk",
    "Precip_Extremes",
    "Hail_Impact",
    "Winter_Storm",
    "Winter_Intense_Precip",
    "Summer_Intense_Rain",
]
history_cols = [
    "Failures_7d",
    "Failures_30d",
    "Days_Since_Fail",
    "4G_EMA_3",
    "BaseTech_EMA_3",
    "Hist_Avg_Avail",
]

BUFFER_DAYS = 30


def load_and_prepare(df_main_with_weather):
    df = df_main_with_weather.copy()
    df["target_today"] = (df["Cell Avail Base Tech (%)"] < 99.8).astype(int)

    df["RECDATE"] = pd.to_datetime(df["RECDATE"])
    df = df.sort_values(["Master Site", "RECDATE"]).reset_index(drop=True)

    df, weather_next_cols = add_cyclical_features(df)
    df = add_weather_composite_features(df)
    df = add_alarm_features(df)

    split_date = df["RECDATE"].quantile(0.8)
    train_df = df[df["RECDATE"] <= split_date].copy()
    test_df = df[df["RECDATE"] > split_date].copy()

    train_df = create_features(train_df)
    train_df = train_df.loc[:, ~train_df.columns.duplicated()]

    test_buffer = train_df.groupby("Master Site").tail(BUFFER_DAYS)

    test_buffer = test_buffer.loc[:, ~test_buffer.columns.duplicated()]
    test_df = test_df.loc[:, ~test_df.columns.duplicated()]

    test_with_hist = (
        pd.concat([test_buffer, test_df], ignore_index=True)
        .sort_values(["Master Site", "RECDATE"])
        .reset_index(drop=True)
    )

    test_with_hist = create_features(test_with_hist)
    test_with_hist = test_with_hist.loc[:, ~test_with_hist.columns.duplicated()]

    test_df = test_with_hist[test_with_hist["RECDATE"] > split_date].copy()

    train_df["target_today"] = (train_df["Cell Avail Base Tech (%)"] < 99.8).astype(int)
    test_df["target_today"] = (test_df["Cell Avail Base Tech (%)"] < 99.8).astype(int)

    train_df["target_next_day"] = train_df.groupby("Master Site")["target_today"].shift(
        -1
    )
    test_df["target_next_day"] = test_df.groupby("Master Site")["target_today"].shift(
        -1
    )

    train_df = train_df.dropna(subset=["target_next_day"])
    test_df = test_df.dropna(subset=["target_next_day"])

    alarm_cols = [c for c in train_df.columns if c.startswith("Alarm_")]

    all_feature_names = (
        tech_cols
        + weather_cols
        + weather_next_cols
        + cyclical_cols
        + interaction_cols
        + history_cols
        + alarm_cols
    )

    feature_cols = [c for c in all_feature_names if c in train_df.columns]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target_next_day"].astype(int)

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target_next_day"].astype(int)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, train_df, test_df, feature_cols
