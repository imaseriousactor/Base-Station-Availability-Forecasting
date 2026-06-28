import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "main_data_with_weather_anonymized.parquet")
CSV_PATH = os.path.join(BASE_DIR, "main_data_with_weather_anonymized.csv")
DATA_PATH = PARQUET_PATH if os.path.exists(PARQUET_PATH) else CSV_PATH

KPI_COLUMNS = {
    "2G": "Cell Avail 2G (%)",
    "3G": "Cell Avail 3G (%)",
    "4G": "Cell Avail 4G (%)",
    "Base Tech": "Cell Avail Base Tech (%)",
}
LAT_COL = "Y"
LON_COL = "X"
SITE_ID_COL = "Master Site"
MASTER_SITE_COL = "Master Site"

WEATHER_COLUMNS = [
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

CORR_CANDIDATES = (
    list(KPI_COLUMNS.values())
    + WEATHER_COLUMNS
    + [
        "Высота подвеса, м",
        "Alarm Count",
        "Total Fault Time",
    ]
)

CORR_DEFAULT = [
    "Cell Avail Base Tech (%)",
    "TempDelta",
    "MaxWind",
    "Alarm Count",
    "Total Fault Time",
]

MONTH_NAMES_RU = [
    "Янв",
    "Фев",
    "Мар",
    "Апр",
    "Май",
    "Июн",
    "Июл",
    "Авг",
    "Сен",
    "Окт",
    "Ноя",
    "Дек",
]
WEEKDAY_NAMES_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]

LAG_DAYS = [0, 1, 2, 3, 5, 7]


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".gz"):
        df = pd.read_csv(
            path, sep=";", encoding="utf-8-sig", compression="gzip", low_memory=False
        )
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)

    for col in CORR_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["RECDATE", "Дата запуска", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "RECDATE" in df.columns:
        df["Month"] = df["RECDATE"].dt.month
        df["Weekday"] = df["RECDATE"].dt.weekday

    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    return df


df = load_data(DATA_PATH)

CORR_CANDIDATES = [c for c in CORR_CANDIDATES if c in df.columns]
CORR_DEFAULT = [c for c in CORR_DEFAULT if c in CORR_CANDIDATES]
WEATHER_COLUMNS = [c for c in WEATHER_COLUMNS if c in df.columns]

date_min = df["RECDATE"].min()
date_max = df["RECDATE"].max()

daily_agg_dict = {col: "mean" for col in KPI_COLUMNS.values()}
daily_agg_dict.update({col: "mean" for col in WEATHER_COLUMNS})
daily_means = (
    df.groupby("RECDATE", as_index=False).agg(daily_agg_dict).sort_values("RECDATE")
)

app = Dash(__name__)

CARD_STYLE = {
    "background": "#fff",
    "borderRadius": "12px",
    "padding": "20px 24px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
    "marginBottom": "24px",
}


def kpi_dropdown(id_, default="Base Tech"):
    return dcc.Dropdown(
        id=id_,
        options=[{"label": k, "value": k} for k in KPI_COLUMNS],
        value=default,
        clearable=False,
        style={"width": "200px", "marginBottom": "12px"},
    )


app.layout = html.Div(
    style={
        "backgroundColor": "#f4f5f7",
        "padding": "24px 32px",
        "fontFamily": "Segoe UI, Roboto, Arial, sans-serif",
    },
    children=[
        html.H2("Дашборд доступности сети"),
        html.Div(
            style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={**CARD_STYLE, "flex": "1", "minWidth": "380px"},
                    children=[
                        html.H4("Доля записей по Base Tech Availability"),
                        html.Div(
                            id="pie-threshold-label",
                            style={"fontWeight": "600", "marginBottom": "8px"},
                        ),
                        dcc.Slider(
                            id="pie-threshold",
                            min=0,
                            max=100,
                            step=0.5,
                            value=95,
                            marks={i: str(i) for i in range(0, 101, 10)},
                            tooltip={"placement": "top", "always_visible": False},
                        ),
                        html.Div(style={"height": "16px"}),
                        dcc.Graph(id="pie-chart"),
                    ],
                ),
                html.Div(
                    style={**CARD_STYLE, "flex": "1", "minWidth": "380px"},
                    children=[
                        html.H4("Распределение доступности по KPI"),
                        kpi_dropdown("hist-kpi"),
                        html.Div(
                            id="hist-ceiling-label",
                            style={"fontWeight": "600", "marginBottom": "8px"},
                        ),
                        dcc.Slider(
                            id="hist-ceiling",
                            min=1,
                            max=100,
                            step=0.5,
                            value=100,
                            marks={i: str(i) for i in range(0, 101, 10)},
                            tooltip={"placement": "top", "always_visible": False},
                        ),
                        html.Div(style={"height": "16px"}),
                        dcc.Graph(id="hist-chart"),
                    ],
                ),
            ],
        ),
        html.Div(
            style=CARD_STYLE,
            children=[
                html.H4("Динамика доступности во времени"),
                kpi_dropdown("ts-kpi"),
                dcc.Graph(id="ts-chart"),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={**CARD_STYLE, "flex": "1", "minWidth": "380px"},
                    children=[
                        html.H4("Топ-10 худших вышек"),
                        kpi_dropdown("top-kpi"),
                        dcc.Graph(id="top10-chart"),
                    ],
                ),
                html.Div(
                    style={**CARD_STYLE, "flex": "1", "minWidth": "380px"},
                    children=[
                        html.H4("Сезонность: месяц × день недели"),
                        kpi_dropdown("season-kpi"),
                        dcc.Graph(id="season-chart"),
                    ],
                ),
            ],
        ),
        html.Div(
            style=CARD_STYLE,
            children=[
                html.H4("Влияние погоды на доступность (со сдвигом по дням)"),
                html.Div(
                    [
                        html.Div(
                            "Погодный признак:",
                            style={"fontWeight": "600", "marginBottom": "6px"},
                        ),
                        dcc.Dropdown(
                            id="lag-weather",
                            options=[{"label": w, "value": w} for w in WEATHER_COLUMNS],
                            value=WEATHER_COLUMNS[0] if WEATHER_COLUMNS else None,
                            clearable=False,
                            style={"width": "240px", "marginBottom": "12px"},
                        ),
                    ],
                ),
                kpi_dropdown("lag-kpi"),
                dcc.Graph(id="lag-chart"),
            ],
        ),
        html.Div(
            style=CARD_STYLE,
            children=[
                html.H4("Матрица корреляций"),
                html.Div(
                    "Выберите признаки (только числовые):",
                    style={"marginBottom": "6px", "fontWeight": "600"},
                ),
                dcc.Dropdown(
                    id="corr-features",
                    options=[{"label": c, "value": c} for c in CORR_CANDIDATES],
                    value=CORR_DEFAULT,
                    multi=True,
                ),
                html.Div(style={"height": "12px"}),
                dcc.Graph(id="corr-chart"),
            ],
        ),
        html.Div(
            style=CARD_STYLE,
            children=[
                html.H4("Карта сайтов"),
                html.Div(
                    [
                        html.Div(
                            "Дата:", style={"fontWeight": "600", "marginBottom": "6px"}
                        ),
                        dcc.DatePickerSingle(
                            id="map-date",
                            min_date_allowed=date_min,
                            max_date_allowed=date_max,
                            date=date_max,
                            display_format="YYYY-MM-DD",
                        ),
                    ],
                    style={"marginBottom": "12px"},
                ),
                dcc.Graph(id="map-chart", style={"height": "600px"}),
            ],
        ),
    ],
)


@app.callback(
    Output("pie-threshold-label", "children"), Input("pie-threshold", "value")
)
def show_pie_threshold(value):
    return f"Порог X: {value}%"


@app.callback(Output("hist-ceiling-label", "children"), Input("hist-ceiling", "value"))
def show_hist_ceiling(value):
    return f"Потолок шкалы: {value}%"


@app.callback(
    Output("pie-chart", "figure"),
    Input("pie-threshold", "value"),
)
def update_pie(threshold):
    col = KPI_COLUMNS["Base Tech"]
    valid = df[col].dropna()
    above = int((valid > threshold).sum())
    below_eq = int((valid <= threshold).sum())
    total = above + below_eq

    fig = px.pie(
        names=[f"> {threshold}%", f"≤ {threshold}%"],
        values=[above, below_eq],
        hole=0.45,
    )
    fig.update_traces(textinfo="label+percent+value")
    fig.update_layout(
        annotations=[
            dict(text=f"всего: {total}", x=0.5, y=0.5, showarrow=False, font_size=14)
        ],
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


@app.callback(
    Output("hist-chart", "figure"),
    Input("hist-kpi", "value"),
    Input("hist-ceiling", "value"),
)
def update_hist(kpi_label, ceiling):
    col = KPI_COLUMNS[kpi_label]
    valid = df[col].dropna()
    visible = valid[valid <= ceiling]

    fig = px.histogram(
        visible, nbins=40, labels={"value": f"{kpi_label} Availability, %"}
    )
    fig.update_layout(
        xaxis_range=[0, ceiling],
        bargap=0.05,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig.add_annotation(
        text=f"показано {len(visible)} из {len(valid)}",
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.98,
        showarrow=False,
        font=dict(size=11, color="#777"),
    )
    return fig


@app.callback(
    Output("ts-chart", "figure"),
    Input("ts-kpi", "value"),
)
def update_ts(kpi_label):
    col = KPI_COLUMNS[kpi_label]

    fig = px.line(
        daily_means,
        x="RECDATE",
        y=col,
        labels={"RECDATE": "Дата", col: f"{kpi_label} Availability, %"},
    )
    fig.update_traces(line=dict(width=1.8))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis_range=[0, 105],
    )
    return fig


@app.callback(
    Output("top10-chart", "figure"),
    Input("top-kpi", "value"),
)
def update_top10(kpi_label):
    col = KPI_COLUMNS[kpi_label]

    by_tower = (
        df.groupby(MASTER_SITE_COL)[col]
        .mean()
        .dropna()
        .sort_values()
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        by_tower,
        x=col,
        y=MASTER_SITE_COL,
        orientation="h",
        labels={col: f"Средняя {kpi_label} Availability, %", MASTER_SITE_COL: "Вышка"},
        color=col,
        color_continuous_scale="Reds_r",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    return fig


@app.callback(
    Output("season-chart", "figure"),
    Input("season-kpi", "value"),
)
def update_season(kpi_label):
    col = KPI_COLUMNS[kpi_label]

    pivot = (
        df.groupby(["Month", "Weekday"])[col]
        .mean()
        .unstack("Weekday")
        .reindex(index=range(1, 13), columns=range(7))
    )

    fig = px.imshow(
        pivot,
        labels=dict(color=f"{kpi_label} Availability, %"),
        x=WEEKDAY_NAMES_RU,
        y=MONTH_NAMES_RU,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=".1f",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


@app.callback(
    Output("lag-chart", "figure"),
    Input("lag-weather", "value"),
    Input("lag-kpi", "value"),
)
def update_lag(weather_col, kpi_label):
    if not weather_col:
        return go.Figure().update_layout(title="Нет погодных колонок")

    avail_col = KPI_COLUMNS[kpi_label]

    correlations = []
    for lag in LAG_DAYS:
        shifted_weather = daily_means[weather_col].shift(lag)
        pair = pd.DataFrame(
            {"avail": daily_means[avail_col], "weather": shifted_weather}
        ).dropna()
        corr = pair["avail"].corr(pair["weather"]) if len(pair) > 2 else np.nan
        correlations.append(corr)

    fig = px.line(
        x=LAG_DAYS,
        y=correlations,
        markers=True,
        labels={"x": "Сдвиг погоды, дней назад", "y": "Коэффициент корреляции"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[-1, 1])
    return fig


@app.callback(
    Output("corr-chart", "figure"),
    Input("corr-features", "value"),
)
def update_corr(features):
    if not features or len(features) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Выберите минимум 2 признака",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    corr = df[features].corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


def sites_table_for_date(data: pd.DataFrame, selected_date) -> pd.DataFrame:
    day_data = data[data["RECDATE"].dt.date == pd.to_datetime(selected_date).date()]

    agg_dict = {LAT_COL: "first", LON_COL: "first"}
    for col in KPI_COLUMNS.values():
        agg_dict[col] = "mean"

    return (
        day_data.dropna(subset=[LAT_COL, LON_COL])
        .groupby(SITE_ID_COL, as_index=False)
        .agg(agg_dict)
    )


@app.callback(
    Output("map-chart", "figure"),
    Input("map-date", "date"),
)
def update_map(selected_date):
    sites = sites_table_for_date(df, selected_date)
    col = KPI_COLUMNS["Base Tech"]

    if sites.empty:
        return go.Figure().update_layout(title="Нет данных за эту дату")

    fig = px.scatter(
        sites,
        x=LON_COL,
        y=LAT_COL,
        color=col,
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        hover_name=SITE_ID_COL,
    )
    fig.update_traces(
        marker=dict(size=12, opacity=0.85, line=dict(width=0.5, color="white")),
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        plot_bgcolor="#fafafa",
    )
    return fig


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
