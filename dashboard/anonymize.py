import pandas as pd
import numpy as np
import hashlib

INPUT_FILE = "main_data_with_weather.csv"
OUTPUT_FILE = "main_data_with_weather_anonymized.csv"

ID_COL = "Master Site"
DUP_ID_COL = "Номер сайта"
ADDR_COL = "Адрес"
LAT_COL = "Широта WGS84"
LON_COL = "Долгота WGS84"

SALT = "geo_project_2024"

LAT_BOUNDS = (58.0, 61.0)
LON_BOUNDS = (27.5, 32.5)

df = pd.read_csv(INPUT_FILE, sep=",", encoding="utf-8-sig", low_memory=False)

df.columns = df.columns.str.strip()

cols_to_drop = [c for c in [DUP_ID_COL, ADDR_COL] if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print("Удалены колонки:", cols_to_drop)


def hash_station(code):
    if pd.isna(code):
        return np.nan
    return hashlib.sha256(f"{SALT}_{str(code).strip()}".encode()).hexdigest()[:12]

df[ID_COL] = df[ID_COL].apply(hash_station)
print("Коды станций захэшированы")

df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")

not_na = df[LAT_COL].notna() & df[LON_COL].notna()
not_zero = ~((df[LAT_COL] == 0) & (df[LON_COL] == 0))
in_bounds = df[LAT_COL].between(*LAT_BOUNDS) & df[LON_COL].between(*LON_BOUNDS)
valid_mask = not_na & not_zero & in_bounds

n_outliers = (not_na & not_zero & ~in_bounds).sum()
print(f"Валидных точек: {valid_mask.sum()} / {len(df)}")
if n_outliers:
    print(f"⚠ Найдено {n_outliers} выбросов вне региона — обнулены (NaN)")

df.loc[~valid_mask, [LAT_COL, LON_COL]] = np.nan

# --- переводим реальные координаты в относительные XY (без геопривязки) ---
if valid_mask.any():
    lat = df.loc[valid_mask, LAT_COL]
    lon = df.loc[valid_mask, LON_COL]

    mean_lat_rad = np.radians(lat.mean())
    x = (lon - lon.mean()) * np.cos(mean_lat_rad)
    y = (lat - lat.mean())

    x_norm = 100 * (x - x.min()) / (x.max() - x.min())
    y_norm = 100 * (y - y.min()) / (y.max() - y.min())

    df.loc[valid_mask, "X"] = x_norm
    df.loc[valid_mask, "Y"] = y_norm
    print("Добавлены относительные координаты X, Y (0-100)")

# убираем исходные реальные координаты
df.drop(columns=[LAT_COL, LON_COL], inplace=True)
print(f"Колонки {LAT_COL}, {LON_COL} удалены — оставлены только относительные X, Y")

parquet_output = OUTPUT_FILE.replace('.csv', '.parquet')
df.to_parquet(parquet_output, index=False, compression='snappy')
print(f"\n✅ Файл сохранён: {parquet_output}")
print("Колонки итогового файла:", df.columns.tolist())