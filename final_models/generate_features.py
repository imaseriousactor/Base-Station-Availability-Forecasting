import pandas as pd
import numpy as np
import re


def add_cyclical_features(df):
    df = df.copy()
    df['RECDATE'] = pd.to_datetime(df['RECDATE'])

    # День недели (0=Пн, 6=Вс)
    df['day_of_week'] = df['RECDATE'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # День месяца (1-31)
    df['day_of_month'] = df['RECDATE'].dt.day
    df['day_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

    # Месяц (1-12)
    df['month'] = df['RECDATE'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Квартал (1-4)
    df['quarter'] = df['RECDATE'].dt.quarter
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    # Сезон
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })

    # Циклическое кодирование сезона
    season_num_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    df['season_num'] = df['season'].map(season_num_map)

    df['season_sin'] = np.sin(2 * np.pi * df['season_num'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season_num'] / 4)

    df.drop(columns=['season_num'], inplace=True)

    weather_cols = ['TempDelta', 'MaxWind', 'MaxInt', 'Rain', 'Snow', 'Sprinkling', 'Hail', 'U', 'Po']
    weather_next_cols = []
    for col in weather_cols:
        if col in df.columns:
            col_next = f'{col}_next'
            df[col_next] = df.groupby('Master Site')[col].shift(-1)
            weather_next_cols.append(col_next)

    return df, weather_next_cols


def add_weather_composite_features(df):
    df = df.copy()

    # 1. Штормовой индекс
    df['Storm_Intensity'] = df['MaxWind'] * df['MaxInt']

    # 2. Риск обледенения
    df['Ice_Storm_Risk'] = df['TempDelta'] * df['MaxInt'] * (df['Snow'] + df['Rain'] + df['Sprinkling'])

    # 3. Экстремальные осадки
    df['Precip_Extremes'] = df['MaxInt'] * (df['Rain'] + df['Snow'] + df['Hail'])

    # 4. Град + интенсивность
    df['Hail_Impact'] = df['Hail'] * df['MaxInt']

    # 5. Зимний шторм
    df['Winter_Storm'] = df['Snow'] * df['MaxInt'] * df['MaxWind']

    # 7. Сезон × Интенсивность
    df['Winter_Intense_Precip'] = df['MaxInt'] * (df['season'] == 'winter').astype(int)
    df['Summer_Intense_Rain'] = df['MaxInt'] * df['Rain'] * (df['season'] == 'summer').astype(int)

    return df


def create_features(data):
    data = data.copy()

    data['is_failure'] = (data['Cell Avail Base Tech (%)'] < 99.8).astype(int)

    data['Failures_7d'] = data.groupby('Master Site')['is_failure'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    data['Failures_30d'] = data.groupby('Master Site')['is_failure'].transform(lambda x: x.rolling(30, min_periods=1).sum())

    data['Last_Fail_Date'] = data['RECDATE'].where(data['is_failure'] == 1, pd.NaT)
    data['Last_Fail_Date'] = data.groupby('Master Site')['Last_Fail_Date'].ffill()
    data['Days_Since_Fail'] = (data['RECDATE'] - data['Last_Fail_Date']).dt.days
    data['Days_Since_Fail'] = data['Days_Since_Fail'].fillna(999).astype(int)

    data['4G_EMA_3'] = data.groupby('Master Site')['Cell Avail 4G (%)'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    data['BaseTech_EMA_3'] = data.groupby('Master Site')['Cell Avail Base Tech (%)'].transform(lambda x: x.ewm(span=3, adjust=False).mean())

    data['Hist_Avg_Avail'] = data.groupby('Master Site')['Cell Avail Base Tech (%)'].transform(lambda x: x.expanding().mean())

    data.drop(columns=['is_failure', 'Last_Fail_Date'], inplace=True, errors='ignore')

    return data


def add_alarm_features(df):
    df = df.copy()

    df['Alarm_Clean'] = df['Alarm Descriptions'].fillna('').str.strip()
    df['Alarm_Clean'] = df['Alarm_Clean'].replace('', 'No_alarms')
    df['Alarm_Clean'] = df['Alarm_Clean'].str.replace(r'\s*[|;,]\s*', '|', regex=True)

    alarm_dummies = df['Alarm_Clean'].str.get_dummies(sep='|')

    clean_names = []
    for col in alarm_dummies.columns:
        safe_name = re.sub(r'[^A-Za-z0-9_]', '_', str(col))
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        clean_names.append(f'Alarm_{safe_name}')

    alarm_dummies.columns = clean_names

    if 'Alarm_No_alarms' in alarm_dummies.columns:
        alarm_dummies = alarm_dummies.drop(columns=['Alarm_No_alarms'])

    df = pd.concat([df, alarm_dummies], axis=1)
    df.drop(columns=['Alarm_Clean'], inplace=True)

    return df
