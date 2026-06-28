# Прогноз аварий на базовых станциях

Проект по предсказанию доступности базовых станций Т2 в рамках курсовой работы 2 курса ПАДИИ ВШЭ СПб.

Выполнили: Овсянникова Мария, Тищенко Софья  
Научный руководитель: Михайлов Михаил
## Структура

- `notebooks/`: ноутбуки с разведочным анализом и обучением моделей (код + графики)
- `baseline/`: простые модели (LogReg, RF, NaiveBayes)
- `final_models/`: бустинги (XGBoost, LightGBM, CatBoost) и нейросети (LSTM, GRU)
- `dashboard/`: деплой дашборда на HuggingFace Pages

## Данные

Все скрипты ожидают файлы в текущей рабочей директории:

- `main_data_with_weather.xlsx`
- `Cell_Avail.xlsx`
- `северо-запад_Rollout_RF-Site_vsf_20260224.xlsx`
- `NOC_TT_Report_23.xlsx`, `NOC_TT_Report_24.xlsx`, `NOC_TT_Report_25.xlsx`
Данные в `dashboard/` зашифрованы, изначально все файлы были предоставлены сотрудниками T2.
## eda/

| Ноутбук                      | Содержание                                                             |
| ---------------------------- | ---------------------------------------------------------------------- |
| `00_init.ipynb`              | импорты, загрузка таблиц                                               |
| `01_data_merge.ipynb`        | склейка Cell_Avail + координаты станций + аварии                       |
| `02_eda_cell_avail.ipynb`    | EDA по таблице доступности (Cell_Avail)                                |
| `03_eda_weather.ipynb`       | EDA по таблице с погодой, корреляции, корреляции со сдвигом по времени |
| `04_clustering.ipynb`        | кластеризация станций (K-means, DBSCAN)                                |
| `05_cross_validation.ipynb`  | утилита `run_time_series_cv` (TimeSeriesSplit)                         |
| `06_anonymization.ipynb`     | анонимизация координат и идентификаторов станций                       |
| `07_feature_selection.ipynb` | отбор признаков (Lasso, RF importance, mutual information)             |
| `08_presentation.ipynb`      | итоговые графики для презентации                                       |

Ноутбуки открываются с помощью `jupyter notebook` / `jupyter lab` / Colab. Графики сохранены в выводах ячеек, перезапуск не обязателен - можно просто пролистать.
## baseline/

Каждая модель — отдельный скрипт, сама модель определена в `models/`:

```
baseline/
├── data_prep.py  # переиспользует пайплайн признаков из final_models
├── models/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── naive_bayes.py
├── run_logistic_regression.py
├── run_random_forest.py
└── run_naive_bayes.py
```

Запуск (из папки `baseline/`):

```bash
python run_logistic_regression.py
python run_random_forest.py
python run_naive_bayes.py
```

`data_prep.py` в `baseline/` подгружает тот же `load_and_prepare` из `final_models/data_prep.py`.

Зависимости: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## final_models/

```
final_models/
├── generate_features.py # все функции генерации признаков, включая create_features
├── data_prep.py # train/test split по времени
├── cv_utils.py # run_time_series_cv
├── models/
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── catboost_model.py
│   ├── lstm.py
│   └── gru.py
├── run_xgboost.py
├── run_lightgbm.py
├── run_catboost.py
├── run_lstm.py
└── run_gru.py
```

Запуск (из папки `final_models/`):

```bash
python run_xgboost.py
python run_lightgbm.py
python run_catboost.py
python run_lstm.py
python run_gru.py
```

Каждый `run_*.py` для бустингов:
1. готовит данные через `data_prep.load_and_prepare`;
2. подбирает гиперпараметры через Optuna (`cv_utils.run_time_series_cv` внутри objective);
3. обучает финальную модель с фиксированными параметрами;
4. строит Precision-Recall кривую и важность признаков.
`run_xgboost.py` дополнительно включает модель без исторических признаков
и сравнение пойманных/пропущенных аварий.

Зависимости: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `optuna`,
`xgboost`, `lightgbm`, `catboost`, `torch`.

