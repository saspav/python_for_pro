import numpy as np
import pandas as pd
import random

import optuna
import xgboost as xgb

from sklearn.model_selection import train_test_split
from some_functions import SEED, memory_compression, find_depth, train_valid_model


def objective(trial):
    """Функция для оптимизации гиперпараметров."""
    params_cv = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': SEED,
        'verbosity': 0,
        # Параметры для оптимизации
        'eta': trial.suggest_float('eta', 0.01, 0.3),  # learning_rate
        'max_depth': trial.suggest_int('max_depth', 6, 9),
        'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 0.9, 1.0]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                      [0.7, 0.8, 0.9, 1.0]),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
    }
    # Кросс-валидация с ранней остановкой
    results_cv = xgb.cv(
        params=params_cv,
        dtrain=dtrain,
        num_boost_round=1000,  # Большое число (ранняя остановка остановит раньше)
        stratified=train['Sex'],
        nfold=3,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    # Возвращаем лучший RMSE
    return results_cv['test-rmse-mean'].min()


# Зафиксируем сиды
np.random.seed(SEED)
random.seed(SEED)

df = pd.read_csv('train_сalories.csv')
# Колонка "id" не несет смысла - это индекс
df.set_index("id", inplace=True)

target = 'Calories'
# Выбираем категориальные колонки (включая строки и категории)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# Выбираем числовые колонки
num_cols = df.select_dtypes(include=['number']).columns.tolist()
# Удаляем дубликаты
df.drop_duplicates(inplace=True)
print("Количество дубликатов:", df.duplicated().sum())

# строим распределение каждого из признаков
numeric_cols = num_cols.copy()
numeric_cols.remove(target)

df['Calories_log'] = np.log1p(df['Calories'])

df = memory_compression(df)

# Не будем пока выделять целевой признак, т.к. далее будет работа с выбросами
train, valid = train_test_split(df, test_size=0.2, stratify=df['Sex'], random_state=SEED)

model_columns = numeric_cols + cat_cols

params = {'enable_categorical': True, 'n_jobs': -1}

xb1, metrics_df3, _ = train_valid_model(xgb.XGBRegressor, 3, params,
                                        train, valid, model_columns, target)

print(metrics_df3)
metrics_df = metrics_df3

# Подберем оптимальную глубину дерева
opt_depth_xb = find_depth(xgb.XGBRegressor, params, train, valid, model_columns, target)
# Ранняя остановка: лучший Valid RMSE (3.71) был на глубине 8

X_train, y_train = train[model_columns], train[target]
X_valid, y_valid = valid[model_columns], valid[target]

# Подготовка данных в DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

# Создаем study и запускаем оптимизацию
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)  # 50 итераций

# Вывод результатов
print(f"Лучший RMSE: {study.best_value:.2f}")
print("Лучшие параметры:", study.best_params)

xb_best_grid = {'eta': 0.028886882560958116, 'max_depth': 7, 'subsample': 0.8,
                'colsample_bytree': 0.9, 'gamma': 0.2110692155097766,
                'enable_categorical': True, 'n_jobs': -1}

xb2, metrics_df4, _ = train_valid_model(xgb.XGBRegressor, 4, xb_best_grid,
                                        train, valid, model_columns, target)

metrics_df = pd.concat([metrics_df, metrics_df4.drop(columns=['Metric'])], axis=1)
print(metrics_df4)
