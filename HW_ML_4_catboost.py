import numpy as np
import pandas as pd
import random

import optuna
from optuna.integration import CatBoostPruningCallback
from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split

from some_functions import (SEED,
                            memory_compression,
                            find_depth,
                            train_valid_model,
                            find_best_model,
                            )

__import__("warnings").filterwarnings('ignore')


def objective(trial):
    """Функция для оптимизации гиперпараметров CatBoost."""
    params_cv = {
        'random_seed': SEED,
        'verbose': False,
        # Параметры для оптимизации
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 6, 13),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100, step=5),
        'colsample_bylevel': trial.suggest_categorical('colsample_bylevel',
                                                       [0.7, 0.8, 0.9, 1.0]),
        'grow_policy': trial.suggest_categorical('grow_policy',
                                                 ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type',
                                                    ['Bayesian', 'Bernoulli', 'MVS', 'No'])
    }
    # Особые условия для bootstrap_type
    if params_cv['bootstrap_type'] == 'Bayesian':
        params_cv['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    if params_cv['bootstrap_type'] == 'Bernoulli':
        params_cv['subsample'] = trial.suggest_categorical('subsample', [0.7, 0.8, 0.9, 1.0])

    loss_function = 'RMSE'  # Используем RMSE для регрессии
    eval_metric = 'RMSE'  # Метрическая оценка RMSE

    model = CatBoostRegressor(
        loss_function=loss_function,
        eval_metric=eval_metric,
        cat_features=cat_cols,
        **params_cv
    )
    pruning_callback = CatBoostPruningCallback(trial, eval_metric)
    model.fit(pool_train, eval_set=pool_valid,
              verbose=0,
              early_stopping_rounds=50,
              callbacks=[pruning_callback],
              )
    pruning_callback.check_pruned()

    return model.best_score_['validation'][eval_metric]


# Зафиксируем сиды
np.random.seed(SEED)
random.seed(SEED)

df = pd.read_csv(r'G:\python-txt\_курс\train_сalories.csv')
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

params = {'cat_features': cat_cols, 'verbose': False}

cb1, metrics_df7, _ = train_valid_model(CatBoostRegressor, 7, params,
                                        train, valid, model_columns, target)

print(metrics_df7)
metrics_df = metrics_df7

# Подберем оптимальную глубину дерева
opt_depth_cb = find_depth(CatBoostRegressor, params, train, valid, model_columns, target)
# Ранняя остановка: лучший Valid RMSE (3.54) был на глубине 10

X_train, y_train = train[model_columns], train[target]
X_valid, y_valid = valid[model_columns], valid[target]

# Подготовка данных в Pool
pool_train = Pool(data=X_train, label=y_train, cat_features=cat_cols)
pool_valid = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)

# Создаем study и запускаем оптимизацию
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)  # 100 итераций

# Вывод результатов
print(f"Лучший RMSE: {study.best_value:.2f}")
print("Лучшие параметры:", study.best_params)

cb_best_grid = {'learning_rate': 0.1559073096894334, 'depth': 11,
                'l2_leaf_reg': 9.204228730749122, 'min_data_in_leaf': 95,
                'colsample_bylevel': 0.9, 'grow_policy': 'SymmetricTree',
                'bootstrap_type': 'Bernoulli', 'subsample': 1.0,
                'cat_features': cat_cols, 'verbose': False}

cb2, metrics_df8, _ = train_valid_model(CatBoostRegressor, 8, cb_best_grid,
                                        train, valid, model_columns, target)

metrics_df = pd.concat([metrics_df, metrics_df8.drop(columns=['Metric'])], axis=1)
print(metrics_df8)

cb3, metrics_df9, _ = train_valid_model(CatBoostRegressor, 9, cb_best_grid,
                                        train, valid, model_columns, target, target_log=True)

metrics_df = pd.concat([metrics_df, metrics_df9.drop(columns=['Metric'])], axis=1)
print(metrics_df9)

# Поиск лучшей модели по метрикам
find_best_model(metrics_df)
