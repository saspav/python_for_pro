import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import optuna
from optuna.integration import CatBoostPruningCallback
from catboost import CatBoostClassifier, Pool

import warnings

warnings.filterwarnings("ignore")

from some_functions_clf import (SEED, make_train_valid, DataTransform, find_depth,
                                train_valid_model, make_submit)


def objective(trial):
    """Функция для оптимизации гиперпараметров CatBoost."""
    params_cv = {
        'random_seed': SEED,
        'verbose': False,
        # Параметры для оптимизации
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
        'depth': trial.suggest_int('depth', 4, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'colsample_bylevel': trial.suggest_categorical('colsample_bylevel',
                                                       [0.6, 0.7, 0.8, 0.9, 1.0]),
        'grow_policy': trial.suggest_categorical('grow_policy',
                                                 ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type',
                                                    ['Bayesian', 'Bernoulli', 'MVS', 'No'])
    }
    # Особые условия для bootstrap_type
    if params_cv['bootstrap_type'] == 'Bayesian':
        params_cv['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    if params_cv['bootstrap_type'] == 'Bernoulli':
        params_cv['subsample'] = trial.suggest_categorical('subsample',
                                                           [0.6, 0.7, 0.8, 0.9, 1.0])

    loss_function = 'Logloss'
    eval_metric = 'AUC'

    model = CatBoostClassifier(
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


# Настройки отображения данных в Pandas
pd.set_option('display.max_columns', 50)  # Отображать до 50 столбцов
pd.set_option('display.precision', 5)  # Отображение ДФ с 5-ю знаками после запятой

# Зафиксируем сиды
np.random.seed(SEED)
random.seed(SEED)

# Целевая переменная
target = 'Personality'

train, valid, test = make_train_valid()

n_neighbors = 4

dts = DataTransform(set_category=True, preprocessor=KNNImputer, n_neighbors=n_neighbors)

# Применяем трансформации
train = dts.fit_transform(train)
valid = dts.transform(valid)

# Колонки для моделей
model_columns = dts.all_features
cat_cols = dts.category_columns

# Обучение модели с параметрами по умолчанию
params = {'cat_features': cat_cols, 'verbose': False}

xb1, metrics_df7, _ = train_valid_model(CatBoostClassifier, 'Cat', params,
                                        train, valid, model_columns, target)

print(metrics_df7, '\n')
metrics_df = metrics_df7

# # Подберем оптимальную глубину дерева
# opt_depth_xb = find_depth(CatBoostClassifier, params, train, valid, model_columns, target)
# # Ранняя остановка: лучший Valid F1 (0.94063) был на глубине 4
# exit()

X_train, y_train = train[model_columns], train[target]
X_valid, y_valid = valid[model_columns], valid[target]

# Подготовка данных в Pool
pool_train = Pool(data=X_train, label=y_train, cat_features=cat_cols)
pool_valid = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)
# Отключение инфо выводов
optuna.logging.set_verbosity(optuna.logging.WARNING)

# # Создаем study с направлением maximize
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=200, show_progress_bar=True)
#
# # Вывод результатов
# print(f"Лучший AUC: {study.best_value:.4f}")  # Форматирование для AUC
# print("Лучшие параметры:", study.best_params)
# exit()

cb_best_grid = {'learning_rate': 0.011960936787877805, 'depth': 5,
                'l2_leaf_reg': 8.09698650761031, 'min_data_in_leaf': 12,
                'colsample_bylevel': 1.0, 'grow_policy': 'SymmetricTree',
                'bootstrap_type': 'MVS',
                'cat_features': cat_cols,
                'verbose': False}

cb2, metrics_df8, optimal_threshold = train_valid_model(CatBoostClassifier, 8, cb_best_grid,
                                                        train, valid, model_columns, target)

metrics_df = pd.concat([metrics_df, metrics_df8.drop(columns=['Metric'])], axis=1)

print(f'n_neighbors = {n_neighbors}')
print(metrics_df)

# Трансформируем признаки тестовой выборки
test = dts.transform(test)

optimal_threshold = 0.5

# Вызываем функцию для формирования сабмита: передаем обученную модель
_ = make_submit(cb2, test[model_columns], optimal_threshold, dts.reverse_mapping)

# 'depth': 5 Оптимальный порог: 0.2308
# n_neighbors = 4
#       Metric  TrainCat  ValidCat  DiffCat,%   Train8   Valid8  Diff8,%
# 0   accuracy   0.97368   0.96896      -0.48  0.96990  0.96950    -0.04
# 1  precision   0.94902   0.94737      -0.17  0.94361  0.95032     0.71
# 2     recall   0.95000   0.93264      -1.83  0.94067  0.93161    -0.96
# 3         f1   0.94951   0.93995      -1.01  0.94214  0.94087    -0.13
# 4    roc_auc   0.99158   0.96907      -2.27  0.97926  0.96995    -0.95

# Оптимальный порог: 0.1972
# n_neighbors = 4
#       Metric  TrainCat  ValidCat  DiffCat,%   Train8   Valid8  Diff8,%
# 0   accuracy   0.97368   0.96896      -0.48  0.96957  0.96950    -0.01
# 1  precision   0.94902   0.94737      -0.17  0.94192  0.94937     0.79
# 2     recall   0.95000   0.93264      -1.83  0.94119  0.93264    -0.91
# 3         f1   0.94951   0.93995      -1.01  0.94156  0.94093    -0.07
# 4    roc_auc   0.99158   0.96907      -2.27  0.97967  0.96924    -1.06