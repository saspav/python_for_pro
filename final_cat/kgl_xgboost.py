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
import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")

from some_functions_clf import (SEED, make_train_valid, DataTransform, find_depth,
                                train_valid_model, make_submit)


def objective(trial):
    """Функция для оптимизации гиперпараметров."""
    params_cv = {
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'hist']),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'objective': 'binary:logistic',  # Изменено на бинарную классификацию
        'eval_metric': 'auc',  # Метрика - AUC
        'seed': SEED,
        'verbosity': 0,
        # Параметры для оптимизации
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 4, 7),
        'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 0.9, 1.0]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                      [0.7, 0.8, 0.9, 1.0]),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
    }
    if params_cv["booster"] == "dart":
        params_cv["rate_drop"] = trial.suggest_float("rate_drop", 0.0, 0.3)
        params_cv["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.3)

    # # Иногда добавляем scale_pos_weight, иногда нет
    # if trial.suggest_categorical('use_scale_pos_weight', [True, False]):
    #     params_cv['scale_pos_weight'] = default_spw

    # Кросс-валидация с ранней остановкой
    results_cv = xgb.cv(
        params=params_cv,
        dtrain=dtrain,
        num_boost_round=1000,
        stratified=True,  # Исправлено: стратификация по меткам класса
        nfold=3,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    # Возвращаем лучший AUC (максимизируем)
    return results_cv['test-auc-mean'].max()  # Берем максимальное значение AUC


# Настройки отображения данных в Pandas
pd.set_option('display.max_columns', 50)  # Отображать до 50 столбцов
pd.set_option('display.precision', 5)  # Отображение ДФ с 5-ю знаками после запятой

# Зафиксируем сиды
np.random.seed(SEED)
random.seed(SEED)

# Целевая переменная
target = 'Personality'

train, valid, test = make_train_valid()

dts = DataTransform(preprocessor=KNNImputer, n_neighbors=5)

# Применяем трансформации
train = dts.fit_transform(train)
valid = dts.transform(valid)

# Колонки для моделей
model_columns = dts.all_features

# Обучение модели с параметрами по умолчанию
params = {'n_jobs': -1}

xb1, metrics_df3, _ = train_valid_model(xgb.XGBClassifier, 'XGB', params,
                                        train, valid, model_columns, target)

print(metrics_df3)
metrics_df = metrics_df3

# # Подберем оптимальную глубину дерева
# opt_depth_xb = find_depth(xgb.XGBClassifier, params, train, valid, model_columns, target)
# # Ранняя остановка: лучший Valid F1 (0.94136) был на глубине 3
# exit()

X_train, y_train = train[model_columns], train[target]
X_valid, y_valid = valid[model_columns], valid[target]

# Подготовка данных в DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

# Отключение инфо выводов
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Считаем scale_pos_weight один раз (вне функции)
neg, pos = np.bincount(y_train)
default_spw = neg / pos

# # Создаем study с направлением maximize
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100, show_progress_bar=True)
#
# # Вывод результатов
# print(f"Лучший AUC: {study.best_value:.4f}")  # Форматирование для AUC
# print("Лучшие параметры:", study.best_params)

xb_best_grid = {'booster': 'dart', 'tree_method': 'exact', 'grow_policy': 'depthwise',
                'eta': 0.1372379059873951, 'max_depth': 4, 'subsample': 0.8,
                'colsample_bytree': 0.7, 'gamma': 0.001034181684304851,
                'rate_drop': 0.2569127887510873, 'skip_drop': 0.15933058764935298,
                'n_jobs': -1,
                }

xb2, metrics_df4, optimal_threshold = train_valid_model(xgb.XGBClassifier, '2', xb_best_grid,
                                                        train, valid, model_columns, target)

metrics_df = pd.concat([metrics_df, metrics_df4.drop(columns=['Metric'])], axis=1)
print(metrics_df)

# Трансформируем признаки тестовой выборки
test = dts.transform(test)

# Вызываем функцию для формирования сабмита: передаем обученную модель
_ = make_submit(xb2, test[model_columns], optimal_threshold, dts.reverse_mapping)

# Оптимальный порог: 0.1779
#       Metric  TrainXGB  ValidXGB  DiffXGB,%   Train2   Valid2  Diff2,%
# 0   accuracy   0.97415   0.96761      -0.67  0.96909  0.96977     0.07
# 1  precision   0.95074   0.94804      -0.28  0.94113  0.95037     0.98
# 2     recall   0.95000   0.92642      -2.48  0.94016  0.93264    -0.80
# 3         f1   0.95037   0.93711      -1.40  0.94064  0.94142     0.08
# 4    roc_auc   0.99615   0.96367      -3.26  0.97729  0.96994    -0.75
