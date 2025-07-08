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
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMClassifier, Dataset, cv

import warnings

warnings.filterwarnings("ignore")

from some_functions_clf import (SEED, make_train_valid, DataTransform, find_depth,
                                train_valid_model, make_submit)

def objective(trial):
    # boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    boosting_type = 'gbdt'

    params_cv = {
        'boosting_type': boosting_type,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': -1,
        'feature_pre_filter': False,  # 💡 критично!
    }

    # GBDT и DART используют bagging
    if boosting_type != 'goss':
        params_cv['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        params_cv['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.6, 1.0)
        params_cv['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 7)
    else:
        params_cv['subsample'] = 1.0  # для совместимости

    # DART-специфичные параметры
    if boosting_type == 'dart':
        params_cv['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params_cv['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)
        params_cv['max_drop'] = trial.suggest_int('max_drop', 10, 100)
        params_cv['uniform_drop'] = trial.suggest_categorical('uniform_drop', [True, False])

    # Кросс-валидация с ранней остановкой и pruning
    cv_result = cv(
        params=params_cv,
        train_set=dtrain,
        num_boost_round=500,
        nfold=3,
        stratified=True,
        # early_stopping_rounds=50,
        metrics='auc',
        seed=SEED,
        callbacks=[LightGBMPruningCallback(trial, 'auc')],
        # verbose_eval=False
    )
    # Возвращаем лучший AUC
    return np.mean(cv_result['valid auc-mean'])


# Настройки отображения данных в Pandas
pd.set_option('display.max_columns', 50)  # Отображать до 50 столбцов
pd.set_option('display.precision', 5)  # Отображение ДФ с 5-ю знаками после запятой

# Зафиксируем сиды
np.random.seed(SEED)
random.seed(SEED)

# Целевая переменная
target = 'Personality'

train, valid, test = make_train_valid()

dts = DataTransform(set_category=True, preprocessor=KNNImputer, n_neighbors=5)

# Применяем трансформации
train = dts.fit_transform(train)
valid = dts.transform(valid)

# Колонки для моделей
model_columns = dts.all_features
cat_cols = dts.category_columns

# Обучение модели с параметрами по умолчанию
params = {'categorical_feature': cat_cols, 'force_row_wise': True,
          'n_jobs': -1, 'verbosity': -1}

xb1, metrics_df5, _ = train_valid_model(LGBMClassifier, 'LGC', params,
                                        train, valid, model_columns, target)

print(metrics_df5, '\n')
metrics_df = metrics_df5

# # Подберем оптимальную глубину дерева
# opt_depth_xb = find_depth(LGBMClassifier, params, train, valid, model_columns, target)
# # Ранняя остановка: лучший Valid F1 (0.94136) был на глубине 6
# exit()

X_train, y_train = train[model_columns], train[target]
X_valid, y_valid = valid[model_columns], valid[target]

# Подготовка данных в DMatrix
dtrain = Dataset(X_train, y_train, categorical_feature=cat_cols, free_raw_data=False)
dvalid = Dataset(X_valid, y_valid, categorical_feature=cat_cols, free_raw_data=False)
# Отключение инфо выводов
optuna.logging.set_verbosity(optuna.logging.WARNING)

# # Создаем study с направлением maximize
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=200, show_progress_bar=True)
#
# # Вывод результатов
# print(f"Лучший AUC: {study.best_value:.4f}")  # Форматирование для AUC
# print("Лучшие параметры:", study.best_params)

# lg_best_grid = {'boosting_type': 'goss', 'learning_rate': 0.010527356559607244,
#                 'num_leaves': 54, 'max_depth': 4, 'min_child_samples': 26,
#                 'colsample_bytree': 0.6498369482631124, 'reg_alpha': 0.001996825899468912,
#                 'reg_lambda': 0.0029056516001497996,
#                 'n_jobs': -1, 'verbosity': -1}

lg_best_grid = {'learning_rate': 0.012168409321350424, 'num_leaves': 78, 'max_depth': 4,
                'min_child_samples': 31, 'colsample_bytree': 0.7152350985965117,
                'reg_alpha': 0.0152465652290482, 'reg_lambda': 0.0026314520377431084,
                'subsample': 0.7475755007546926, 'bagging_fraction': 0.6356741993284825,
                'bagging_freq': 6,
                'n_jobs': -1, 'verbosity': -1}

lg2, metrics_df6, optimal_threshold = train_valid_model(LGBMClassifier, 6, lg_best_grid,
                                                        train, valid, model_columns, target)

metrics_df = pd.concat([metrics_df, metrics_df6.drop(columns=['Metric'])], axis=1)
print(metrics_df)

# Трансформируем признаки тестовой выборки
test = dts.transform(test)

optimal_threshold = 0.5

# Вызываем функцию для формирования сабмита: передаем обученную модель
_ = make_submit(lg2, test[model_columns], optimal_threshold, dts.reverse_mapping)

# Оптимальный порог: 0.4264 (max_depth = 4)
#       Metric  TrainLGC  ValidLGC  DiffLGC,%   Train6   Valid6  Diff6,%
# 0   accuracy   0.97173   0.97004      -0.17  0.96896  0.96869    -0.03
# 1  precision   0.94538   0.95042       0.53  0.94271  0.94921     0.69
# 2     recall   0.94611   0.93368      -1.31  0.93782  0.92953    -0.88
# 3         f1   0.94575   0.94198      -0.40  0.94026  0.93927    -0.11
# 4    roc_auc   0.99309   0.96473      -2.86  0.97151  0.96841    -0.32

# Оптимальный порог: 0.2792 (max_depth = 5)
#       Metric  TrainLGC  ValidLGC  DiffLGC,%   Train6   Valid6  Diff6,%
# 0   accuracy   0.97173   0.97004      -0.17  0.96896  0.96896     0.00
# 1  precision   0.94538   0.95042       0.53  0.94225  0.94926     0.74
# 2     recall   0.94611   0.93368      -1.31  0.93834  0.93057    -0.83
# 3         f1   0.94575   0.94198      -0.40  0.94029  0.93982    -0.05
# 4    roc_auc   0.99309   0.96473      -2.86  0.97544  0.96955    -0.60

# Оптимальный порог: 0.3020 (max_depth = 4 - 'gbdt')
#       Metric  TrainLGC  ValidLGC  DiffLGC,%   Train6   Valid6  Diff6,%
# 0   accuracy   0.97173   0.97004      -0.17  0.96903  0.96869    -0.03
# 1  precision   0.94538   0.95042       0.53  0.94295  0.94921     0.66
# 2     recall   0.94611   0.93368      -1.31  0.93782  0.92953    -0.88
# 3         f1   0.94575   0.94198      -0.40  0.94038  0.93927    -0.12
# 4    roc_auc   0.99309   0.96473      -2.86  0.97425  0.96932    -0.51