import numpy as np
import pandas as pd
import random

import optuna
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMRegressor, Dataset

from sklearn.model_selection import train_test_split

from some_functions import SEED, memory_compression, find_depth, train_valid_model


def objective(trial):
    """Функция для оптимизации гиперпараметров LightGBM."""
    params_cv = {
        'seed': SEED,
        'verbosity': -1,
        # Параметры для оптимизации
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 30, 150, step=10),
        'max_depth': trial.suggest_int('max_depth', 6, 13),
        'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 0.9, 1.0]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                      [0.7, 0.8, 0.9, 1.0]),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1),  # L1-регуляризация
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1),  # L2-регуляризация
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    if params_cv["boosting_type"] != "goss":
        params_cv["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)
        params_cv["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)

    loss_function = 'rmse'

    clf = LGBMRegressor(
        objective=loss_function,
        random_seed=SEED,
        verbose=-1,
        **params_cv
    )

    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
            categorical_feature=cat_cols,
            eval_metric=loss_function,
            callbacks=[LightGBMPruningCallback(trial, metric=loss_function)],
            )

    return clf.best_score_['valid_0'][loss_function]


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

params = {'force_row_wise': True, 'n_jobs': -1, 'verbosity': -1}

lg1, metrics_df5, _ = train_valid_model(LGBMRegressor, 5, params,
                                        train, valid, model_columns, target)

print(metrics_df5)
metrics_df = metrics_df5

# Подберем оптимальную глубину дерева
opt_depth_lg = find_depth(LGBMRegressor, params, train, valid, model_columns, target)

X_train, y_train = train[model_columns], train[target]
X_valid, y_valid = valid[model_columns], valid[target]

# Подготовка данных в DMatrix
dtrain = Dataset(X_train, y_train, categorical_feature=cat_cols)
dvalid = Dataset(X_valid, y_valid, categorical_feature=cat_cols)

# Создаем study и запускаем оптимизацию
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200, show_progress_bar=True)  # 200 итераций

# Вывод результатов
print(f"Лучший RMSE: {study.best_value:.2f}")
print("Лучшие параметры:", study.best_params)

lg_best_grid = {'boosting_type': 'gbdt', 'learning_rate': 0.2, 'num_leaves': 140,
                'max_depth': 13, 'subsample': 0.7, 'colsample_bytree': 1.0,
                'reg_alpha': 0.393, 'reg_lambda': 0.33888189, 'min_child_samples': 41,
                'bagging_fraction': 0.715, 'bagging_freq': 6, 'n_jobs': -1, 'verbosity': -1}

lg2, metrics_df6, _ = train_valid_model(LGBMRegressor, 6, lg_best_grid,
                                        train, valid, model_columns, target)

metrics_df = pd.concat([metrics_df, metrics_df6.drop(columns=['Metric'])], axis=1)
print(metrics_df6)
