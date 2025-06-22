import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (  # Метрики оценки качества модели
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    make_scorer,
)
from collections import Counter

# Зафиксируем сиды
SEED = 127
np.random.seed(SEED)
random.seed(SEED)


# Переиспользуем все функции с прошлого ноутбука

def memory_compression(df, use_category=True, use_float=True, exclude_columns=None):
    """
    Изменение типов данных для экономии памяти
    :param df: исходный ДФ
    :param use_category: преобразовывать строки в категорию
    :param use_float: преобразовывать float в пониженную размерность
    :param exclude_columns: список колонок, которые нужно исключить из обработки
    :return: сжатый ДФ
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:

        if exclude_columns and col in exclude_columns:
            continue

        if str(df[col].dtype)[:4] in 'datetime':
            continue

        elif str(df[col].dtype) not in ('object', 'category'):
            col_min = df[col].min()
            col_max = df[col].max()
            if str(df[col].dtype)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif use_float and str(df[col].dtype)[:5] == 'float':
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif (col_min > np.finfo(np.float32).min
                      and col_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        elif use_category and str(df[col].dtype) == 'object':
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f'Исходный размер датасета в памяти '
          f'равен {round(start_mem, 2)} мб.')
    print(f'Конечный размер датасета в памяти '
          f'равен {round(end_mem, 2)} мб.')
    print(f'Экономия памяти = {(1 - end_mem / start_mem):.1%}')
    return df


def rmsle(y_true, y_pred):
    """
    Расчет метрики Root Mean Squared Logarithmic Error
    :param y_true: y_true
    :param y_pred: y_pred
    :return: RMSLE 
    """
    y_true = np.array(y_true)
    y_pred = np.maximum(0, np.array(y_pred))
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def get_metrics(y_true, y_pred):
    """
    Расчет метрики Root Mean Squared Logarithmic Error
    :param y_true: y_true
    :param y_pred: y_pred
    :return: метрики 
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmsle_ = rmsle(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'rmsle': rmsle_}


def calculate_outliers(dfs, q_range=1.5):
    """
    Функция для расчета выбросов через IQR
    :param dfs: датафрейм
    :param q_range: диапазон IQR   
    :return: маска с выбросами
    """
    Q1 = dfs.quantile(0.25)
    Q3 = dfs.quantile(0.75)
    IQR = (Q3 - Q1) * q_range
    return (dfs < (Q1 - IQR)) | (dfs > (Q3 + IQR))


def train_valid_model(model_class, model_num, model_params, df_train, df_valid,
                      model_cols, target_col, target_log=False):
    """
    Процедура обучения и валидации модели
    :param model_class: Модель, которую используем для обучения
    :param model_num: Порядковый номер модели
    :param model_params: Параметры модели
    :param df_train: обучающий ДФ
    :param df_valid: валидационный ДФ
    :param model_cols: список признаков
    :param target_col: целевая переменная
    :param target_log: логарифмировать целевой признак
    :return: обученная модель, ДФ с метриками, Серия с важностью признаков
    """
    X_train, y_train = df_train[model_cols], df_train[target_col]
    X_valid, y_valid = df_valid[model_cols], df_valid[target_col]

    model = model_class(**model_params, random_state=SEED)

    if target_log:
        # Логарифмируем целевой признак
        model.fit(X_train, np.log1p(y_train))
        # Преобразуем в нормальный масштаб предсказания
        y_pred_train = np.expm1(model.predict(X_train))
        y_pred_valid = np.expm1(model.predict(X_valid))
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)

    metrics_train = get_metrics(y_train, y_pred_train)
    metrics_valid = get_metrics(y_valid, y_pred_valid)

    metric_t = f'Train{model_num}'
    metric_v = f'Valid{model_num}'

    # Создаем DataFrame
    metrics = pd.DataFrame({'Metric': list(metrics_train.keys()),
                            metric_t: list(metrics_train.values()),
                            metric_v: list(metrics_valid.values())})

    # Добавляем разницу между train и valid (в %)
    metrics[f'Diff{model_num},%'] = ((metrics[metric_v] - metrics[metric_t])
                                     / metrics[metric_t] * 100).round(2)
    return model, metrics, model.feature_importances_


def find_best_model(metrics_df):
    """
    Процедура поиска лучшей модели по метрикам
    :param metrics_df: ДФ с метриками
    :return: None
    """
    models = []
    for idx, row in metrics_df.iterrows():
        metric_name = row['Metric']

        # Отбираем только численные значения из колонок Valid*
        valid_cols = [col for col in metrics_df.columns if col.startswith('Valid')]
        valid_values = row[valid_cols]

        if metric_name.strip().lower() == 'r2':
            best_col = valid_values.idxmax()
            best_val = valid_values.max()
        else:
            best_col = valid_values.idxmin()
            best_val = valid_values.min()

        models.append(best_col)

        print(f"Метрика: {metric_name:<5} --> лучшая модель: {best_col} ({best_val:.4f})")

    result = Counter(models).most_common()[0]
    print('\nЛучшая модель: {} на {} метриках из {}'.format(*result, len(models)))


def find_depth(model_class, model_params, train, valid, model_columns, target, depths=None):
    """
    Функция поиска оптимальной глубины деревьев
    :param model_class: Модель, которую используем для обучения
    :param model_params: Параметры модели
    :param train: обучающий ДФ
    :param valid: валидационный ДФ
    :param model_columns: список признаков
    :param target: целевая переменная
    :param depths: диапазон поиска глубины дерева
    :return: Оптимальная глубина дерева
    """
    if depths is None:
        depths = range(4, 21, 2)

    train_rmse, valid_rmse = [], []
    for depth in depths:

        model_params['max_depth'] = depth

        _, metrics, _ = train_valid_model(model_class, 0, model_params,
                                          train, valid, model_columns, target)

        # Получаем значения метрик
        t_rmse, v_rmse = metrics[metrics['Metric'] == 'rmse'][['Train0', 'Valid0']].values[0]
        train_rmse.append(t_rmse)
        valid_rmse.append(v_rmse)

        print(f"Depth: {depth:2} | Train RMSE: {t_rmse:6.2f} | Valid RMSE: {v_rmse:6.2f}")

        # Ранняя остановка, если текущий RMSE > минимального найденного,
        # и минимум был хотя бы 2 шага назад
        min_valid_rmse = min(valid_rmse)
        min_index = valid_rmse.index(min_valid_rmse)
        if len(valid_rmse) > min_index + 2 and valid_rmse[-1] > min_valid_rmse:
            print(f"Ранняя остановка: лучший Valid RMSE ({min_valid_rmse:.2f}) "
                  f"был на глубине {depths[min_index]}")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(depths[:len(train_rmse)], train_rmse, label='Train RMSE', marker='o')
    plt.plot(depths[:len(valid_rmse)], valid_rmse, label='Validation RMSE', marker='o')
    plt.xlabel('Max Depth')
    plt.ylabel('RMSE')
    plt.title('Зависимость RMSE от глубины дерева')
    plt.legend()
    plt.grid(True)
    plt.show()

    return depths[min_index]


def cv_params(model, param_grid, df_train, df_valid,
              model_cols, target_col, target_log=False, cv_folds=3):
    """
    Процедура подбора гиперпараметров
    :param model: Экземпляр регрессионной модели
    :param param_grid: сетка для перебора параметров
    :param df_train: тренировочный ДФ
    :param df_valid: валидационный ДФ
    :param model_cols: списков признаков для модели
    :param target_col: целевой признак
    :param target_log: нужно ли логарифмировать целевой признак
    :param cv_folds: количество фолдов для кросс-валидации
    :return:
    """
    X_train, y_train = df_train[model_cols], df_train[target_col]
    X_valid, y_valid = df_valid[model_cols], df_valid[target_col]

    scoring = 'neg_root_mean_squared_error'

    opt_params = GridSearchCV(
        estimator=model,  # Модель
        param_grid=param_grid,  # Параметры
        scoring=scoring,  # Стратегия валидации
        cv=cv_folds,  # Количество фолдов кросс валидации
        n_jobs=-1,  # Количество потоков для обучения, -1 = все
        verbose=2,
    )

    if target_log:
        # Логарифмируем целевой признак
        opt_params.fit(X_train, np.log1p(y_train))
    else:
        opt_params.fit(X_train, y_train)

    print(f'Best score: {round(-opt_params.best_score_, 3)}\n')
    print(f'Best parameters: {opt_params.best_params_}')
    return opt_params.best_params_
