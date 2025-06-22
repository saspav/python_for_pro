import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from some_functions import SEED, memory_compression, find_depth, train_valid_model, cv_params

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

map_dict = {'male': 1, 'female': 0}
train["sex"] = train["Sex"].map(map_dict).astype(np.int8)
valid["sex"] = valid["Sex"].map(map_dict).astype(np.int8)

model_columns = numeric_cols + ["sex"]

params = {'n_jobs': -1}

rf1, metrics_df, _ = train_valid_model(RandomForestRegressor, 1, params,
                                       train, valid, model_columns, target)

print(metrics_df)

opt_depth_rf = find_depth(RandomForestRegressor, params, train, valid, model_columns, target,
                          depths=range(6, 23, 2))

rf_param_grid = {
    'n_estimators': [100, 200, 300, 500],  # Количество деревьев
    'max_depth': [15, 16, 17],  # Глубина деревьев (None - без ограничений)
    'min_samples_split': [2, 5, 8],  # Минимальное число samples для разделения узла
    'min_samples_leaf': [1, 2, 5],  # Минимальное число samples в листе
    'max_features': [0.5, 0.8, 1.0]  # Доля признаков для разделения
}

# Поиск параметров по сетке
rf_best_params = cv_params(RandomForestRegressor(random_state=SEED), rf_param_grid,
                           train, valid, model_columns, target)

rf_best_grid = {'max_depth': 17, 'max_features': 0.5, 'min_samples_leaf': 1,
                'min_samples_split': 8, 'n_estimators': 500, 'n_jobs': -1}

rf2, metrics_df2, _ = train_valid_model(RandomForestRegressor, 2, rf_best_grid,
                                        train, valid, model_columns, target)

metrics_df = pd.concat([metrics_df, metrics_df2.drop(columns=['Metric'])], axis=1)
print(metrics_df)
