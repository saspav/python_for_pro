import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (  # Метрики оценки качества модели
    make_scorer, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from collections import Counter

# Зафиксируем сиды
SEED = 127
np.random.seed(SEED)
random.seed(SEED)


def find_optimal_threshold(y_true, y_proba):
    """
    Подбор оптимального порога
    :param y_true: истинные метки классов
    :param y_proba: предсказанные вероятности классов
    :return: оптимальный порог
    """
    n_samples = len(y_true)  # количество наблюдений
    P = sum(y_true)  # количество истинных меток "1"
    N = n_samples - P  # количество истинных меток "0"
    # Функция для подбора оптимального порога
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Вычисляем F1-score для каждого порога
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    # Находим порог с максимальным F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def find_optimal_threshold_by_accuracy(y_true, y_proba):
    """
    Подбор оптимального порога по метрике accuracy.
    :param y_true: истинные метки классов (0 или 1)
    :param y_proba: предсказанные вероятности положительного класса
    :return: оптимальный порог
    """
    thresholds = np.linspace(0, 1, 10_000)  # пробуем 10_000 порогов от 0 до 1
    best_accuracy = 0
    best_threshold = 0.5  # по умолчанию

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_threshold


def get_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Расчёт метрик для бинарной классификации.
    :param y_true: Истинные метки (0 или 1)
    :param y_pred: предсказанные метки (0 или 1)
    :param y_proba: вероятности положительного класса (по желанию)
    :return: словарь с метриками
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        # y_proba должен быть вероятностями положительного класса
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

    return metrics


def train_valid_model(model_class, model_num, model_params, df_train, df_valid,
                      model_cols, target_col):
    """
    Процедура обучения и валидации модели
    :param model_class: Модель, которую используем для обучения
    :param model_num: Порядковый номер модели
    :param model_params: Параметры модели
    :param df_train: обучающий ДФ
    :param df_valid: валидационный ДФ
    :param model_cols: список признаков
    :param target_col: целевая переменная
    :return: обученная модель, ДФ с метриками
    """
    X_train, y_train = df_train[model_cols], df_train[target_col]
    X_valid, y_valid = df_valid[model_cols], df_valid[target_col]

    if model_class.__name__ == 'GradientBoosting':
        model = model_class(**model_params)
        model.fit(X_train, y_train, eval_sets=[{'X': X_valid, 'y': y_valid}, ])
    else:
        model = model_class(**model_params, random_state=SEED)
        model.fit(X_train, y_train)

    # Получение вероятностей
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
    else:
        y_train_proba = model.predict(X_train)
        y_valid_proba = model.predict(X_valid)

    # Подбираем порог на валидационной выборке
    optimal_threshold = find_optimal_threshold(y_valid, y_valid_proba)
    # optimal_threshold = find_optimal_threshold_by_accuracy(y_valid, y_valid_proba)
    print(f"Оптимальный порог: {optimal_threshold:.4f}")  # Оптимальный порог: 0.2336
    # Применяем порог к обеим выборкам
    y_train_pred = (y_train_proba >= optimal_threshold).astype(int)
    y_valid_pred = (y_valid_proba >= optimal_threshold).astype(int)

    metrics_train = get_classification_metrics(y_train, y_train_pred, y_train_proba)
    metrics_valid = get_classification_metrics(y_valid, y_valid_pred, y_valid_proba)

    metric_t = f'Train{model_num}'
    metric_v = f'Valid{model_num}'

    # Создаем DataFrame
    metrics = pd.DataFrame({'Metric': list(metrics_train.keys()),
                            metric_t: list(metrics_train.values()),
                            metric_v: list(metrics_valid.values())})

    # Добавляем разницу между train и valid (в %)
    metrics[f'Diff{model_num},%'] = ((metrics[metric_v] - metrics[metric_t])
                                     / metrics[metric_t] * 100).round(2)
    return model, metrics, optimal_threshold


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

        best_col = valid_values.idxmax()
        best_val = valid_values.max()

        models.append(best_col)

        print(f"Метрика: {metric_name:<10} --> лучшая модель: {best_col} ({best_val:.4f})")

    result = Counter(models).most_common()[0]
    print('\nЛучшая модель: {} на {} метриках из {}'.format(*result, len(models)))


def plot_feature_importance(model, feature_names, name_model='CatBoost', top_n=20):
    """
    Визуализация важности признаков модели.
    :param model: Обученная модель
    :param feature_names: Список признаков (в том же порядке, что использовались для обучения)
    :param name_model: Имя модели для заголовка
    :param top_n: Сколько самых важных признаков отобразить
    """
    # Получаем важности признаков
    # importances = model.get_feature_importance()
    importances = model.feature_importances_

    # Создаём DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    # Отрисовка
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(top_n), palette='viridis')
    plt.title(f'{name_model} — Топ {top_n} признаков по важности')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.tight_layout()
    plt.show()
    return feat_imp_df  # возвращаем на всякий случай


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
        depths = range(3, 21)

    train_f1, valid_f1 = [], []
    for depth in depths:

        model_params['max_depth'] = depth

        _, metrics, _ = train_valid_model(model_class, 0, model_params,
                                          train, valid, model_columns, target)

        # Получаем значения метрик
        t_f1, v_f1 = metrics[metrics['Metric'] == 'f1'][['Train0', 'Valid0']].values[0]
        train_f1.append(t_f1)
        valid_f1.append(v_f1)

        print(f"Depth: {depth:2} | Train F1: {t_f1:.5f} | Valid F1: {v_f1:.5f}")

        # Ранняя остановка, если текущий F1 < минимального найденного,
        # и минимум был хотя бы 2 шага назад
        max_valid_f1 = max(valid_f1)
        max_index = valid_f1.index(max_valid_f1)
        if len(valid_f1) > max_index + 3 and valid_f1[-1] < max_valid_f1:
            print(f"Ранняя остановка: лучший Valid F1 ({max_valid_f1:.5f}) "
                  f"был на глубине {depths[max_index]}")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(depths[:len(train_f1)], train_f1, label='Train F1', marker='o')
    plt.plot(depths[:len(valid_f1)], valid_f1, label='Valid F1', marker='o')
    plt.xlabel('Max Depth')
    plt.ylabel('F1')
    plt.title('Зависимость F1 от глубины дерева')
    plt.legend()
    plt.grid(True)
    plt.show()

    return depths[max_index]


def make_submit(model, X_test, threshold, reverse_mapping, postfix='', save_to_excel=False):
    """
    Подготовка файла сабмита для загрузки на Каггл
    :param model: обученная модель
    :param X_test: подготовленные тестовые данные
    :param threshold: порог для классификации
    :param reverse_mapping: реверсный словарь целевой переменной
    :param postfix: постфикс для имени файла сабмита
    :param save_to_excel: сохранить X_test в эксель
    :return: Имя файла сабмита
    """
    # Целевая переменная
    target = 'Personality'
    # Формируем имя файла
    file_submit = f'submit_{model.__class__.__name__}{postfix}.csv'
    # Получаем предсказания
    # Получение вероятностей
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = model.predict(X_test)
    # Преобразуем вероятности в классы 0 / 1
    y_test_pred = (y_test_proba >= threshold).astype(int)
    # Записываем в датафрейм
    X_test[target] = y_test_pred
    # Возвращаем метки классов
    X_test[target] = X_test[target].map(reverse_mapping)
    # Сохраняем в файл
    X_test[target].to_csv(file_submit)
    if save_to_excel:
        X_test.to_excel(file_submit.replace('.csv', '.xlsx'))
    print(f'Сформирован файл сабмита: {file_submit}')
    return file_submit


def set_types(df, num_cols, cat_cols):
    # Явно задаем типы
    for col in num_cols:
        df[col] = df[col].astype(int)
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df


class DataTransform:
    def __init__(self, numeric_columns=None, category_columns=None, set_category=False,
                 features2drop=None, set_num_int=True, preprocessor=None, **kwargs):
        """
        Преобразование данных
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param set_category: установить категориальные колонки как "category"
        :param features2drop: колонки, которые нужно удалить
        :param set_num_int: после заполнения пропусков установить тип INT
        :param preprocessor: препроцессор для заполнения пропусков
        :param kwargs: параметры препроцессора
        """
        self.set_num_int = set_num_int
        self.set_category = set_category
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.features2drop = [] if features2drop is None else features2drop
        self.preprocessor = KNNImputer if preprocessor is None else preprocessor
        self.prep_kwargs = kwargs if kwargs else dict(n_neighbors=7)
        self.p_imputer = None
        self.cat_cols = self.category_columns.copy()
        # Целевая переменная
        self.target = 'Personality'
        # Колонки: числовые + категориальные
        self.model_columns = []
        # Колонки для заполнения пропусков
        self.imputer_cols = []
        # Колонки, используемые в модели с пропусками
        self.columns_with_nans = []
        self.columns_with_missing = []
        # Колонки, используемые в модели
        self.all_features = []
        # Исходный словарь кодирования целевой переменной
        self.mapping_target = {'Extrovert': 0, 'Introvert': 1}
        # Создание реверсного словаря
        self.reverse_mapping = {v: k for k, v in self.mapping_target.items()}
        # Словарь кодирования категориальных признаков
        self.mapping_yes_no = {'Yes': 1, 'No': 0, 'nan': 2}
        # Словарь группировок по целевой переменной
        self.grp_stats = {}
        self.grp_stats_cols = []

    def preprocess_data(self, df, fill_nan_cat=False):
        """
        Предобработка данных
        :param sample: датафрейм
        :param fill_nan_cat: заполнять пропуски категориальных переменных значением 'nan'
        :return: предобработанный датафрейм
        """
        for col in self.cat_cols:
            if fill_nan_cat:
                # Заполним пропуски категориальных переменных значением 'nan'
                df[col] = df[col].astype(str).fillna('nan')
            df[col] = df[col].map(self.mapping_yes_no)
        # Закодируем целевую переменную
        if self.target in df.columns:
            df[self.target] = df[self.target].map(self.mapping_target).astype(int)
        return df

    def make_attribute_columns(self, df):
        """
        Процедура формирования списков категориальных и цифровых колонок
        :param df: ДФ
        :return: списки категориальных и цифровых колонок
        """
        # Колонки, которые нужно удалить
        features2drop = self.features2drop + [self.target]

        # Выбираем категориальные колонки (включая строки и категории)
        category_columns = (df.drop(columns=features2drop, errors='ignore')
                            .select_dtypes(include=['object', 'category'])
                            .columns.tolist())

        # Выбираем числовые признаки
        numeric_columns = (df.drop(columns=features2drop, errors='ignore')
                           .select_dtypes(include=['number'])
                           .columns.tolist())

        # Колонки, используемые в модели
        self.model_columns = numeric_columns + category_columns
        return category_columns, numeric_columns

    def set_category_cols(self, df):
        if self.set_category:
            # Вернем категориальные признаки
            for col in self.cat_cols:
                df[col] = df[col].astype('category')
        else:
            for col in self.cat_cols:
                df[col] = df[col].astype(int)
        return df

    @staticmethod
    def apply_group_stats(df, group_stats, global_mean, feature_cols):
        for col in feature_cols:
            new_col = f"{col}_group_mean"
            df[new_col] = df[col].map(group_stats[col])
            df[new_col].fillna(global_mean, inplace=True)
        return df

    def fit(self, df, fill_nan_cat=False, add_new_features=False):
        """
        Формирование фич
        :param df: исходный ФД
        :param fill_nan_cat: заполнять пропуски категориальных переменных значением 'nan'
        :param add_new_features: добавить новые признаки
        :return: ДФ с агрегациями
        """
        df = df.copy()

        # Процедура формирования списков категориальных и цифровых колонок
        category_columns, numeric_columns = self.make_attribute_columns(df)

        # если нет категориальных колонок --> заполним их
        if not self.category_columns:
            self.category_columns = category_columns.copy()
            self.cat_cols = category_columns.copy()

        # если нет цифровых колонок --> заполним их
        if not self.numeric_columns:
            self.numeric_columns = numeric_columns.copy()

        self.columns_with_nans = []
        self.columns_with_missing = []
        for col in df.columns:
            if df[col].isnull().any() and col != 'Pers_orig':
                self.columns_with_nans.append(col)
                self.columns_with_missing.append(f"{col}_nan")

        # Предобработка данных
        df = self.preprocess_data(df.copy(), fill_nan_cat=fill_nan_cat)

        self.imputer_cols = self.model_columns.copy()

        # Формируем группировки по целевой переменной
        self.grp_stats_cols = []
        for col in self.imputer_cols:
            # Удаляем пропуски и приводим к int
            temp = df[[col, self.target]].dropna(subset=[col]).copy()
            temp[col] = temp[col].astype(int)
            # Считаем статистики
            self.grp_stats[col] = temp.groupby(col)[self.target].mean().to_dict()
            self.grp_stats_cols.append(f"{col}_tar_mean")

        # Создаем объект Imputer
        self.p_imputer = self.preprocessor(**self.prep_kwargs)
        self.p_imputer.fit(df[self.imputer_cols])

        if add_new_features:
            # Заполнение пропусков
            if self.set_num_int:
                df[self.imputer_cols] = (self.p_imputer.transform(df[self.imputer_cols])
                                         .round()
                                         .astype(int)
                                         )
            else:
                df[self.imputer_cols] = self.p_imputer.transform(df[self.imputer_cols])

            # Вернем категориальные признаки
            df = self.set_category_cols(df)

            # Добавление новых признаков
            df = self.add_new_features(df)

            # Добавление агрегатов по признакам
            pass

            # Процедура формирования списков категориальных и цифровых колонок
            self.category_columns, self.numeric_columns = self.make_attribute_columns(df)

    def transform(self, df, fill_nan_cat=False, add_grp_target=False, add_new_features=False):
        """
        Формирование остальных фич
        :param df: ДФ
        :param fill_nan_cat: заполнять пропуски категориальных переменных значением 'nan'
        :param add_grp_target: добавить группировки по целевой переменной
        :param add_new_features: добавить новые признаки
        :return: ДФ с фичами
        """
        df = df.copy()
        # Отметим строки в колонках с пропусками
        for col, col_nan in zip(self.columns_with_nans, self.columns_with_missing):
            df[col_nan] = df[col].isnull().astype(int)

        # Предобработка данных
        df = self.preprocess_data(df, fill_nan_cat=fill_nan_cat)

        # Заполнение пропусков
        if self.set_num_int:
            df[self.imputer_cols] = (self.p_imputer.transform(df[self.imputer_cols])
                                     .round()
                                     .astype(int)
                                     )
        else:
            df[self.imputer_cols] = self.p_imputer.transform(df[self.imputer_cols])

        if add_grp_target:
            for col in self.imputer_cols:
                new_col = f"{col}_tar_mean"
                df[new_col] = df[col].map(self.grp_stats[col])
                # df[new_col].fillna(df[self.target].mean(), inplace=True)

            # print(df.isna().sum())

        # Вернем категориальные признаки
        df = self.set_category_cols(df)

        all_features_add = []
        if add_new_features:
            # Добавление новых признаков
            df = self.add_new_features(df)
            all_features_add = df.drop(columns=self.target, errors='ignore').columns

        if not self.all_features:
            self.all_features = self.model_columns + self.columns_with_missing
            self.all_features.extend([col for col in all_features_add
                                      if col not in self.all_features])
            if add_grp_target:
                self.all_features.extend([col for col in self.grp_stats_cols
                                          if col not in self.all_features])

        model_columns = self.all_features.copy()
        if self.target in df.columns:
            model_columns.append(self.target)

        # Оставим только колонки для обучения модели в нужном нам порядке
        return df[model_columns]

    def fit_transform(self, df, fill_nan_cat=False, add_grp_target=False,
                      add_new_features=False):
        """
        Fit + transform data
        :param df: исходный ФД
        :param fill_nan_cat: заполнять пропуски категориальных переменных значением 'nan'
        :param add_grp_target: добавить группировки по целевой переменной
        :param add_new_features: добавить новые признаки
        :return: ДФ с новыми признаками
        """
        self.fit(df, fill_nan_cat=fill_nan_cat, add_new_features=add_new_features)
        df = self.transform(df, fill_nan_cat=fill_nan_cat, add_grp_target=add_grp_target,
                            add_new_features=add_new_features)
        return df

    @staticmethod
    def drop_constant_columns(df):
        # Ищем колонки с константным значением для удаления
        col_to_drop = []
        for col in df.columns:
            if df[col].nunique() == 1:
                col_to_drop.append(col)
        if col_to_drop:
            df.drop(columns=col_to_drop, inplace=True)
        return df

    @staticmethod
    def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление новых признаков
        :param df: исходный ДФ
        :return: ДФ с новыми признаками
        """
        df = df.copy()

        # 1. 📊 Биннинги признаков
        df['alone_bin'] = pd.cut(
            df['Time_spent_Alone'],
            bins=[-1, 2, 4, 11],
            labels=['low', 'medium', 'high']
        )  # Мало / средне / много времени в одиночестве

        df['friends_bin'] = pd.cut(
            df['Friends_circle_size'],
            bins=[-1, 5, 10, 15],
            labels=['few', 'medium', 'many']
        )  # Размер круга общения

        df['outside_bin'] = pd.cut(
            df['Going_outside'],
            bins=[-1, 3, 5, 7],
            labels=['homebody', 'balanced', 'outgoing']
        )  # Частота выхода из дома

        df['posts_bin'] = pd.cut(
            df['Post_frequency'],
            bins=[-1, 3, 6, 10],
            labels=['inactive', 'moderate', 'active']
        )  # Частота постинга

        df['events_bin'] = pd.cut(
            df['Social_event_attendance'],
            bins=[-1, 3, 6, 10],
            labels=['rare', 'moderate', 'frequent']
        )  # Частота участия в мероприятиях

        # 2. 🧠 Инженерные признаки

        # Социальная изоляция
        df['loneliness_index'] = df['Time_spent_Alone'] / (df['Friends_circle_size'] + 1)

        # Общая активность вне дома
        df['social_activity'] = df['Social_event_attendance'] + df['Going_outside']

        # Индекс интроверсии (если усталость от общения — +3 балла)
        df['introvert_score'] = (df['Time_spent_Alone'] +
                                 df['Drained_after_socializing'].astype(int) * 3)

        # Частота постов на одного друга
        df['post_per_friend'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)

        # Баланс оффлайн/онлайн активности
        df['event_vs_post_ratio'] = df['Social_event_attendance'] / (df['Post_frequency'] + 1)

        # Насколько человек активен и не устаёт от общества
        df['active_life_index'] = df['Going_outside'] * (
                1 - df['Drained_after_socializing'].astype(int))

        # Индикатор социальной тревожности (оба признака = 1)
        df['social_anxiety'] = (df['Stage_fear'].astype(int) &
                                df['Drained_after_socializing'].astype(int)).astype(int)

        # 3. 🔍 Признак "есть ли пропуски вообще"
        nan_cols = [col for col in df.columns if col.endswith('_nan')]
        df['has_any_missing'] = df[nan_cols].sum(axis=1).gt(0).astype(int)

        return df


def make_train_valid(test_size=0.2, return_full_df=False, add_original_df=False,
                     round_to_int=False):
    """
    Функция чтения данных и разделения на тренировочную и валидационную выборки
    :param test_size: размер валидационной части
    :param return_full_df: вернуть полный тренировочный ДФ
    :param add_original_df: добавить метки оригинального ДФ
    :param round_to_int: округлить float до int
    :return: train, valid, test
    """
    # Чтение данных
    df = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    if add_original_df:
        merge_cols = test.drop(columns='id').columns.to_list()
        # Загрузим оригинальный ДФ на основе которого были сделаны train.csv и test.csv
        df_orig = (pd.read_csv('personality_datasert.csv')
                   .rename(columns={'Personality': 'Pers_orig'})
                   .fillna(-99)
                   .drop_duplicates(subset=merge_cols)
                   )
        if round_to_int:
            # Округляем заполненные NaN до целого
            for col in df_orig.select_dtypes(include=['number']):
                df_orig[col] = df_orig[col].round()
        # Закодируем 'Pers_orig' как целевую переменную
        mapping_target = {'Extrovert': 0, 'Introvert': 1}
        df_orig['Pers_orig'] = df_orig['Pers_orig'].map(mapping_target).astype(int)
        # Добавим истинные метки в оба набора данных
        df = df.fillna(-99).merge(df_orig, on=merge_cols, how='left').replace(-99, np.nan)
        test = test.fillna(-99).merge(df_orig, on=merge_cols, how='left').replace(-99, np.nan)
        df = df.merge(df_orig, on=merge_cols, how='left')
        test = test.merge(df_orig, on=merge_cols, how='left')

    # Колонка "id" не несет смысла - это индекс
    df.set_index("id", inplace=True)
    test.set_index("id", inplace=True)

    # Целевая переменная
    target = 'Personality'

    # Т.к. у нас наблюдается дисбаланс классов: класс с меткой "1" всего 26%
    # Будем делить со стратификацией
    train, valid = train_test_split(df, test_size=test_size, stratify=df[target],
                                    random_state=SEED)
    if return_full_df:
        return train, valid, test, df

    return train, valid, test


def add_group_stats_transform(df: pd.DataFrame, train_stats: dict = None) -> tuple[
    pd.DataFrame, dict]:
    df = df.copy()

    bin_cols = ['alone_bin', 'friends_bin', 'events_bin', 'outside_bin', 'posts_bin']
    features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency']

    # 1. Групповые статистики (по бинам)
    for bin_col in bin_cols:
        for col in features:
            grp = df.groupby(bin_col)[col]
            df[f'{bin_col}_{col}_mean'] = grp.transform('mean')
            df[f'{bin_col}_{col}_std'] = grp.transform('std')

    # 2. Z-оценки (глобальные — по всему train)

    stats = {} if train_stats is None else train_stats  # словарь со средними/стд

    for col in features:
        if train_stats is None:
            mean = df[col].mean()
            std = df[col].std()
            stats[col] = (mean, std)
        else:
            mean, std = stats[col]

        df[f'{col}_zscore'] = (df[col] - mean) / std

    return df, stats


def compute_group_stats(df_train: pd.DataFrame) -> dict:
    """
    Вычисляет средние и std значения целевых признаков по каждой бин-группе.
    Возвращает словарь DataFrame'ов с агрегатами по группам:
    """
    stats = {}
    target_cols = ['Post_frequency', 'Time_spent_Alone', 'Going_outside',
                   'Social_event_attendance']

    bin_cols = ['alone_bin', 'friends_bin', 'events_bin', 'outside_bin', 'posts_bin']

    for bin_col in bin_cols:
        group_stat = df_train.groupby(bin_col)[target_cols].agg(['mean', 'std']).reset_index()
        # Flatten MultiIndex
        group_stat.columns = [f"{bin_col}_{col[0]}_{col[1]}" if col[1] else col[0] for col in
                              group_stat.columns]
        stats[bin_col] = group_stat

    return stats


def apply_group_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Добавляет признаки валидации/теста на основе групп из train:
    :param df:
    :param stats:
    :return:
    """
    df = df.copy()
    for bin_col, group_df in stats.items():
        df = df.merge(group_df, how='left', left_on=bin_col, right_on=group_df.columns[0])
    return df


if __name__ == '__main__':
    train, valid, test, df = make_train_valid(return_full_df=True, add_original_df=True,
                                              round_to_int=False)

    print(train.shape, valid.shape, test.shape)
