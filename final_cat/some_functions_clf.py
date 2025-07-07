import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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

    model = model_class(**model_params, random_state=SEED)

    model.fit(X_train, y_train)

    # Получение вероятностей
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_valid_proba = model.predict_proba(X_valid)[:, 1]

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
        if len(valid_f1) > max_index + 2 and valid_f1[-1] < max_valid_f1:
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


def make_submit(model, X_test, threshold, reverse_mapping):
    """
    Подготовка файла сабмита для загрузки на Каггл
    :param model: обученная модель
    :param X_test: подготовленные тестовые данные
    :param threshold: порог для классификации
    :param reverse_mapping: реверсный словарь целевой переменной
    :return: Имя файла сабмита
    """
    # Целевая переменная
    target = 'Personality'
    # Формируем имя файла
    file_submit = f'submit_{model.__class__.__name__}.csv'
    # Получаем предсказания
    y_test_proba = model.predict_proba(X_test)[:, 1]
    # Преобразуем вероятности в классы 0 / 1
    y_test_pred = (y_test_proba >= threshold).astype(int)
    # Записываем в датафрейм
    X_test[target] = y_test_pred
    # Возвращаем метки классов
    X_test[target] = X_test[target].map(reverse_mapping)
    # Сохраняем в файл
    X_test[target].to_csv(file_submit)
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
                 features2drop=None, preprocessor=None, **kwargs):
        """
        Преобразование данных
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param set_category: установить категориальные колонки как "category"
        :param features2drop: колонки, которые нужно удалить
        :param preprocessor: препроцессор для заполнения пропусков
        :param kwargs: параметры препроцессора
        """
        self.set_category = set_category
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.features2drop = [] if features2drop is None else features2drop
        self.preprocessor = KNNImputer if preprocessor is None else preprocessor
        self.prep_kwargs = kwargs if kwargs else dict(n_neighbors=7)
        self.p_imputer = None
        # Целевая переменная
        self.target = 'Personality'
        # Колонки: числовые + категориальные
        self.model_columns = []
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

    def preprocess_data(self, df, fill_nan_cat=False):
        """
        Предобработка данных
        :param sample: датафрейм
        :param fill_nan_cat: заполнять пропуски категориальных переменных значением 'nan'
        :return: предобработанный датафрейм
        """
        for col in self.category_columns:
            if fill_nan_cat:
                # Заполним пропуски категориальных переменных значением 'nan'
                df[col] = df[col].fillna('nan')
            df[col] = df[col].map(self.mapping_yes_no)
        if self.target in df.columns:
            # Закодируем целевую переменную
            df[self.target] = df[self.target].map(self.mapping_target).astype(int)
        return df

    def fit(self, df, fill_nan_cat=False):
        """
        Формирование фич
        :param df: исходный ФД
        :param fill_nan_cat: заполнять пропуски категориальных переменных значением 'nan'
        :return: ДФ с агрегациями
        """
        # Колонки, которые нужно удалить
        features2drop = self.features2drop + [self.target]

        # если нет категориальных колонок --> заполним их
        if not self.category_columns:
            # Выбираем категориальные колонки (включая строки и категории)
            self.category_columns = df.drop(columns=features2drop).select_dtypes(
                include=['object', 'category']).columns.tolist()

        # если нет цифровых колонок --> заполним их
        if not self.numeric_columns:
            # Выбираем числовые признаки
            self.numeric_columns = df.drop(columns=features2drop).select_dtypes(
                include=['number']).columns.tolist()

        # Колонки, используемые в модели
        self.model_columns = self.numeric_columns + self.category_columns

        self.columns_with_nans = []
        self.columns_with_missing = []
        for col in df.columns:
            if df[col].isnull().any():
                self.columns_with_nans.append(col)
                self.columns_with_missing.append(f"{col}_nan")

        # Предобработка данных
        df = self.preprocess_data(df.copy(), fill_nan_cat=fill_nan_cat)

        # Создаем объект Imputer
        self.p_imputer = self.preprocessor(**self.prep_kwargs)
        self.p_imputer.fit(df[self.model_columns])

    def transform(self, df, fill_nan_cat=False):
        """
        Формирование остальных фич
        :param df: ДФ
        :param fill_nan_cat: заполнять пропуски категориальных переменных значением 'nan'
        :return: ДФ с фичами
        """
        df = df.copy()
        # Отметим строки в колонках с пропусками
        for col, col_nan in zip(self.columns_with_nans, self.columns_with_missing):
            df[col_nan] = df[col].isnull().astype(int)

        # Предобработка данных
        df = self.preprocess_data(df, fill_nan_cat=fill_nan_cat)

        # Заполнение пропусков
        df[self.model_columns] = self.p_imputer.transform(df[self.model_columns]).astype(int)

        if self.set_category:
            # Вернем категориальные признаки
            for col in self.category_columns:
                df[col] = df[col].astype('category')

        if not self.all_features:
            self.all_features = self.model_columns + self.columns_with_missing

        model_columns = self.all_features.copy()
        if self.target in df.columns:
            model_columns.append(self.target)

        # Оставим только колонки для обучения модели в нужном нам порядке
        return df[model_columns]

    def fit_transform(self, df, fill_nan_cat=False):
        """
        Fit + transform data
        :param df: исходный ФД
        :param fill_nan_cat: заполнять пропуски категориальных переменных значением 'nan'
        :return: ДФ с новыми признаками
        """
        self.fit(df, fill_nan_cat=fill_nan_cat)
        df = self.transform(df, fill_nan_cat=fill_nan_cat)
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


def make_train_valid(test_size=0.2):
    """
    Функция чтения данных и разделения на тренировочную и валидационную выборки
    :param test_size: размер валидационной части
    :return: train, valid, test
    """
    # Чтение данных
    df = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Колонка "id" не несет смысла - это индекс
    df.set_index("id", inplace=True)
    test.set_index("id", inplace=True)

    # Целевая переменная
    target = 'Personality'

    # Т.к. у нас наблюдается дисбаланс классов: класс с меткой "1" всего 26%
    # Будем делить со стратификацией
    train, valid = train_test_split(df, test_size=test_size, stratify=df[target],
                                    random_state=SEED)
    return train, valid, test
