import random
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
# from lightautoml.automl.presets.whitebox_presets import WhiteBoxPreset
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import warnings

warnings.filterwarnings("ignore")

from some_functions_clf import SEED, make_train_valid, DataTransform

from set_all_seeds import set_all_seeds

__import__("warnings").filterwarnings('ignore')


def acc_metric(y_true, y_pred, **kwargs):
    return accuracy_score(y_true, (y_pred > 0.5).astype(int), **kwargs)


def f1_metric(y_true, y_pred, **kwargs):
    return f1_score(y_true, (y_pred > 0.5).astype(int), **kwargs)


set_all_seeds(seed=SEED)
# Зафиксируем сиды

SEED = 127
np.random.seed(SEED)
random.seed(SEED)

if __name__ == '__main__':
    # Целевая переменная
    target = 'Personality'

    train, valid, test, df = make_train_valid(return_full_df=True)

    dts = DataTransform(set_category=False, preprocessor=KNNImputer, n_neighbors=5)

    # Применяем трансформации
    # train = dts.fit_transform(train)
    train = dts.fit_transform(df)  # в модель отдадим весь датасет
    valid = dts.transform(valid)

    N_THREADS = 32  # threads cnt for lgbm and linear models
    N_FOLDS = 5  # folds cnt for AutoML
    TEST_SIZE = 0.2  # Test size for metric check
    TIMEOUT = 60 * 30  # Time in seconds for automl run

    # Задаем тип задачи
    task = Task('binary', metric=roc_auc_score)

    roles = {'target': target}

    # Настройка автоматической модели
    automl = TabularAutoML(task=task,
                           timeout=TIMEOUT,
                           cpu_limit=N_THREADS,
                           memory_limit=64,
                           general_params={
                               'use_algos': [
                                   ['lgb_tuned',
                                    'xgb_tuned',
                                    'cb_tuned',
                                    ],
                                   ['cb_tuned',
                                    ],
                               ],

                               # 'use_algos': 'auto',

                           },
                           reader_params={'n_jobs': N_THREADS,
                                          'cv': N_FOLDS,
                                          'random_state': SEED},
                           )

    oof_pred = automl.fit_predict(train, roles=roles, verbose=1)

    valid_pred = automl.predict(valid)

    print('OOF score: {}'.format(roc_auc_score(train[target].values, oof_pred.data[:, 0])))
    print('VAL score: {}'.format(roc_auc_score(valid[target].values, valid_pred.data[:, 0])))

    # Трансформируем признаки тестовой выборки
    test = dts.transform(test)
    # Получаем предсказания
    test_pred = automl.predict(test)
    # Преобразуем вероятности в метки классов
    y_test_pred = (test_pred.data[:, 0] >= 0.5).astype(int)

    # Записываем в датафрейм
    test[target] = y_test_pred
    # Возвращаем метки классов
    test[target] = test[target].map(dts.reverse_mapping)
    # Сохраняем в файл
    file_submit = 'TabularAutoML.csv'
    test[target].to_csv(file_submit)
    # Вывод информации о модели
    print(automl.create_model_str_desc())
