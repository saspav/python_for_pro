# Задание:
# Найдите среднюю оценку по каждому предмету.
# Найдите медианную оценку по каждому предмету.
# Вычислите стандартное отклонение по каждому предмету.
# Определите предмет с самой высокой средней оценкой.

import numpy as np
import pandas as pd

students_data = {'math': [85, 78, 92, 70, 88],
                 'physics': [90, 82, 76, 85, 89],
                 'informatics': [88, 92, 80, 87, 90]
                 }
result = {}
for key, values in students_data.items():
    print(f'Предмет: {key}')
    for idx, (name_stat, func) in enumerate(zip(
            ('Средняя', 'Медианная', 'Стандартное отклонение'),
            (np.mean, np.median, np.std))):
        print(f'\t{name_stat} оценк{"иа"[idx < 2]}: {func(values):.1f}')
        # Сохраним в словарь среднюю оценку
        if not idx:
            result[key] = func(values)

# Список предметов
subjects = list(result.keys())
# Массив оценок
mean_scores = np.array(list(result.values()))
# Получаем предмет по индексу с максимальным значением массива
print('\nПредмет с самой высокой средней оценкой:', subjects[mean_scores.argmax()])

# V2: Сделаем из словаря датафрейм, т.к. решение будет проще
df = pd.DataFrame(students_data)

# Получаем статистики
stats = df.describe()
for name_col in stats.columns:
    print(f'Предмет: {name_col}')
    for idx, (name_stat, func) in enumerate(zip(
            ('Средняя', 'Медианная', 'Стандартное отклонение'),
            ('mean', '50%', 'std'))):
        print(f'\t{name_stat} оценк{"иа"[idx < 2]}: {stats.loc[func, name_col]:.1f}')

print('\nПредмет с самой высокой средней оценкой:', stats.columns[stats.loc['mean'].argmax()])