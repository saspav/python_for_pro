# Задание 2: Базовая статистика
# Напишите программу для вывода моды, медианы и среднего


import random
import numpy as np
from collections import Counter
from scipy import stats

# Устанавливаем фиксированный сид
random.seed(42)

# Генерируем список из 10000 случайных целых чисел
data = [random.randint(1, 100) for _ in range(10_000)]


def calculate_statistics(data):
    # Подсчет среднего
    mean = sum(data) / len(data)

    # Сортировка данных для вычисления медианы
    sorted_data = sorted(data)
    n = len(sorted_data)

    # Подсчет медианы
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]

    # Подсчет моды
    data_counts = Counter(data)
    mode_data = data_counts.most_common()
    max_count = mode_data[0][1]
    mode = [value for value, count in mode_data if count == max_count]

    return mean, median, mode[0]


# Вычисляем статистику
mean, median, mode = calculate_statistics(data)

print(f"Список данных: {data[:10]}...")
print(f"Среднее: {mean}")
print(f"Медиана: {median}")
print(f"Мода: {mode}")

# Вычисляем статистику с использованием scipy.stats
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=False)[0]

print(f"Среднее: {mean}")
print(f"Медиана: {median}")
print(f"Мода: {mode}")