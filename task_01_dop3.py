# Задание 3: Анализ текста
# Напишите программу, которая анализирует текст, введенный пользователем, и выводит
# количество слов, количество уникальных слов и самое частое слово.

import re
from collections import Counter


def analyze_text(text):
    pattern_punct = r'[!@"“’«»#$%&\'()*+,.—/:;<=>?^_`{|}~\[\]]'

    # Приводим текст к нижнему регистру, убираем пунктуацию и разбиваем его на слова
    words = re.sub(pattern_punct, ' ', text.lower().replace('--', ' ').strip()).split()

    # Подсчет общего количества слов
    total_words = len(words)

    # Подсчет уникальных слов
    unique_words = set(words)
    total_unique_words = len(unique_words)

    # Подсчет частоты слов
    word_counts = Counter(words)

    # Находим самое частое слово
    # Если есть несколько, будем брать первое по порядку
    most_common_word, most_common_count = word_counts.most_common(1)[0]

    return total_words, total_unique_words, most_common_word, most_common_count


import_this = """The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""

# Ввод текста пользователем
# user_input = input("Введите текст для анализа: ")

# Анализируем текст
words, unique_words, common_word, common_count = analyze_text(import_this)

# Вывод результатов
print(f"Общее количество слов: {words}")
print(f"Количество уникальных слов: {unique_words}")
print(f"Самое частое слово: '{common_word}' (встречается {common_count} раз(а))")
