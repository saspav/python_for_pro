import re
import pandas as pd
import psycopg2


def recognizing(text):
    """
    Функция выделяет числа из строки и склеивает их в нужном формате
    :param text: текст
    :return: результат
    """
    nums = re.findall('\d+', text)  # Находим числа
    result = []
    for i in range(len(nums) // 5):  # Цикл для обработки более одного договора в строке
        *dog, d, m, y = nums[i * 5: i * 5 + 5]
        # Форматируем дату
        formatted_date = f"{d.zfill(2)}.{m.zfill(2)}.{'20' * (len(y) == 2) + y}"
        # Формируем результат в нужном формате
        result.append(f"{'/'.join(dog)} от {formatted_date}")
    return ', '.join(result)


def get_sql(ps_cursor, sql_text):
    # Выполнение SQL-запроса
    ps_cursor.execute(sql_text)
    # получение списка кортежей результата селекта
    results = ps_cursor.fetchall()
    col_names = [row[0] for row in ps_cursor.description]
    print(f'По запросу: "{sql_text}" получено {ps_cursor.rowcount} записей\n')

    # полученные данные преобразуем в датафрейм
    df = pd.DataFrame(data=results, columns=col_names)

    return df.fillna(0)


# Подключение к базе данных
connection = psycopg2.connect(
    database="postgres",
    host="localhost",
    user="postgres",
    password="postgres",
    port="5433"
)

# Курсор для выполнения операций с базой данных
cursor = connection.cursor()

sql_payments = "SELECT *FROM payments;"

df = get_sql(cursor, sql_payments)

# после выполнения всех действий закрываем соединение с базой
cursor.close()
connection.close()

# Удаляем сумму из строки
df['temp'] = df.apply(lambda row: row['reason'].replace(str(int(row['amt'])), '', 1), axis=1)
# Шаблон для даты чтобы, отделить пробелом слипшиеся номер договора и дату
# (?:[0-3]?d[./ _]{1,2}) - поиск дня и месяца: 2 таких группы
# (?:\d\d$|\d{4}) - две цифры года в конце строки или 4 цифры года в любом месте строки
date_pattern = r'((?:[0-3]?\d[.\\/ _]{1,2}){2}(?:\d\d$|\d{4}))'
df['temp'] = df['temp'].str.replace(date_pattern, r' \1', regex=True)
# Формирование результата
df['result'] = df['temp'].map(recognizing)
# Сравнение
df['compare'] = df['result'] == df['reason_correct']

print(f"Результат: {df['compare'].mean():.3f}")

# Удаляем временные колонки
df.drop(columns=['temp'], errors='ignore', inplace=True)

# Сохраним для анализа
df.to_excel('payments_new.xlsx', index=False)
