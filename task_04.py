import re
import pandas as pd
import psycopg2


def format_date(date_str):
    """
    Функция для преобразования даты
    :param date_str: строка с датой
    :return: отформатированная дата DD.MM.YYYY
    """
    # Находим даты с помощью re.findall
    day, month, year = re.findall(r'\d+', date_str)

    return f"{day.zfill(2)}.{month.zfill(2)}.{'20' * (len(year) == 2) + year}"


def recognizing(row):
    """
    Функция распознает договора в изначально записанном верном формате
    :param row: строка ДФ
    :return: результат
    """
    reason = row['reason']

    # Находим даты
    matches = re.findall(r'(\d+)\.(\d+)\.(\d{4})\b', reason)

    for match in matches:
        day, month, year = match  # Разбиваем на группы
        # Форматируем дату
        formatted_date = f"{day.zfill(2)}.{month.zfill(2)}.{year}"
        # Заменяем старую дату на новую в строке
        reason = reason.replace(".".join(match), formatted_date)

    # Приводим номер договора к нужному формату
    reason = re.sub(r'(\d{4,})[ \\_/-]{1,3}(\d{2})', r'\1/\2', reason)
    # Ищем соответствие шаблону
    pattern = r'\d{4,}/\d{2} от (?:\d\d\.){2}\d{4}'
    reason = ', '.join(re.findall(pattern, reason))
    return reason if reason else row['result']


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

sql_payments = r"""
SELECT 
    dt,
    amt,
    regexp_replace(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    reason,
                    E'\22', '/22'  -- \22 → ASCII 0x12 (SYN)
                ),
                E'\74', '/74'      -- \74 → ASCII 0x3C ('<')
            ),
            E'\2', '/2'           -- \2 → ASCII 0x02 (STX)
        ),
        '\\9', '/9'              
    ) AS reason,
    reason_correct
FROM payments;
"""

df = get_sql(cursor, sql_payments)

# после выполнения всех действий закрываем соединение с базой
cursor.close()
connection.close()

# Ищем формат “число(. )( .)число” и заменяем найденное выражение на “число.число”
df['temp'] = df['reason'].str.replace(r'(\d)(\.\s+|\s+\.)(\d)', r'\1.\3', regex=True)

# Шаблон для даты
ptn_data = r'([0-3]?\d[.\\/ _][0-3]?\d[.\\/ _]\d{2,4}$)'
# Ищем дату в конце строки и пишем в отдельную колонку
df['data'] = df['temp'].str.strip().str.extract(ptn_data)
# Удаляем дату из строки
df['temp'] = df['temp'].replace(ptn_data, '', regex=True).str.strip()
# Форматируем дату
df['data'] = df['data'].map(format_date)

# Шаблон для договора: 3 и более цифр, 1-3 разделителя, 1-2 цифры, после которых нет цифр
ptn_dog = r'(\d{3,}[ \\_/-]{1,3}\d{1,2})(?!\d)'
df['dog'] = df['temp'].str.extract(ptn_dog).fillna('')
# Добавим номер договора, если он пуст
df['dog'] = df.apply(
    lambda row: row['dog'] if row['dog'] else re.findall(r'\d+', row['temp'])[0], axis=1)
# Формируем договор в нужном формате
df['dog'] = df['dog'].map(lambda z: '/'.join(re.findall(r'\d+', z)) if z else '')
# Формируем ответ в нужном формате
df['result'] = df[['dog', 'data']].apply(lambda row: ' от '.join(row), axis=1)
# Распознавание договоров в изначально записанном верном формате
df['result'] = df.apply(lambda row: recognizing(row), axis=1)
# Сравнение
df['compare'] = df['result'] == df['reason_correct']

print(f"Результат: {df['compare'].mean():.3f}")

# Удаляем временные колонки
df.drop(columns=['temp', 'data', 'dog'], errors='ignore', inplace=True)

# Сохраним для анализа
df.to_excel('payments.xlsx', index=False)
