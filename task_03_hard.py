import pandas as pd
import psycopg2


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

# # Распечатать сведения о PostgreSQL
# print("Информация о сервере PostgreSQL")
# print(connection.get_dsn_parameters(), "\n")

sql_med_name = 'select n.id "Код пациента", n.name "ФИО", n.phone "Телефон" from med_name n'
sql_med_an_name = ('select m.id "Анализ", m.name "Название анализа", m.is_simple, '
                   'm.min_value, m.max_value from med_an_name m')

med_name = get_sql(cursor, sql_med_name)
med_an_name = get_sql(cursor, sql_med_an_name)

# после выполнения всех действий закрываем соединение с базой
cursor.close()
connection.close()

# забираете данные с листа hard
# вычисляете пациентов, у которых не в порядке два и более анализов
# сохраняете в xlsx имя, телефон, название анализа и заключение:
# 'Повышен', 'Понижен' или 'Положительный'

data = pd.read_excel('medicine.xlsx', sheet_name='hard', converters={'Анализ': str})

# Объединяем датафреймы
data = (data
        .merge(med_name, on='Код пациента', how='left')
        .merge(med_an_name, on='Анализ', how='left')
        .sort_values(['ФИО', 'Анализ'])
        .reset_index(drop=True)
        )
# Приводим положительные значения бинарных результатов к единым значениям
data['Значение'] = data['Значение'].map(lambda z: 'Положит.' if z == '+' else z)

# Строим новую колонку с численными значениями: строки будут NaN
data['value'] = pd.to_numeric(data['Значение'], errors='coerce', downcast='float')

# Строим новую колонку с отклонениями
data['outlet'] = data.apply(
    lambda row: (row['value'] > row.max_value) - (row['value'] < row.min_value), axis=1)

# Заполняем отклонения для бинарных результатов с положительным значением
data.loc[data['Значение'].astype(str).str.contains('Полож'), 'outlet'] = 2

# Фильтруем строки с отклонениями
hard = data[data['outlet'] != 0].copy()

# Выбираем ФИО, у которых есть отклонения в двух и более анализах
fio_outlet = hard['ФИО'].value_counts()
fio_outlet = fio_outlet[fio_outlet > 1].index.tolist()

# Фильтруем строки с найденными ФИО
hard = hard[hard['ФИО'].isin(fio_outlet)].copy()

# Формируем Заключение в нужном формате
hard['Заключение'] = hard['outlet'].map({-1: 'Понижен', 1: 'Повышен', 2: 'Положительный'})

# Сохраняем в файл
save_columns = ['ФИО', 'Телефон', 'Название анализа', 'Заключение']
hard[save_columns].to_excel('hard.xlsx', index=False, sheet_name='Результат')

print(hard[save_columns])
