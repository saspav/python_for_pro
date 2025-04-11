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

# забираете данные с листа easy
# вычисляете пациентов, у которых не в порядке хотя бы один анализ
# сохраняете в xlsx имя, телефон, название анализа и заключение: 'Повышен' или 'Понижен'

data = pd.read_excel('medicine.xlsx', sheet_name='easy', converters={'Анализ': str})

# Объединяем датафреймы
data = (data
        .merge(med_name, on='Код пациента', how='left')
        .merge(med_an_name, on='Анализ', how='left')
        .sort_values(['ФИО', 'Анализ'])
        .reset_index(drop=True)
        )
# Строим новую колонку с отклонениями
data['outlet'] = data.apply(lambda row: (row['Значение'] > row.max_value) -
                                        (row['Значение'] < row.min_value), axis=1)
# Фильтруем строки с отклонениями
easy = data[data['outlet'] != 0].copy()
# Формируем Заключение в нужном формате
easy['Заключение'] = easy['outlet'].map({-1: 'Понижен', 1: 'Повышен'})
# Сохраняем в файл
save_columns = ['ФИО', 'Телефон', 'Название анализа', 'Заключение']
easy[save_columns].to_excel('easy.xlsx', index=False, sheet_name='Результат')

print(easy[save_columns])
