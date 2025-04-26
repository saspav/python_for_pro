import numpy as np
import pandas as pd
import polars as pl

from bisect import bisect_left

__import__("warnings").filterwarnings('ignore')

rus = 'АЕКМНОРСТУХВ'
eng = 'AEKMHOPCTYXB'
MAPPING = {}
for ru, en in zip(rus, eng):
    MAPPING[ru] = f'[{ru}{en}]'
    MAPPING[en] = f'[{ru}{en}]'


def mapping_symbols(text):
    return ''.join(MAPPING.get(x.upper(), x) for x in text)


def clean_column_name(col_name):
    col_name = col_name.lower()  # преобразование в нижний регистр
    col_name = col_name.replace('(', '_')  # замена скобок на _
    col_name = col_name.replace(')', '')  # удаление закрывающих скобок
    col_name = col_name.replace('.', '_')  # замена точек на _
    # name = re.sub(r"(?<=\d)_(?=\d)", "", name)  # удаление подчеркивания между числами
    return col_name


def df_to_excel(save_df, file_excel, ins_col_width=None, float_cells=None):
    """ экспорт в эксель """
    writer = pd.ExcelWriter(file_excel, engine='xlsxwriter')
    # save_df.to_excel(file_writer, sheet_name='find_RFC', index=False)
    # Convert the dataframe to an XlsxWriter Excel object.
    # Note that we turn off the default header and skip one row to allow us
    # to insert a user defined header.
    save_df.to_excel(writer, sheet_name='logs', startrow=1, header=False, index=False)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['logs']
    # Add a header format.
    header_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 13,
        'bold': True,
        'text_wrap': True,
        'align': 'center',
        'valign': 'center',
        'border': 1})
    # Write the column headers with the defined format.
    worksheet.freeze_panes(1, 0)
    for col_num, value in enumerate(save_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    # вставка ссылок
    url_format = workbook.add_format({
        'font_color': 'blue',
        'underline': 1,
        'font_name': 'Times New Roman',
        'font_size': 13
    })
    cell_format = workbook.add_format()
    cell_format.set_font_name('Times New Roman')
    cell_format.set_font_size(13)
    float_format = workbook.add_format({'num_format': '0.000000'})
    float_format.set_font_name('Times New Roman')
    float_format.set_font_size(13)
    col_width = [7, 7, 10, 32, 8, 12, 12] + [32] * 9
    if ins_col_width:
        for num_pos, width in ins_col_width:
            col_width.insert(num_pos, width)
    for num, width in enumerate(col_width[:len(save_df.columns)]):
        now_cell_format = cell_format
        if float_cells and num in float_cells:
            now_cell_format = float_format
        worksheet.set_column(num, num, width, now_cell_format)
    worksheet.autofilter(0, 0, len(save_df) - 1, len(save_df.columns) - 1)
    writer.close()


def age_bucket(x):
    return bisect_left([18, 25, 35, 45, 55, 65], x)


def memory_compression_pl(df: pl.DataFrame, use_category=True, use_float=True,
                          exclude_columns=None):
    """
    Оптимизация типов данных в Polars DataFrame для уменьшения потребления памяти.

    :param df: Polars DataFrame
    :param use_category: Преобразовывать строки в категорию
    :param use_float: Преобразовывать float в пониженную разрядность
    :param exclude_columns: Список колонок, которые исключаются из обработки
    :return: Оптимизированный Polars DataFrame
    """
    exclude_columns = set(exclude_columns) if exclude_columns else set()

    start_mem = df.estimated_size() / 1024 ** 2  # Размер в MB
    schema = df.schema  # Текущие типы колонок

    new_columns = []

    for col, dtype in schema.items():
        if col in exclude_columns:
            new_columns.append(df[col])
            continue

        if isinstance(dtype, (pl.Date, pl.Datetime)):
            new_columns.append(df[col])  # Даты не трогаем
            continue

        if isinstance(dtype, (pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt16, pl.UInt32)):
            col_min, col_max = df[col].min(), df[col].max()
            if col_min is not None and col_max is not None:
                if col_min >= 0:  # Проверяем на беззнаковость
                    if col_max < np.iinfo(np.uint8).max:
                        new_columns.append(df[col].cast(pl.UInt8))
                    elif col_max < np.iinfo(np.uint16).max:
                        new_columns.append(df[col].cast(pl.UInt16))
                    elif col_max < np.iinfo(np.uint32).max:
                        new_columns.append(df[col].cast(pl.UInt32))
                    else:
                        new_columns.append(df[col])  # Оставляем Int64
                else:  # Если есть отрицательные числа, остаёмся в знаковых int
                    if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                        new_columns.append(df[col].cast(pl.Int8))
                    elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(
                            np.int16).max:
                        new_columns.append(df[col].cast(pl.Int16))
                    elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(
                            np.int32).max:
                        new_columns.append(df[col].cast(pl.Int32))
                    else:
                        new_columns.append(df[col])  # Оставляем Int64
            else:
                new_columns.append(df[col])

        elif use_float and isinstance(dtype, (pl.Float64, pl.Float32)):
            col_min, col_max = df[col].min(), df[col].max()
            if col_min is not None and col_max is not None:
                if col_min > np.finfo(np.float32).min and col_max < np.finfo(
                        np.float32).max:
                    new_columns.append(df[col].cast(pl.Float32))
                else:
                    new_columns.append(df[col])  # Оставляем Float64
            else:
                new_columns.append(df[col])

        elif use_category and isinstance(dtype, pl.Utf8):
            new_columns.append(df[col].cast(pl.Categorical))

        else:
            new_columns.append(df[col])  # Оставляем без изменений

    # Создаём новый DataFrame с оптимизированными типами
    df_compressed = pl.DataFrame(new_columns)

    end_mem = df_compressed.estimated_size() / 1024 ** 2
    print(f'Исходный размер: {round(start_mem, 2)} MB')
    print(f'Оптимизированный размер: {round(end_mem, 2)} MB')
    print(f'Экономия памяти: {(1 - end_mem / start_mem):.1%}')

    return df_compressed


def memory_compression(df, use_category=True, use_float=True, exclude_columns=None):
    """
    Изменение типов данных для экономии памяти
    :param df: исходный ДФ
    :param use_category: преобразовывать строки в категорию
    :param use_float: преобразовывать float в пониженную размерность
    :param exclude_columns: список колонок, которые нужно исключить из обработки
    :return: сжатый ДФ
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        # print(f'{col} тип: {tmp[col].dtype}', str(tmp[col].dtype)[:4])

        if exclude_columns and col in exclude_columns:
            continue

        if str(df[col].dtype)[:4] in 'datetime':
            continue

        elif str(df[col].dtype) not in ('object', 'category'):
            col_min = df[col].min()
            col_max = df[col].max()
            if str(df[col].dtype)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif use_float and str(df[col].dtype)[:5] == 'float':
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif (col_min > np.finfo(np.float32).min
                      and col_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        elif use_category and str(df[col].dtype) == 'object':
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Исходный размер датасета в памяти '
          f'равен {round(start_mem, 2)} мб.')
    print(f'Конечный размер датасета в памяти '
          f'равен {round(end_mem, 2)} мб.')
    print(f'Экономия памяти = {(1 - end_mem / start_mem):.1%}')
    return df


if __name__ == "__main__":
    pass
