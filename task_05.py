import re
import pandas as pd

from df_addons import df_to_excel

CONFINES = 0.7  # Порог для распознавания русского текста
EN_LETTER = re.compile(r'[a-z/<>\n\t]', flags=re.IGNORECASE)  # английские буквы
RUS_DIGITS = re.compile(r'[а-яё0-9/., -]', flags=re.IGNORECASE)  # русские буквы и цифры
NOT_PATTERN = re.compile(r'[^а-яё]', flags=re.IGNORECASE)  # кроме русских букв
BYTE_PATTERN = re.compile(r'[a-wyzG-Z .,/<>-]')  # английские буквы, которых нет в байт-строке


def decode_unknown(text):
    """
    Функция перекодировки из кразозябр в читаемый русский текст
    :param text: текст
    :return: перекодированный текст
    """
    # Если в строке более порога русских букв и цифр - то текст уже годный к использованию
    if len(RUS_DIGITS.findall(text)) > len(text) * CONFINES:
        return text

    # Список кодировок для перебора
    encodings = ['UTF-8', 'KOI8-R', 'CP866', 'WINDOWS-1251', 'WINDOWS-1252', ]
    for encoding in encodings:
        try:
            if any(0 <= ord(c) < 255 for c in BYTE_PATTERN.sub('', text)):
                # Если есть хоть один символ от байт-кодов -->
                # Восстанавливаем байт-строку двойной перекодировкой
                text_ = text.encode('latin1').decode('unicode_escape')
                decoded = text_.encode('latin1').decode(encoding)
            else:
                # Обычную строку с кракозабрами пытаемся закодировать и раскодировать в UTF-8
                decoded = text.encode(encoding, errors='ignore').decode('UTF-8')

            # В некоторых кодировках преобладают буквы п, я - уберем их
            temp = EN_LETTER.sub('', decoded).replace('п', '').replace('я', '')
            #  Если в строке русских букв и цифр больше порога - текст годный к использованию
            if len(RUS_DIGITS.findall(temp)) > len(temp) * CONFINES:
                return decoded
        except:
            pass
    # Не смогли распознать кракозябры --> возвращаем как есть и удалим их потом
    return text


def main_func(file_name='durty_data.csv', pandas_read_csv=True, use_decode_text=True):
    """
    Чтение текстового файла, очистка от кавычек, декодирование кракозябр и сохранение в эксель
    :param file_name: Имя текстового файла
    :param pandas_read_csv: Использовать pandas для чтения файла.
    :param use_decode_text: Декодировать кракозябры или оставить как есть.
    :return: 2 датафрейма с найденными адресами и координатами
    """
    pd_suffix = '_pandas'  # Добавляется к имени файла-результата

    if pandas_read_csv:
        print(f'Читаем файл "{file_name}" методом pd.read_csv()')
        # Читаем файл пандасом и переименовываем колонку
        df = pd.read_csv(file_name, encoding='utf-8', header=None).rename(columns={0: 'text'})
    else:
        pd_suffix = ''
        print(f'Читаем файл "{file_name}" обычным методом')
        # Читаем файл обычным методом
        with open(file_name, encoding='utf-8') as file:
            rows = file.readlines()

        # Создаем ДФ
        df = pd.DataFrame(data=rows, columns=['text'])

    df['text_old'] = df['text']  # Для отладки оставим оригинальный текст

    print('len(df):', len(df))

    # Замена "\\" на "\"
    df['text'] = df['text'].str.replace('\\\\', '\\')

    if use_decode_text:
        pd_suffix = f'{pd_suffix}_decoded'  # Добавим еще один суффикс к имени файла
        # Перекодируем кракозябры и удалим пробелы и слэши с концов строки в цикле,
        # т.к. есть двойные перекодировки
        for _ in range(2):
            df['text'] = df['text'].map(decode_unknown).str.strip().str.strip('\\')

    # Очистим строку от кавычек и лишних пробелов
    df['text'] = df['text'].str.strip().str.strip('"').str.strip()

    # Удаляем дубликаты
    df.drop_duplicates(inplace=True)

    print('Уникальных строк в df:', len(df))

    df['len_str'] = df['text'].map(len)  # длина строки для отладки

    # Создаем маску для координат: цифры точка цифры разделитель цифры точка цифры
    coord_mask = df['text'].str.contains(r'^\d+\.\d*[, ]{1,3}\d+\.\d*$', regex=True)
    # Фильтруем координаты
    df_coords = df[coord_mask]
    # Фильтруем адреса
    df_adress = df[~coord_mask]

    # Создаем маску, что в адресе есть русские буквы в количестве не менее 3
    rus_mask = df_adress['text'].map(lambda z: len(NOT_PATTERN.sub('', z)) >= 3
                                     # and '</search>' not in z  # это раскоментарить,
                                     # если нужно выкинуть адреса с тегами
                                     )
    print(f'Не распознано адресов: {len(df_adress) - sum(rus_mask)} из {len(df_adress)}')

    df_adress['russian'] = rus_mask
    df_to_excel(df_coords, f'df_coords{pd_suffix}.xlsx')
    df_to_excel(df_adress, f'df_adress{pd_suffix}.xlsx')

    # Оставляем только нужные строки с адресами
    df_adress = df_adress[rus_mask]

    print('Уникальных адресов:   ', len(df_adress))
    print('Уникальных координат: ', len(df_coords))

    # Запись в Excel на разные листы выбранных колонок из списка без заголовков
    # cols = df_coords.columns
    cols = ['text']
    with pd.ExcelWriter(f'result{pd_suffix}.xlsx', engine='xlsxwriter') as writer:
        df_adress[cols].to_excel(writer, sheet_name='Адреса', index=False, header=None)
        df_coords[cols].to_excel(writer, sheet_name='Координаты', index=False, header=None)

    return df_adress, df_coords


if __name__ == "__main__":
    # file_csv = 'sample.csv' # Имя файла для опытов
    file_csv = 'durty_data.csv'  # Имя файла
    use_pandas = True  # Использовать pandas для чтения файла

    main_func(file_name=file_csv, pandas_read_csv=use_pandas, use_decode_text=True)

# Читаем файл "durty_data.csv" методом pd.read_csv()
# len(df): 207209
# Уникальных строк в df: 87452
# Не распознано адресов: 13 из 25078
# Уникальных адресов:    25065
# Уникальных координат:  62374
#
# Читаем файл "durty_data.csv" обычным методом
# len(df): 207251
# Уникальных строк в df: 87464
# Не распознано адресов: 17 из 25090
# Уникальных адресов:    25073
# Уникальных координат:  62374
#
# Читаем файл "durty_data.csv" методом pd.read_csv() - без декодирования текста
# len(df): 207209
# Уникальных строк в df: 87452
# Не распознано адресов: 15965 из 25078
# Уникальных адресов:    9113
# Уникальных координат:  62374
