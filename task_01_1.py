# Задание 1: Кодировщик Цезаря
# Описание:
# Напишите функцию, которая реализует кодирование строки шифром Цезаря.
# In:
# строка
# размер сдвига
# Out:
# зашифрованная строка
# Ограничения: строка содержит только буквы русского алфавита


class CaesarCipher:
    def __init__(self, abc=None, shift=7):
        self.shift = shift
        self.abc = abc if abc else 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'

    def _caesar(self, text, shift=None, key=1):
        """
        Метод, реализующий алгоритм Цезаря
        :param text: текст
        :param shift: сдвиг
        :param key: направление
        :return: зашифрованный/дешифрованный текст
        """
        shift = self.shift if shift is None else shift
        encoded = ''
        for ch in text:
            if ch.upper() in self.abc:
                x = self.abc[(self.abc.find(ch.upper()) + shift * key) % len(self.abc)]
                encoded += x.lower() if ch.islower() else x
            else:
                encoded += ch
        return encoded

    def encode(self, text, shift=None):
        """
        Функция шифрования текста
        :param text: текст
        :param shift: сдвиг
        :return: зашифрованный текст
        """
        shift = self.shift if shift is None else shift
        return self._caesar(text, shift)

    def decode(self, text, shift=None):
        """
        Функция дешифровки текста
        :param text: текст
        :param shift: сдвиг
        :return: дешифрованный текст
        """
        shift = self.shift if shift is None else shift
        return self._caesar(text, shift, -1)

    def brute_force(self, text):
        """
        Метод для взлома шифра Цезаря
        :param text: зашифрованный текст
        :return: список с кортежами вида: (сдвиг, расшифрованный текст)
        """
        result = []
        for shift in range(len(self.abc)):
            result.append((shift, self.decode(text, shift)))
        return result


letters = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й',
           'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У',
           'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э',
           'Ю', 'Я']

# Входная строка
input_text = 'Напишите функцию, которая реализует кодирование строки шифром Цезаря.'
# Сдвиг для шифра
shift_text = 13

# Оставим только буквы из списка letters
input_text = ''.join(ch for ch in input_text.upper() if ch in letters)
print('Исходный текст:', input_text)

caesar = CaesarCipher()

encoded_text = caesar.encode(input_text, shift_text)
print('\nЗашифрованный текст:', encoded_text)
