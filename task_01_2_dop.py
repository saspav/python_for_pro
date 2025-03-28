# Задание 2: Дешифровщик Цезаря
# Описание:
# Напишите функцию, которая реализует декодирование строки, зашифрованной шифром Цезаря.
# In:
# зашифрованная строка
# Out:
# список кортежей (сдвиг, результат дешифровки)
# Ограничения: метод дешифрования - brute force =)

import re
from nltk.corpus import wordnet


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


input_text = """1 zcysrgdsj gq zcrrcp rfyl sejw.
2 ibtpmgmx mw fixxiv xler mqtpmgmx.
3 rhlokd hr adssdq sgzm bnlokdw.
4 tfdgcvo zj svkkvi kyre tfdgcztrkvu.
5 xdsl ak twllwj lzsf fwklwv.
6 wtevwi mw fixxiv xler hirwi.
7 cplolmtwtej nzfyed.
8 qncagyj ayqcq ypcl'r qncagyj clmsef rm zpcyi rfc psjcq.
9 itbpwcop xzikbqkitqbg jmiba xczqbg.
10 uhhehi ixekbt duluh fqii iybudjbo.
11 sljcqq cvnjgagrjw qgjclacb.
12 sx dro pkmo yp kwlsqesdi, bopeco dro dowzdkdsyx dy qeocc.
13 lzwjw kzgmdv tw gfw-- sfv hjwxwjstdq gfdq gfw --gtnagmk osq lg vg al.
14 grznuamn zngz cge sge tuz hk uhbouay gz loxyz atrkyy eua'xk jazin.
15 dem yi rujjuh jxqd duluh.
16 fqymtzlm sjajw nx tkyjs gjyyjw ymfs *wnlmy* stb.
17 fc qeb fjmibjbkqxqflk fp exoa ql bumixfk, fq'p x yxa fabx.
18 xu iwt xbeatbtcipixdc xh tphn id tmeapxc, xi bpn qt p vdds xstp.
19 zmyqebmoqe mdq azq tazwuzs sdqmf upqm — xqf'e pa yadq ar ftaeq!"""

pattern_punct = r'[!@"“’«»#$%&\'()*+,.—/:;<=>?^_`{|}~\[\]]'

caesar_en = CaesarCipher(abc='ABCDEFGHIJKLMNOPQRSTUVWXYZ')

for row in input_text.splitlines():
    for shift in range(len(caesar_en.abc)):
        decoded = caesar_en.decode(row, shift)
        _decoded = re.sub(pattern_punct, ' ', decoded.lower()).strip()
        scores = [wordnet.synsets(word) for word in _decoded.split()]
        if sum(map(bool, scores)) / len(scores) > 0.6:
            print(f'shift={shift:>2}, decoded: {decoded}')
