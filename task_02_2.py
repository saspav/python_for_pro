# Задание:
# Найдите общий объем продаж за год.
# Найдите средний объем продаж в месяц.
# Найдите месяц с наибольшим и наименьшим объемом продаж.
# Постройте линейный график изменения объема продаж по месяцам (Matplotlib).

import pandas as pd
import matplotlib.pyplot as plt

sales_data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
              'September', 'October', 'November', 'December'],
    'Sales': [15000, 17000, 16000, 18000, 14000, 19000, 22000, 21000, 23000, 25000, 24000,
              26000]
}
# Сделаем из словаря датафрейм
df = pd.DataFrame(sales_data)

print('Общий объем продаж за год:', df['Sales'].sum())
print('Cредний объем продаж в месяц:', df['Sales'].mean())
print('Месяц с наибольшим объемом продаж:', df.loc[df['Sales'].argmax(), 'Month'])
print('Месяц с наименьшим объемом продаж:', df.loc[df['Sales'].argmin(), 'Month'])

# Создание графика
plt.figure(figsize=(10, 6))  # Установка размера графика
plt.plot(df['Month'], df['Sales'], marker='o', color='b', linestyle='-', linewidth=2)

# Настройка заголовка и подписей осей
plt.title('График объема продаж', fontsize=16)
plt.xlabel('Месяц', fontsize=12)
plt.ylabel('Продажи', fontsize=12)

# Поворот подписей месяцев для лучшей читаемости
plt.xticks(rotation=45)

# Добавление сетки
plt.grid(True, linestyle='--', alpha=0.7)

# Отображение графика
plt.tight_layout()  # Автоматическая подгонка layout
plt.show()
