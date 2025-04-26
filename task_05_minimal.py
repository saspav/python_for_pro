import re
import pandas as pd

df = pd.read_csv('durty_data.csv', encoding='utf-8', header=None).rename(columns={0: 'text'})
df['text'] = df['text'].str.replace('\\\\', '\\').str.strip().str.strip('"').str.strip()
df.drop_duplicates(inplace=True)
coord_mask = df['text'].str.contains(r'^\d+\.\d*[, ]{1,3}\d+\.\d*$', regex=True)
df_coords = df[coord_mask]
df_adress = df[~coord_mask]
rus_mask = df_adress['text'].map(lambda z: len(re.sub(r'[^а-яё]', '', z, flags=re.I)) >= 3)
df_adress = df_adress[rus_mask]
with pd.ExcelWriter(f'result_.xlsx', engine='xlsxwriter') as writer:
    df_adress[['text']].to_excel(writer, sheet_name='Адреса', index=False, header=None)
    df_coords[['text']].to_excel(writer, sheet_name='Координаты', index=False, header=None)
