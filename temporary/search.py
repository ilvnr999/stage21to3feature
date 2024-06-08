import numpy as np
import pandas as pd

data1 = pd.read_excel('5-1excels/name.xlsx')
data1_name = data1['name']
data2 = pd.read_excel('5-1excels/all_symbiodinium.xlsx', sheet_name = 'color')
data2_name = data2['照片名稱']
data2_color = data2['對應色卡顏色']
df = np.column_stack((data2_name,data2_color))
df = list(df)
for name1 in data1_name:
    for name2, color2  in df:
        if name1 == name2:
            print(color2)