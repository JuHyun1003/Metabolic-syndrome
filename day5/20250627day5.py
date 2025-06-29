import pandas as pd
import numpy as np
dcell = pd.read_excel(r'C:\Users\kgh44\Desktop\경희대\2025년(2학년)\2-1\세포생물학1\중간고사.xlsx')
print(dcell.head())

#이건 출력 안됨.
#iloc[]은 정수기반이라 함.
print(dcell.iloc[2, '이름'])

print(dcell.iloc[2])

dcell.iloc[1:3,[0,2]]
dcell.loc[dcell['총합'] >= 140, ['이름','총합']]

dcell1 = dcell
n=len(dcell1)
aplus=int(n*0.15)
azero=int(n*0.30)
aminus=int(n*0.45)
dcell1['학점']=np.where(dcell1.index < aplus,'A+',
                      np.where(dcell1.index < azero, 'A0',
                               np.where(dcell1.index < aminus, 'A-', 'B')))

dcell1.info()
dcell1.describe()
dcell1.groupby('학점').mean()
print(dcell1.dtypes)
dcell1.groupby('학점')[['중간고사','기말고사','총합']].mean()
dcell1['학점'].value_counts()
dcell1.isnull().sum()
dcell1[dcell1['기말고사'].isnull()]