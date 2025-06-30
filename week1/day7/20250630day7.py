import pandas as pd
import numpy as np

data = {
    '이름': ['철수', '영희', '민수', '지민'],
    '키': [173, np.nan, 180, 165],
    '몸무게': [70, 55, np.nan, 48]
}
df = pd.DataFrame(data)

#1. 결측치 확인
# isnull()
df=df.set_index('이름')
print(df.isnull().sum())
print(df['키'].isnull())
print(df['몸무게'].isnull())

#2. 결측치 메꾸기
#fillna
#아래 방법은 결측치를 평균치로 메꾸는 법
df['키']=df['키'].fillna(df['키'].mean())
df['몸무게']=df['몸무게'].fillna(df['몸무게'].mean())
df

df.describe()

#3. 결측치 없애기
# dropna()
print(df.dropna())

#4. interpolate()
# 결측치를 주변 값들로 메꾸기
df1 = pd.DataFrame({
    '날짜': pd.date_range('2025-06-01', periods=5),
    '체중': [70, np.nan, np.nan, 76, 77]
})
df1
df1['체중']=df1['체중'].interpolate()