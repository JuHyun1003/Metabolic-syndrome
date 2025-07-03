import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# 한글 깨짐 방지
plt.rc('font', family='Malgun Gothic')

# 데이터 만들기
data = {
    '이름': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    '키': [170, 165, 180, 175, 160, 168, 172, 250], 
    '몸무게': [60, 55, 70, 65, 50, 58, 62, 90]
}
df = pd.DataFrame(data)
print(df.describe())

# IQR로 이상치 찾기
Q1 = df['키'].quantile(0.25)
Q3 = df['키'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

outliers_height = df[(df['키']<lower_bound) | (df['키'] > upper_bound)]
print(outliers_height)

# z score는 정규분포를 따를때 주로 사용. 표본수가 적거나 정규분포 따르지 않을땐 사용x
# 일반적으로 3보다 크면 이상치라고 판단
z_scores = stats.zscore(df[['키','몸무게']])
z_scores
outliers_z = df[(np.abs(z_scores)>3).any(axis=1)]
outliers_z
# 이 결과로는 이상치가 나오지 않음. 정규분포 하지 않고 표본수가 적다.

# 정규성 검정
from scipy.stats import shapiro
stat, p = shapiro(df['몸무게'])
print(p)

df_clean = df[~df.index.isin(outliers_height.index)]
df_clean

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# 한글 깨짐 방지
plt.rc('font', family='Malgun Gothic')

# 데이터 만들기
data = {
    '이름': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    '키': [170, 165, 180, 175, 160, 168, 172, 250], 
    '몸무게': [60, 55, 70, 65, 50, 58, 62, 90]
}
df = pd.DataFrame(data)
print(df.describe())

# IQR로 이상치 찾기
Q1 = df['키'].quantile(0.25)
Q3 = df['키'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

outliers_height = df[(df['키']<lower_bound) | (df['키'] > upper_bound)]
print(outliers_height)

# z score는 정규분포를 따를때 주로 사용. 표본수가 적거나 정규분포 따르지 않을땐 사용x
# 일반적으로 3보다 크면 이상치라고 판단
z_scores = stats.zscore(df[['키','몸무게']])
z_scores
outliers_z = df[(np.abs(z_scores)>3).any(axis=1)]
outliers_z
# 이 결과로는 이상치가 나오지 않음. 정규분포 하지 않고 표본수가 적다.

# 정규성 검정
from scipy.stats import shapiro
stat, p = shapiro(df['몸무게'])
print(p)

df_clean = df[~df.index.isin(outliers_height.index)]
df_clean

# 정규화
# 데이터 값의 범위를 재조정해서 모델이 특정 변수만 편애하지 않게 만드는 작업
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# MinMaxScaler : 0~1로 압축 | (x-min) / (max-min) - 이상치에 민감함
# StandardScaler : 평균 0, 표준편차 1로 정규화 | (x - 평균) / 표준편차 - 정규분포 아니면 사용 x
scaler_minmax=MinMaxScaler()
scaler_standard=StandardScaler()

df_minmax=df_clean.copy()
df_standard=df_clean.copy()

df_minmax[['키','몸무게']]=scaler_minmax.fit_transform(df_minmax[['키','몸무게']])
df_standard[['키','몸무게']]=scaler_standard.fit_transform(df_standard[['키','몸무게']])
print(df_minmax)
print(df_standard)

#정규화 후 시각화
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
sns.boxplot(data=df_clean[['키','몸무게']])
plt.title('정제 후 원본')

plt.subplot(1,3,2)
sns.boxplot(data=df_minmax[['키','몸무게']])
plt.title('minmax 정규화')

plt.subplot(1,3,3)
sns.boxplot(data=df_standard[['키','몸무게']])
plt.title('standard 정규화')

plt.tight_layout()
plt.show()
plt.savefig('ex1.png')
df_clean.describe()
df_minmax.describe()
df_standard.describe()
df.describe()