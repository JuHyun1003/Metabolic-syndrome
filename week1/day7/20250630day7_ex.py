import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
data = {
    '이름': ['지훈', '민서', '서준', '예은', '도윤', '수아', '현우', '하은'],
    '성별': ['남', '여', '남', '여', '남', '여', '남', '여'],
    '키(cm)': [173, 160, np.nan, 158, 180, 155, np.nan, 165],
    '몸무게(kg)': [70, 55, 80, 48, 85, 45, np.nan, np.nan],
    '운동횟수': [3, 0, 5, 1, 7, 0, 2, 0]
}
df = pd.DataFrame(data)
df=df.set_index('이름')
# 1. 결측치 파악 및 처리
df.isnull().sum()
df=df.set_index('이름')
print(df['키(cm)'].isnull())
df['키(cm)']=df['키(cm)'].interpolate()
df['몸무게(kg)']=df['몸무게(kg)'].interpolate()
df

# 2. 파생변수 만들기 – apply 쓰기
df['bmi']=df.apply(lambda row: row['몸무게(kg)'] / ((row['키(cm)'] / 100) ** 2), axis=1)
def grade(bmi):
    if bmi>=30:
        return '비만'
    elif bmi >= 25:
        return '과체중'
    elif bmi >= 18.5:
        return '정상'
    else:
        return '저체중'
df['비만도']=df['bmi'].apply(grade)
df
# 3. value_counts로 통계 뽑기(비만도 비율 제시)
print(df['비만도'].value_counts(normalize=True))

# 4. 시각화
# 운동횟수 기준 히스토그램
df['운동횟수'].plot(kind='hist')
plt.show()
plt.savefig('ex1.png')
# 비만도별 인원수 바 차트
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  
df['비만도'].value_counts().plot(kind='bar')
plt.show()
plt.savefig('ex2.png')