import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
df=pd.read_csv('C:/portfolio/code/week2/day7/gym_members.csv')
df
df.select_dtypes(include='number').columns
df['나이'].mean()
df['운동경력'].median()
df['건강점수'].agg(['mean','min','max','std'])
df['성별'].mode().values[0]
df['성별'].value_counts()


matplotlib.rc('font', family = "Malgun Gothic")
sns.histplot(df['건강점수'], kde=True)
plt.show()

#첨도
# 꼬리부분의 길이와 중앙부분의 뾰족함에 대한 정보
# 0보다 큰 경우 : 긴 꼬리, 중앙부분이 뾰족함(분포가 중앙에 덜 집중)
# 0보다 작은 경우 : 짧은 꼬리, 중앙부분이 완만함(분포가 중앙에 더 집중)
df['나이'].kurt()

#왜도
# 분포가 한쪽으로 삐뚤어졌는지 보는 지표
# 왼쪽으로 긴 꼬리를 가진 경우 : -
# 오른쪽으로 긴 꼬리를 가진 경우 : +
# 비대칭도가 커질수록 왜도의 절댓값이 커진다
df['나이'].skew()

def healthscore(df):
    sns.histplot(df['건강점수'], kde=True)
    plt.axvline(df['건강점수'].mean(), color='r', linestyle = '--', label='평균')
    plt.title('건강점수 분포')
    plt.legend()
    plt.show()


healthscore(df)
#예측
#1. 오른쪽으로 긴 꼬리이니 + 값의 왜도 skew
#2. 중앙부가 완만하니 -값의 첨도  kurt
df['건강점수'].agg(['skew','kurt'])


sns.barplot(data=df, x='성별', y='근육량', errorbar='sd')
plt.title('성별에 따른 평균 근육량')
plt.show()
plt.savefig('성별에따른평균근육량.png')

df['월구독료'].corr(df['건강점수'])
sns.scatterplot(data=df, x='월구독료', y='건강점수')
plt.title('월구독료 vs 건강점수')
plt.show()
plt.savefig('월구독료건강점수비교.png')


# 이상치 탐색 함수
def out_liers(a):
    Q1 = a.quantile(0.25)
    Q3 = a.quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(a < Q1 - 1.5*IQR) | (a > Q3 + 1.5*IQR)]
    print(outliers)

out_liers(df['주당운동일수'])


#체지방률이 높은 사람들만 선택해서 근육량 평균 비교
조건 = df['인바디체지방률']>df['인바디체지방률'].median()
평균 = df[조건].groupby('성별')['근육량'].mean()
print(평균)