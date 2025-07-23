import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns
df=pd.read_csv('C:/portfolio/code/week3/day3/gym_member.csv')

#1. 
#OLS : ordinary least squares '제곱오차'를 가장 작게 만드는 방식 / 목적 : 해석 / 어떤 변수가 통계적으로 유의미하냐
#LinearRegression / 목적 : 예측 / 모델이 얼마나 잘 맞추느냐가 중요하다
x=df[['운동빈도', '보충제코드', '나이', '성별코드', '월구독료']]
x=sm.add_constant(x) #이 식이 없다면 회귀식은 원점을 지나게 된다. 
y=df['건강점수']
model=sm.OLS(y,x).fit()
print(model.summary())
#성별코드의 P>|t| : 0.9, 의미 없는 변수

#2.
#vif : variance inflation factor(분산 팽창 지수) - 다중공선성(회귀모델에서 독립변수끼리의 상관관계)이 얼마나 심한지 알려주는 수치
#vif = 1 : 독립
#vif = 1~5 : 약간의 상관
#vif = 5~10 : 중복 영향이 있음
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['변수']=x.columns
vif['VIF']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
print(vif)
#대부분의 변수가 vif값이 1정도. 서로 독립됨

#3. 잔차 분석(Residuals)
# 잔차 = 실제값 - 예측값
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus']=False
residuals=model.resid

def residual_graph(df):
    plt.figure(figsize=(18,5))

    plt.subplot(1,3,1)
    sm.qqplot(residuals, line='45', fit=True, ax=plt.gca())
    plt.title('잔차 Q-Q plot')

    plt.subplot(1,3,2)
    sns.histplot(residuals, kde=True)
    plt.title('잔차 분포')

    plt.subplot(1,3,3)
    sns.residplot(x=model.fittedvalues, y=residuals, lowess=True)
    plt.xlabel('예측값')
    plt.ylabel('잔차')
    plt.title('잔차 vs 예측값')

    plt.tight_layout()
    plt.show()
residual_graph(df)
#q-q plot해석
# 점들이 빨간선에 붙어있음->잔차가 정규분포 한다. 때문에, p-value 해석이 가능

#히스토그램 해석
# 종 모양 그래프. 잔차 분포가 정규성 만족한다

#잔차 vs 예측값
#잔차의 등분산성 만족
#잔차에 특정한 추세나 비선형성이 없다. 모델이 데이터 전반에 걸쳐 일관되게 작동한다.

model.summary()