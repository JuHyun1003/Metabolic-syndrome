import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df=pd.read_csv('C:/portfolio/code/week3/day3/gym_member.csv')
x=df[['운동빈도', '보충제코드', '나이', '성별코드', '월구독료']]
y=df['건강점수']
model=LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)

a=df[['운동빈도', '보충제코드', '나이', '성별코드', '월구독료']]
a=sm.add_constant(a)
b=df['건강점수']
model2=sm.OLS(b,a).fit()
print(model2.summary())
b_pred = model2.predict(a)


def raw_predict_linear(df):
    mse=mean_squared_error(y, y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y,y_pred)
    print('RMSE :', rmse)
    print('R2 :', r2)

def raw_predict_ols(df):
    mse=mean_squared_error(b, b_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(b,b_pred)
    print('RMSE :', rmse)
    print('R2 :', r2)

raw_predict_linear(df)
raw_predict_ols(df)

vif=pd.DataFrame()
vif['변수']=a.columns
vif['vif']=[variance_inflation_factor(a.values, i) for i in range (a.shape[1])]
vif


model2.summary() #성별코드의 P>|t| : 0.9 따라서, 이 변수 제거

p=df[['운동빈도', '보충제코드', '나이', '월구독료']]
z=sm.add_constant(p)
q=df['건강점수']
model3=sm.OLS(q,z).fit()
print(model3.summary())
q_pred=model3.predict(z)

model_compare=pd.read_excel('C:/portfolio/code/week3/day5/model_compare.xlsx')
model_compare.set_index('Unnamed: 0', inplace=True)
model_compare.to_excel('C:/portfolio/code/week3/day5/model_compare.xlsx')
model_compare
#R-squared(결정계수) : 설명력. (ex. 건강점수의 78.3%는 이 입력변수들로 설명 가능)
#AIC : 모델의 복잡도와 성능을 동시에 따졌을 때, 얼마나 좋은 모델인지.
 # 작을수록 좋다.(성능 좋고 복잡도가 적음)
#BIC : AIC와 같은 개념이지만, 변수의 수에 더 강한 패널티를 준다.
 # 작을수록 좋다(간단한 모델인데 성능도 좋다는 것을 뜻함)

def new_predict_ols(df):
    mse=mean_squared_error(q, q_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(q,q_pred)
    print('RMSE :', rmse)
    print('R2 :', r2)
new_predict_ols(df)
raw_predict_ols(df)


#day20
#잔차 시각화
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus']=False

residuals=q-q_pred

def residual_visual(df):
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    sns.histplot(residuals, bins=20, kde=True)
    plt.title('잔차 분포 히스토그램')
    plt.xlabel('잔차')
    plt.ylabel('빈도')

    plt.subplot(1,3,2)
    sm.qqplot(residuals,fit=True, line='45', ax=plt.gca())
    plt.title('잔차 q-q plot')
    
    plt.subplot(1,3,3)
    plt.scatter(q_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('예측값 vs 잔차')
    plt.xlabel('예측값')
    plt.ylabel('잔차')

    plt.tight_layout()
    plt.show()
residual_visual(df)
