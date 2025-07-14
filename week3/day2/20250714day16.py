import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
df=pd.read_csv('C:/portfolio/code/week2/day7/gym_members.csv')
df=df.set_index('이름')

#linear regression을 통해 월구독료->건강점수 예측

#변수 설정
x=df[['월구독료']] #2차원 변수. 대괄호 2개
y=df['건강점수']

#모델 훈련
model=LinearRegression()
model.fit(x,y)

#계수 확인
print('기울기 :', model.coef_[0]) #기울기 : x가 1증가할 때, y가 얼마나 증가하냐
print('절편 :', model.intercept_)

#예측값 구하기
y_pred = model.predict(x)
df['예측건강점수']=y_pred
df

#시각화
def today(df):
    matplotlib.rc('font',family=('Malgun Gothic'))
    plt.figure(figsize=(8,6))
    plt.scatter(x,y,label='실제건강점수', alpha=0.7)
    plt.plot(x,y_pred,color='red',label='예측 선형회귀선')
    plt.xlabel('월구독료')
    plt.ylabel('건강점수')
    plt.title('단변수 선형회귀')
    plt.legend()
    plt.grid()
    plt.show()
today(df)

#성능평가
#mse : 오차 제곱 평균 -> 낮을수록 좋다
#R^2 : 설명력 -> 0.0~1.0 : 1에 가까울수록 좋다
df['건강점수'].agg(['min','max']) #예측하고자 하는 범위가 41-91사이. 
mse=mean_squared_error(y,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y,y_pred)
print(mse)
print(rmse)
print(r2)


df['월구독료'].corr(df['건강점수'])