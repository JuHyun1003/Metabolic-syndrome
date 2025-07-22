import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib

df=pd.read_csv('C:/portfolio/code/week3/day3/gym_member.csv')
df
df.info()
df.describe()

#단변수 모델
#pairplot으로 각 변수들간의 관계 파악
def corr_graph(df):
    plt.figure(figsize=(8,5))
    matplotlib.rc('font', family='Malgun Gothic')
    sns.pairplot(df.select_dtypes(include='number'))
    plt.tight_layout()
    plt.show()
corr_graph(df)
#월구독료, 건강점수가 관계있어 보임
df[['월구독료','건강점수']].corr()
df.corr(numeric_only=True)

sns.lmplot(data=df, x='월구독료', y='건강점수')
plt.show()

x=df[['월구독료']]
y=df['건강점수']
model=LinearRegression()
model.fit(x,y)
y_pred=model.predict(x)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
mse=mean_squared_error(y,y_pred)
rmse=np.sqrt(mse)
print('rmse :', rmse)
print('r2 :', r2_score(y,y_pred))
df['건강점수'].agg(['min','max'])



#단변수모델로 돌린 결과값들 기록하기

#다변수 모델
#1. 독립변수(a), 종속변수(b) 정의
a=df[['월구독료','운동빈도','보충제코드','나이','성별코드']]
b=df['건강점수']
#2. 모델 학습
model=LinearRegression()
model.fit(a,b)
#3. 예측값 생성
b_pred=model.predict(a)
#4. 성능평가
def ability_multimodel(df):
    mse=mean_squared_error(b,b_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(b,b_pred)
    print('rmse :', rmse)
    print('r2 :', r2)
ability_multimodel(df)
#해석 : rmse값은 더 내려갔고, r2값은 1과 더 가까워졌다. 모델의 성능이 좋아졌음을 알 수 있다.

#5. 예측값 vs 실제값 시각화
plt.scatter(b,b_pred)
plt.plot([b.min(),b.max()], [b.min(), b.max()], 'r--')
plt.xlabel('실제 건강점수')
plt.ylabel('예측 건강점수')
plt.show()
plt.savefig('ex2.png')

#6. 변수별 기울기(파라미터 출력)
for name, coef in zip(a.columns, model.coef_):
    print(f'{name} : {coef}')