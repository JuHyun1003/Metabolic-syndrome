import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('C:/portfolio/code/week3/day7/grade.csv')
df.info()
df=df.drop(columns=['학점'])


#학점 구간을 나누기 위한 시각화
matplotlib.rc('font', family='Malgun Gothic')
def grade(df):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.boxplot(data=df, y='총점')

    plt.subplot(1,2,2)
    sns.histplot(data=df, x='총점', kde=True)
    plt.show()
grade(df)
#A+ : 110이상, A0 : 100이상, B+ : 90이상, B0 : 80이상, C  : 70이상, D : 60이상, 그외 : F
def grade(score):
    if score >= 110:
        return 'A+'
    elif score >= 100:
        return 'A0'
    elif score >= 90:
        return 'B+'
    elif score  >= 80:
        return 'B0'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'
df['학점']= df['총점'].apply(grade)
df['학점'].value_counts()
df.to_excel('grade.xlsx', index=False)

df.corr(numeric_only=True)['총점']

sns.pairplot(data=df)
plt.figure(figsize=(10,6))
plt.tight_layout()
plt.show()

x=df[['하루공부시간', '과제투자시간', '출석률', 'SNS사용시간', '게임시간', '카페인섭취량(mg)']]
y=df['총점']
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
y_pred=model.predict(x)

def model_evaluate(df):
    mse=mean_squared_error(y,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y,y_pred)
    print('RMSE :', rmse)
    print('R2 :', r2)
model_evaluate(df)

model2=LinearRegression()
model2.fit(x,y)
y_pred2=model2.predict(x)

def model2_evaluate(df):
    mse=mean_squared_error(y,y_pred2)
    rmse=np.sqrt(mse)
    r2=r2_score(y,y_pred2)
    print('RMSE :', rmse)
    print('R2 :', r2)
model2_evaluate(df)

def linear_visual(df):
    plt.scatter(y,y_pred2)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('실제 총점')
    plt.ylabel('예측 총점')
    plt.show()
linear_visual(df)


residuals=y-y_pred2
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
    plt.scatter(y_pred2, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('예측값')
    plt.ylabel('잔차')
    plt.show()
residual_visual(df)

# P>|t| 가 0.05보다 큰 변수들 제거 후 모델 간소화
x1=df[['하루공부시간', '과제투자시간', '출석률']]
y1=df['총점']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x1,y1,test_size=0.2, random_state=42)

model3=LinearRegression()
model3.fit(x_train, y_train)

y_train_pred=model3.predict(x_train)
y_test_pred=model3.predict(x_test)

def model3_evaluate(df):
    train_rmse = mean_squared_error(y_train, y_train_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred)
    print('Train RMSE :', np.sqrt(train_rmse))
    print('Test_RMSE :', np.sqrt(test_rmse))
model3_evaluate(df)

def linear3_visual(df):
    plt.figure(figsize=(7,6))
    plt.scatter(y_test, y_test_pred, label='실제값 vs 예측값')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='이상적 예측치')
    plt.xlabel('실제 총점')
    plt.ylabel('예측 총점')
    plt.legend()
    plt.show()
linear3_visual(df)



# 표준화 후 모델학습
a=df[['하루공부시간', '과제투자시간', '출석률']]
b=df['총점']
a_train, a_test, b_train, b_test = train_test_split(a,b,test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
a_train_scaled=scaler.fit_transform(a_train)
a_test_scaled=scaler.transform(a_test)

model4=LinearRegression()
model4.fit(a_train_scaled, b_train)

b_train_pred=model4.predict(a_train_scaled)
b_test_pred=model4.predict(a_test_scaled)

def linear4_evaluate(df):
    print('Train RMSE:', np.sqrt(mean_squared_error(b_train, b_train_pred)))
    print('Test RMSE:', np.sqrt(mean_squared_error(b_test, b_test_pred)))
linear4_evaluate(df)