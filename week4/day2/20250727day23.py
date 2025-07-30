import pandas as pd
import numpy as np

np.random.seed(42)

n = 200
df = pd.DataFrame({
    '수면시간': np.random.normal(9, 1, n).round(1),  # 평균 7시간, 표준편차 1
    'SNS사용시간': np.random.normal(6, 0.8, n).round(1),
    '하루공부시간': np.random.normal(8, 1, n).round(1),
    '아침식사빈도': np.random.randint(0, 8, n),
    '중간': np.random.randint(50, 100, n),
    '기말': np.random.randint(50, 100, n),
    '과제': np.random.randint(50, 100, n)
})

# 총점 계산
df['총점'] = (df['중간'] * 0.3 + df['기말'] * 0.4 + df['과제'] * 0.3).round(1)

df.info()
df

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split
x=df[['수면시간', 'SNS사용시간', '하루공부시간', '아침식사빈도']]
y=df['총점']

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)

def linear_evaluate(df):
    mse=mean_squared_error(y_test, y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test, y_pred)
    print('RMSE :', rmse)
    print('R2 :', r2)
linear_evaluate(df)

df.corr(numeric_only=True)['총점']

import seaborn as sns
sns.pairplot(data=df)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='Malgun Gothic')



def linear_visual(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', color='red')
    plt.xlabel('실제 총점')
    plt.ylabel('예측 총점')
    plt.tight_layout()
    plt.show()
linear_visual(df)

#kfold
kfold=KFold(n_splits=5, shuffle=True, random_state=42)

neg_mse_scores = cross_val_score(model, x, y, cv=kfold, scoring='neg_mean_squared_error')
rmse_score=np.sqrt(-neg_mse_scores)

print('평균 RMSE :', rmse_score.mean())

plt.plot(range(1, 6), rmse_score, marker='o')
plt.title("K-Fold 별 RMSE")
plt.xlabel("Fold 번호")
plt.ylabel("RMSE")
plt.ylim(0, max(rmse_score) + 5)
plt.grid(True)
plt.show()

df.corr(numeric_only=True)['총점'].sort_values(ascending=False)