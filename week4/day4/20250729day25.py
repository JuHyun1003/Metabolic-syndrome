import pandas as pd
import numpy as np
import random

np.random.seed(42)

n = 100

# 성별 랜덤
genders = np.random.choice(['남', '여'], n)

# 전공 랜덤
majors = np.random.choice(['생명과학', '경영학', '심리학', '화학', '물리학', '컴공', 'AI'], n)

# 나이: 20~25세
ages = np.random.randint(20, 26, n)

# 수면시간: 평균 6.5, std 1
sleep_hours = np.round(np.random.normal(6.5, 1, n), 1)
sleep_hours = np.clip(sleep_hours, 4, 9)

# 하루 공부시간: 평균 2.5, std 1
study_hours = np.round(np.random.normal(2.5, 1, n), 1)
study_hours = np.clip(study_hours, 0.5, 5)

# SNS 사용시간: 평균 2.5, std 1.5
sns_hours = np.round(np.random.normal(2.5, 1.5, n), 1)
sns_hours = np.clip(sns_hours, 0, 6)

# 중간, 기말, 과제 점수 (정규분포 + clipping)
midterms = np.clip(np.random.normal(75, 10, n).astype(int), 50, 100)
finals = np.clip(np.random.normal(78, 12, n).astype(int), 50, 100)
assignments = np.clip(np.random.normal(80, 10, n).astype(int), 50, 100)

# 이름은 그냥 학생1~100
names = [f'학생{i+1}' for i in range(n)]

# 데이터프레임 생성
df = pd.DataFrame({
    '이름': names,
    '성별': genders,
    '전공': majors,
    '나이': ages,
    '수면시간': sleep_hours,
    '하루공부시간': study_hours,
    'SNS사용시간': sns_hours,
    '중간': midterms,
    '기말': finals,
    '과제': assignments
})

# 총점 계산
df['총점'] = df['중간']*0.3 + df['기말']*0.4 + df['과제']*0.3

# 학점 부여
def grade(score):
    if score >= 90:
        return 'A+'
    elif score >= 85:
        return 'A0'
    elif score >= 80:
        return 'A-'
    elif score >= 75:
        return 'B+'
    elif score >= 70:
        return 'B0'
    elif score >= 65:
        return 'C+'
    elif score >= 60:
        return 'C0'
    else:
        return 'F'

df['학점'] = df['총점'].apply(grade)
df['A등급여부'] = df['학점'].isin(['A+', 'A0', 'A-']).astype(int)

from sklearn.preprocessing import StandardScaler

# 범주형 처리
df['성별코드'] = df['성별'].map({'남': 0, '여': 1})
df = pd.get_dummies(df, columns=['전공'], drop_first=True)
df.info()
df

#수치형 스케일링
scaler = StandardScaler()
df[['수면시간', '하루공부시간', 'SNS사용시간', '나이']] = scaler.fit_transform(
    df[['수면시간', '하루공부시간', 'SNS사용시간', '나이']]
)
df.describe()

df.head()

# 입력변수와 라벨 나누기
x = df.drop(columns=['이름', '성별', '중간', '기말', '과제', '총점', '학점', 'A등급여부'])  # 필요없는 거 제거
y= df['A등급여부']  # 맨 끝에 붙여서 확인용

x.info()
print(x.head())
y.info()
y.head()


#day25
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#로지스틱 회귀 모델 학습
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train, y_train)

#예측 및 평가
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = model.predict(x_test)

print('정확도 :', accuracy_score(y_test, y_pred))
print('오차 행렬 : \n', confusion_matrix(y_test, y_pred))
print('분류 리포트 : \n', classification_report(y_test, y_pred))


# 확률 출력 + 시각화
y_prob=model.predict_proba(x_test)[:, 1]

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', family='Malgun Gothic')
plt.hist(y_prob[y_test==1], bins=10, alpha=0.5, label='True A 등급')
plt.hist(y_prob[y_test==0], bins=10, alpha=0.5, label='False A등급')
plt.legend()
plt.title('예측 확률 분포')
plt.show()

df.corr(numeric_only=True)['A등급여부'].sort_values(ascending=False)