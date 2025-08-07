## 모델 튜닝
# GridSearchCV, RandomizedSearchCV

import pandas as pd
import numpy as np
from scipy.special import expit  # 시그모이드

np.random.seed(42)
n = 2000

# Step 1: 기본 변수 생성
df = pd.DataFrame({
    '성별': np.random.choice(['남', '여'], size=n),
    '나이': np.random.randint(20, 80, size=n),
    '수축기혈압': np.random.normal(125, 15, size=n).astype(int),
    '이완기혈압': np.random.normal(80, 10, size=n).astype(int),
    '공복혈당': np.random.normal(100, 20, size=n).astype(int),
    '총콜레스테롤': np.random.normal(200, 30, size=n).astype(int),
    '흡연여부': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    '음주빈도': np.random.poisson(2, size=n),
    '운동빈도': np.random.poisson(3, size=n),
    'BMI': np.round(np.random.normal(24, 3, size=n), 1),
    '간수치': np.random.normal(35, 10, size=n).astype(int)
})

# Step 2: 위험점수 계산 (비선형 + 상호작용 + 노이즈)
risk_score = (
    0.04 * (df['나이'] ** 1.1) +
    0.06 * np.log1p(df['공복혈당']) +
    0.07 * (df['총콜레스테롤'] / 200) ** 1.5 +
    0.12 * df['BMI'] * df['흡연여부'] +
    0.1 * (df['수축기혈압'] > 140).astype(int) +
    0.08 * (df['간수치'] > 50).astype(int) +
    0.1 * (df['운동빈도'] < 2).astype(int) +
    0.03 * df['음주빈도'] +
    0.02 * (df['성별'] == '남').astype(int) +
    np.random.normal(0, 5, size=n)  # 현실 노이즈 추가
)

# Step 3: 시그모이드 함수로 확률화 + 이진화
risk_prob = expit((risk_score - 6) / 2)  # 기준값 조정
df['심혈관질환위험'] = np.where(np.random.rand(n) < risk_prob, 1, 0)

df.info()
df.describe()
df.head()
df['심혈관질환위험'].value_counts()

df['성별코드']=df['성별'].map({'남' : 0, '여' : 1})

X=df.drop(['심혈관질환위험', '성별'], axis=1)
y=df['심혈관질환위험']

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

logi = LogisticRegression()
logi.fit(X_train_scaled, y_train)

rf=RandomForestClassifier(random_state=42)
dt=DecisionTreeClassifier(random_state=42)
xgb=XGBClassifier(random_state=42, use_label_encoder=False, eval_metrics='logloss')

from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='Malgun Gothic')

def evaluate(name, model):
    y_pred=model.predict(X_test)
    y_proba=model.predict_proba(X_test)[:, 1]

    print(f'{name} 성능 지표')
    print('confussion matrix \n', confusion_matrix(y_test, y_pred))
    print('f1 score :', f1_score(y_test, y_pred))
    print('AUC :', roc_auc_score(y_test, y_proba))
    print('classification report \n', classification_report(y_test, y_pred))

    precisions, recalls, threshold=precision_recall_curve(y_test, y_proba)
    plt.plot(threshold, precisions[:-1], 'b--', label='정밀도')
    plt.plot(threshold, recalls[:-1], 'g-', label='재현율')
    plt.xlabel('Threshold')
    plt.ylabel('score')
    plt.title(f'{name} 정밀도 vs 재현율 트레이드오프')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def evaluate_logi(logi):
    y_pred=logi.predict(X_test_scaled)
    y_proba=logi.predict_proba(X_test_scaled)[:, 1]

    print('로지스틱 회귀 성능 지표')
    print('confussion matrix \n', confusion_matrix(y_test, y_pred))
    print('f1 score :', f1_score(y_test, y_pred))
    print('AUC :', roc_auc_score(y_test, y_proba))
    print('classification report \n', classification_report(y_test, y_pred))

    precisions, recalls, threshold=precision_recall_curve(y_test, y_proba)
    plt.plot(threshold, precisions[:-1], 'b--', label='정밀도')
    plt.plot(threshold, recalls[:-1], 'g-', label='재현율')
    plt.xlabel('Threshold')
    plt.ylabel('score')
    plt.title('로지스틱 회귀 정밀도 vs 재현율 트레이드오프')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
xgb.fit(X_train, y_train)

evaluate_logi(logi)
evaluate('랜덤포레스트', rf)
evaluate('결정트리', dt)
evaluate('XGBoost', xgb)


# GridSearchCV
# “하이퍼파라미터를 ‘그리드(격자)’처럼 조합해서 전수조사하고, 그 중 제일 성능 좋은 조합을 찾아주는 애".

param_rf = {
    'n_estimators':[100,200,300],
    'max_depth' : [5,10,15],
    'min_samples_split' : [2,5],
    'min_samples_leaf' : [1,2,3]
}

grid_rf = GridSearchCV(rf, param_rf, cv=3, scoring='recall', n_jobs=-1)
#랜덤포레스트 모델을 f1-score기준으로 param_rf에 있는 모든 하이퍼파라미터 조합을 3-Fold교착검증으로 돌려보고 그 중 제일 f1좋은 조합 뽑기
grid_rf.fit(X_train, y_train)

evaluate('튜닝 랜덤포레스트', grid_rf)
evaluate('랜덤포레스트', rf)

param_xgb = {
    'n_estimators':[100,200],
    'max_depth' : [3,4,5],
    'learning_rate' : [0.01, 0.1, 0.3],
    'subsample' :[0.7, 1],
    'colsample_bytree' :[0.7, 1]
}

rand_xgb=RandomizedSearchCV(xgb, param_xgb, cv=3, scoring='recall', n_iter=10, n_jobs=-1, random_state=42)
rand_xgb.fit(X_train, y_train)

evaluate('튜닝 xgboost', rand_xgb)
evaluate('xgboost', xgb)