## 결정트리, XGBoost 구현

import pandas as pd
import numpy as np

np.random.seed(42)
n = 1500 # 샘플 수

# 기본 인구 통계
성별 = np.random.choice(['남', '여'], n)
나이 = np.random.normal(loc=45, scale=12, size=n).astype(int)
나이 = np.clip(나이, 20, 80)

# 라이프스타일
수면시간 = np.round(np.random.normal(loc=6.5, scale=1.5, size=n), 1)
운동빈도 = np.random.choice(['주1회 이하', '주2~3회', '주4회 이상'], n, p=[0.4, 0.4, 0.2])
흡연여부 = np.random.choice(['비흡연', '과거흡연', '현재흡연'], n, p=[0.5, 0.2, 0.3])
음주빈도 = np.random.choice(['거의안함', '주1회', '주2회 이상'], n, p=[0.3, 0.4, 0.3])

# 건강 수치
BMI = np.round(np.random.normal(loc=24, scale=4, size=n), 1)
수축기혈압 = np.round(np.random.normal(loc=125, scale=15, size=n), 1)
공복혈당 = np.round(np.random.normal(loc=95, scale=20, size=n), 1)
콜레스테롤 = np.round(np.random.normal(loc=200, scale=30, size=n), 1)

# 질병위험 생성 (가짜 알고리즘)
risk_score = (
    (BMI > 27).astype(int) +
    (수축기혈압 > 135).astype(int) +
    (공복혈당 > 110).astype(int) +
    (콜레스테롤 > 240).astype(int) +
    (흡연여부 == '현재흡연').astype(int) +
    (운동빈도 == '주1회 이하').astype(int)
)

질병위험 = (risk_score >= 3).astype(int)

# 데이터프레임 구성
df = pd.DataFrame({
    '성별': 성별,
    '나이': 나이,
    '수면시간': 수면시간,
    '운동빈도': 운동빈도,
    '흡연여부': 흡연여부,
    '음주빈도': 음주빈도,
    'BMI': BMI,
    '수축기혈압': 수축기혈압,
    '공복혈당': 공복혈당,
    '콜레스테롤': 콜레스테롤,
    '질병위험': 질병위험
})

df.head()

df.describe()
df.info()
df['질병위험'].value_counts()

#데이터 전처리
#1. 성별 
df['성별코드']=df['성별'].map({'남' : 0, '여' : 1})
df['성별코드'].value_counts()

#2. 운동빈도, 흡연여부, 음주빈도
df=pd.get_dummies(df, columns=['운동빈도', '흡연여부', '음주빈도'], drop_first=True)
df.info()

df.corr(numeric_only=True)['질병위험'].sort_values(ascending=False)

#모델 변수 분리 및 세트 분할
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

X=df.drop(columns=['질병위험', '성별'], axis=1)
y=df['질병위험']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

matplotlib.rc('font', family='Malgun Gothic')
def evaluate_model(mod, model, thr=0.5):
    y_proba=model.predict_proba(X_test)[:,1]
    y_pred=(y_proba >= thr).astype(int)

    print('confusion matrix : \n', confusion_matrix(y_test, y_pred))
    print('정밀도 :', precision_score(y_test, y_pred))
    print('재현율 :', recall_score(y_test, y_pred))
    print('f1 score :', f1_score(y_test, y_pred))
    print('AUC :', roc_auc_score(y_test, y_proba))

    fpr, tpr, thresholds=roc_curve(y_test, y_proba)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f'{model}(AUC = {roc_auc_score(y_test, y_proba)})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f'{mod} ROC Curve')
    plt.legend
    plt.grid()

    plt.subplot(1,2,2)
    precisions, recalls, threshold=precision_recall_curve(y_test, y_proba)
    plt.plot(threshold, precisions[:-1], 'b--', label='정밀도')
    plt.plot(threshold, recalls[:-1], 'g-', label='재현율')
    plt.xlabel('Threshold')
    plt.ylabel('score')
    plt.title(f'{mod} 정밀도 vs 재현율 트레이드오프')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

#모델 학습
#1. 결정트리
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth =5, random_state=42)
tree_model.fit(X_train, y_train)

evaluate_model('결정트리', tree_model)

#2. XGboost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(use_label_encodeer = False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

evaluate_model('XGBoost', xgb_model, 0.3)

#모델의 변수 중요도 시각화
X.info()

def importance_model(mod_name, mod):
    feature_name = X.columns
    importances = pd.Series(mod.feature_importances_, index=feature_name).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x=importances, y=importances.index)
    plt.title(f'{mod_name} 변수 중요도')
    plt.tight_layout()
    plt.show()
importance_model('결정트리', tree_model)
importance_model('XGBoost', xgb_model)