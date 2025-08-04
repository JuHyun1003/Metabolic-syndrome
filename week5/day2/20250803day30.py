#모델 비교

import pandas as pd
import numpy as np
df=pd.read_csv('C:/portfolio/code/week5/day2/학생성적분포.csv')
df.info()
df.describe()
df.head()
df['A등급여부'].value_counts()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

#상관계수 분석
df.corr(numeric_only=True)['A등급여부'].sort_values(ascending=False)
matplotlib.rc('font', family='Malgun Gothic')
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.show()

#기말고사 전 중간, 과제 점수와 생활 패턴을 활용한 A등급 예측 모델
X=df[['나이', '성별코드', '하루공부시간', 'SNS사용시간', '수면시간','과제','중간']]
y=df['A등급여부']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

from sklearn.model_selection import StratifiedKFold, cross_val_score


#성능평가 함수 제적
def evaluate_single(name, model, X_tr, X_te):
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    print(f"[{name}] 단일 test셋 평가")
    print("정확도 :", accuracy_score(y_test, y_pred))
    print("정밀도 :", precision_score(y_test, y_pred, zero_division=0))
    print("재현율 :", recall_score(y_test, y_pred))
    print("F1 :", f1_score(y_test, y_pred))
    print("AUC :", roc_auc_score(y_test, y_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


def evaluate_kfold(name, model, X, y):
    print(f'[{name}] K-Fold 교차 검증 결과')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = cross_val_score(model, X,y,cv=skf, scoring='accuracy')
    print('평균 정확도 :', acc_scores.mean())

    pre_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
    print('평균 정밀도 :', pre_scores.mean())

    re_scores=cross_val_score(model, X,y,cv=skf, scoring='recall')
    print('평균 재현율 :', re_scores.mean())

    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    print('평균 f1:', f1_scores.mean())

    roc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    print('평균 AUC :', roc_scores.mean())

#모델 학습
#로지스틱 회귀
logic=LogisticRegression(max_iter=1000, class_weight='balanced')
logic.fit(X_train_scaled, y_train)
evaluate_single('로지스틱 회귀', logic, X_train_scaled, X_test_scaled)
evaluate_kfold('로지스틱 회귀', logic, X, y)

y_test.value_counts()

#결정트리
dt=DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)
evaluate_single('결정트리', dt, X_train, X_test)
evaluate_kfold('결정트리', dt, X, y)

#랜덤포레스트
rf=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
evaluate_single('랜덤포레스트', rf, X_train, X_test)
evaluate_kfold('랜덤포레스트', rf, X, y)

#xgboost
pos = sum(y==1)
neg = sum(y==0)
scale = neg / pos
xgb=XGBClassifier(scale_pos_weight=scale, eval_metric = 'logloss', random_state=42)
xgb.fit(X_train, y_train)
evaluate_single('XGBoost', xgb, X_train, X_test)
evaluate_kfold('XGBoost', xgb, X, y)

#정밀도 재현율 트레이드 오프 시각화
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def tradeoff_visual(name, model, X_te):
    y_scores=model.predict_proba(X_te)[:,1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    plt.figure(figsize=(8,6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='정밀도')
    plt.plot(thresholds, recalls[:-1], 'g-', label='재현율')
    plt.xlabel('Thresholds')
    plt.ylabel('score')
    plt.title(f'{name} 재현율 vs 정밀도 트레이드 오프')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
tradeoff_visual('[로지스틱회귀]', logic, X_test_scaled)
tradeoff_visual('[결정트리]', dt, X_test)
tradeoff_visual('[랜덤포레스트]', rf, X_test)
tradeoff_visual('[XGBoost]', xgb, X_test)