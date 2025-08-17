from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

data=load_breast_cancer()

df=pd.DataFrame(data.data, columns=data.feature_names)
df['target']=data.target

df['target'].value_counts() #0 : 암, 1 : 정상

df.head()
df.describe()
df.info()

df.corr(numeric_only=True)['target'].sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rc('font', family='Malgun Gothic')

def boxplot_df(df):

    plt.subplot(1,2,1)
    sns.boxplot(df, y='mean perimeter')

    plt.subplot(1,2,2)
    sns.boxplot(df, y='mean area')

    plt.tight_layout()
    plt.show()
boxplot_df(df)

sns.histplot(df, x='mean perimeter', hue='target')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X=df.drop('target', axis=1)
y=df['target']

scaler=StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logi = LogisticRegression()
tree=DecisionTreeClassifier(random_state=42, max_depth=4)
rf=RandomForestClassifier(random_state=42)
xgb=XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', random_state=42)

logi.fit(X_train_scaled, y_train)
tree.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    y_proba=model.predict_proba(X_test)[:,1]

    print(f'{name} 결과')
    print('정확도 :', accuracy_score(y_test, y_pred))
    print('정밀도 :', precision_score(y_test, y_pred))
    print('재현율 :', recall_score(y_test, y_pred))
    print('f1 score :', f1_score(y_test, y_pred))
    print('AUC :', roc_auc_score(y_test, y_proba))
    print('confusion matrix : \n', confusion_matrix(y_test, y_pred))

def evaluate_model_logi(logi):
    y_pred = logi.predict(X_test_scaled)
    y_proba=logi.predict_proba(X_test_scaled)[:,1]

    print('로지스틱 회귀 결과')
    print('정확도 :', accuracy_score(y_test, y_pred))
    print('정밀도 :', precision_score(y_test, y_pred))
    print('재현율 :', recall_score(y_test, y_pred))
    print('f1 score :', f1_score(y_test, y_pred))
    print('AUC :', roc_auc_score(y_test, y_proba)) 
    print('confusion matrix : \n', confusion_matrix(y_test, y_pred))

evaluate_model('랜덤포레스트', rf)
evaluate_model_logi(logi)
evaluate_model('결정트리', tree)
evaluate_model('XGBoost', xgb)

#roc curve 시각화
from sklearn.metrics import roc_curve

def draw_roc(models, names, test_sets):
    plt.figure(figsize=(8,6))
    for model, name, X in zip(models, names, test_sets):
        y_proba = model.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc=roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc : })')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('roc curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

models=[logi, tree, rf, xgb]
names=['로지스틱회귀', '결정트리', '랜덤포레스트', 'XGBoost']
test_sets=[X_test_scaled, X_test, X_test, X_test]
draw_roc(models, names, test_sets)

def tradeoff(name, model):
    y_proba = model.predict_proba(X_test)[:,1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(8,6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='정밀도')
    plt.plot(thresholds, recalls[:-1], 'g-', label='재현율')
    plt.xlabel('thresholds')
    plt.ylabel('score')
    plt.title(f'{name} 정밀도 재현율 트레이드오프')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def tradeoff_logi(logi):
    y_proba = logi.predict_proba(X_test_scaled)[:,1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(8,6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='정밀도')
    plt.plot(thresholds, recalls[:-1], 'g-', label='재현율')
    plt.xlabel('thresholds')
    plt.ylabel('score')
    plt.title('로지스틱회귀 정밀도 재현율 트레이드오프')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
tradeoff('랜덤포레스트', rf)
tradeoff('xgboost', xgb)
tradeoff_logi(logi)

def importance_logi(logi):
    coef = pd.Series(logi.coef_[0], index=X.columns).sort_values()

    plt.figure(figsize=(10,6))
    sns.barplot(x=coef.values, y=coef.index)
    plt.title('로지스틱 회귀 - 변수 영향력 (계수)')
    plt.xlabel('회귀계수 (Coefficient)')
    plt.tight_layout()
    plt.show()
importance_logi(logi)

def importance(name, mod):
    importances = pd.Series(mod.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f'{name} - 변수 중요도')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
importance('랜덤포레스트', rf)
importance('xgboost', xgb)
importance('결정트리', tree)