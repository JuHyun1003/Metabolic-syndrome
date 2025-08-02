import pandas as pd
import numpy as np
df=pd.read_csv('C:/portfolio/code/week4/day7/depression_data.csv')
df.info()

df.head()

#전처리
#1. '이름'열 삭제
df=df.drop(columns=['Name'])
df.info()

#2. 범주형 변수의 수치화
##1. 이진형 변수의 수치화
df['History of Mental Illness'].value_counts()
df['History of Mental Illness'] = df['History of Mental Illness'].map({'Yes': 1, 'No': 0})

df['History of Substance Abuse'].value_counts()
df['History of Substance Abuse']=df['History of Substance Abuse'].map({'Yes' : 1, 'No' : 0})

df['Family History of Depression'].value_counts()
df['Family History of Depression']=df['Family History of Depression'].map({'Yes' : 1, 'No' : 0})

df['Chronic Medical Conditions'].value_counts()
df['Chronic Medical Conditions']=df['Chronic Medical Conditions'].map({'Yes' : 1, 'No' : 0})

df['Employment Status'].value_counts()
df['Employment Status']=df['Employment Status'].map({'Employed' : 1, 'Unemployed' : 0})
##2. 다중 변수의 수치화
multi_cols = [
    'Marital Status',
    'Education Level',
    'Smoking Status',
    'Physical Activity Level',
    'Alcohol Consumption',
    'Dietary Habits',
    'Sleep Patterns'
]
df=pd.get_dummies(df, columns=multi_cols, drop_first=True)
df.info()

#3. 'Depression'열 생성
#History of Mental Illness 또는 Family History of Depression 가 yes이면 우울증 가능성 있음(1), 둘 다 없으면 0
df['Depression'] = np.where(
    (df['Family History of Depression']==1) | 
     (df['History of Mental Illness'] == 1),
     1,0
     )
df['Depression'].value_counts()

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

df.corr()['Depression'].sort_values(ascending=False)
df.info()
# 비율 테이블
ct = pd.crosstab(df['Depression'], df['Marital Status_Widowed'], normalize='index')

# 시각화
matplotlib.rc('font', family='Malgun Gothic')
ct.plot(kind='bar', stacked=True, color=['#a6bddb', '#045a8d'])
plt.title('Depression vs Widowed Status')
plt.xlabel('Depression (0: 없음, 1: 있음)')
plt.ylabel('비율')
plt.legend(title='Widowed (0: 아님, 1: 맞음)')
plt.tight_layout()
plt.show()

sns.countplot(data=df, x='Depression', hue='Marital Status_Widowed')
plt.title('우울증 여부 vs 과부 여부 카운트')
plt.xlabel('우울증 여부 (0: 없음, 1: 있음)')
plt.ylabel('사람 수')
plt.legend(title='Widowed (0: 아님, 1: 맞음)')
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

X=df.drop(columns=['Depression',
                   'History of Mental Illness',
                   'Family History of Depression'])
y=df['Depression']

num_cols=['Age','Income', 'Number of Children']
scaler=StandardScaler()
X[num_cols]=scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#모델 학습
model=LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

def model_evaluate(df):
    print("🔹 confusion matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\n🔹 classification report")
    print(classification_report(y_test, y_pred))

    print("\n🔹 ROC AUC score")
    print(roc_auc_score(y_test, y_proba))
model_evaluate(df)

##모델 튜닝

#roc curve 시각화
auc_score = roc_auc_score(y_test,y_proba)
fpr, tpr, thresholds=roc_curve(y_test, y_proba)
def roc_curve_visual(df):
    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr,label=f'ROC Curve (AUC = {auc_score :.2f})')
    plt.plot([0,1], [0,1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
roc_curve_visual(df)

from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
precisions, recalls, pr_thresholds=precision_recall_curve(y_test, y_proba)
def precision_recall(df):
    plt.figure(figsize=(8,6))
    plt.plot(pr_thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(pr_thresholds, recalls[:-1], 'g--', label='Recall')
    plt.xlabel('Thresholds')
    plt.ylabel('Score')
    plt.title('Precision & Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
precision_recall(df)

#이 모델은 '재현율' 위주 thresholds=0.38정도
def custom_threshold(score):
    custom_threshold = score
    y_pred_custom = (y_proba >= custom_threshold).astype(int)

    print('confusion matrix : \n', confusion_matrix(y_test, y_pred_custom))
    print('classification report : \n', classification_report(y_test, y_pred_custom))
    print('정밀도 :',precision_score(y_test, y_pred_custom))
    print('재현율 :', recall_score(y_test, y_pred_custom))
    print('f1 score :', f1_score(y_test, y_pred_custom))
custom_threshold(0.38)

#평가 : 실제 우울증 환자들 중에서 96%를 잡아냄. 그러나, 