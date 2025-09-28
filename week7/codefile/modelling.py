import pandas as pd
df_clean=pd.read_csv('C:/portfolio/code/week7/data/df_clean.csv')
df_clean.info()

#0. dis_score가 0,1,2이면 '위험군', 3,4이면 '주의', 5,6이면 '저위험군'
df_clean['target']=df_clean['dis_score'].replace({
    0:0,
    1:0,
    2:0,
    3:1,
    4:1,
    5:2,
    6:2
})
df_clean['target'].value_counts()
### 0 : 위험군 / 1 : 주의 / 2 : 저위험군


##target을 포함한 데이터파일 다시 저장
df_clean.to_csv('C:/portfolio/code/week7/data/df_clean.csv')


#1. 독립변수(X), 종속변수(y)선정
X=df_clean[['sex','BD1','age_group','edu_new','smoking']]
y=df_clean['target']

#2. 범주형 변수 원핫인코딩
X.info()
X=X.astype(str)
X=pd.get_dummies(X, drop_first=True)
X.info()

#3. 데이터 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#4. 모델 학습
## 4.1 로지스틱 회귀
from sklearn.linear_model import LogisticRegression
logit_model=LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced')
logit_model.fit(X_train, y_train)

## 4.2 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=1000, 
                                max_depth=16, 
                                min_samples_leaf=12, 
                                min_samples_split=16, 
                                n_jobs=-1)
rf_model.fit(X_train, y_train)

## 4.3 XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train,y_train)


#5. 예측
y_pred_logit=logit_model.predict(X_test)
y_pred_rf=rf_model.predict(X_test)
y_pred_xgb=xgb_model.predict(X_test)

#6. 성능평가
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, roc_auc_score
classification_report(y_test, y_pred_logit)
classification_report(y_test, y_pred_rf)
classification_report(y_test, y_pred_xgb)