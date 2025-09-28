import pandas as pd
df_raw=pd.read_excel('C:/portfolio/code/week7/sas1.xlsx')
df_raw.info()

# 나이 분포 보기
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.histplot(data=df_raw, x='age', kde=True)
plt.show()

df_raw['age'].min()
df_raw['age'].max()

# 나이 구간 나누기(19-29 : 청년층, 30-49세 : 중년층, 50~64세 : 장년기, 65세 이상 : 노년층)
# 1 : 청년층, 2 : 중년층, 3 : 노년층
df_age=df_raw
bins=[18,29,49,64,88]
labels=[1,2,3,4]
df_age['age_group']=pd.cut(df_age['age'], bins=bins, labels=labels)
df_age['age_group'].value_counts()

df_age.info()
df_age.to_csv('C:/portfolio/code/week7/data/df_age.csv')

# 질병 피쳐 생성
# BMI/ 과체중 : HE_BMI >= 23, 비만 : HE_BMI >= 25, 나머지 : 정상
# 혈압 / 고혈압 전단계 : HE_sbp - 120~139, HE_dbp - 80-89 , 고혈압 : HE_sbp >= 140, HE_dbp >= 90
# 이상 지질 혈증 : HE_CHOL >= 240 or HE_TG >= 200 , 위험그룹
df_dis = df_age
df_dis.info()

#1. 정상/과체중/비만 나누기
def bmi_category(bmi):
    if bmi >= 25:
        return '0'
    elif bmi >= 23:
        return '1'
    else:
        return '2'
# 0 : 비만, 1 : 과체중, 2 : 정상
df_dis['obese']=df_dis['HE_BMI'].apply(bmi_category)
df_dis['obese'].value_counts()

#2. 정상/고혈압 전단계/ 고혈압 나누기
#고혈압 전단계 : HE_sbp - 120~139, HE_dbp - 80-89 , 고혈압 : HE_sbp >= 140, HE_dbp >= 90
def bp_category(col):
    sbp, dbp=col['HE_sbp'], col['HE_dbp']
    if sbp >= 140 or dbp >= 90:
        return '0'
    elif (120 <= sbp <= 139) or (80 <= dbp <= 89):
        return '1'
    else:
        return '2'
#0 : 고혈압, 1 : 고혈압 전단계, 2 : 정상
df_dis['bp']=df_dis.apply(bp_category, axis=1)
df_dis[['ID', 'HE_sbp', 'HE_dbp', 'bp']].head()

#3. 이상 지질 혈증 여부
# 이상 지질 혈증 : HE_CHOL >= 240 or HE_TG >= 200, 위험군 : HE_chol이 200이상 239이하 or HE_TG가 150이상, 199 이하
def dys_category(col_1):
    chol, tg = col_1['HE_chol'], col_1['HE_TG']
    if chol >= 240 or tg >= 200:
        return '0'
    elif (200 <= chol <= 239) or (150 <= tg <= 199):
        return '1'
    else:
        return '2'
# 0 : 이상지질혈증(dys), 1 : dys 위험군, 2 : 정상
df_dis['dys']=df_dis.apply(dys_category, axis=1)
df_dis[['ID', 'HE_chol', 'HE_TG', 'dys']]

df_dis.info()
df_dis[['ID', 'obese', 'bp', 'dys']]

#4. 질병 3개 점수의 합 0~6 : 커질수록 병이 많음.
## 각 질병을 숫자형으로 변환
df_dis['obese']=df_dis['obese'].astype(int)
df_dis['bp']=df_dis['bp'].astype(int)
df_dis['dys']=df_dis['dys'].astype(int)

df_dis['dis_score']=df_dis['obese']+df_dis['bp']+df_dis['dys']
df_dis['dis_score'].value_counts()

sns.histplot(data=df_dis, x='dis_score')
plt.show()

df_dis.to_csv('C:/portfolio/code/week7/data/df_dis.csv')



df_daily=df_dis
#생활패턴 라벨링
##1. 소득수준(1 : 하, 2,3 : 중, 4 : 상)
df_daily['incm_new']=df_daily['incm'].replace({
    1 : 1,
    2 : 2,
    3 : 2,
    4 : 3
})
# 1 : 하, 2 : 중, 3 : 상
df_daily['incm_new'].value_counts()

##2. 교육수준(edu)(1,2 : 중졸이하, 3 : 고졸, 4 : 대졸)
df_daily['edu_new']=df_daily['edu'].replace({
    1 : 1,
    2 : 1,
    3 : 2,
    4 : 3
})
# 1 : 중졸이하, 2 : 고졸, 3 : 대졸
df_daily['edu_new'].value_counts()
df_daily.info()

##3. 평생 음주 경험(1 : 마셔본 적 없음, 2 : 있음)
df_daily['BD1'].value_counts()

##4. 흡연 여부(BS1_1 : 3이면, 비흡연자 / BS_1 : 1 또는 2이고, sm_presnt가 0이면, 금연자 / 그 외 : 흡연자)
df_daily['BS1_1'].value_counts()
def smoking(row):
    if row['BS1_1']==3:
        return '0'
    elif row['BS1_1'] in [1,2] and row['sm_presnt']==0:
        return '1'
    else:
        return '2'
# 0 : 비흡연자, 1 : 금연자, 2 : 흡연자
df_daily['smoking']=df_daily.apply(smoking, axis=1)
df_daily['smoking'].value_counts()

##5. 운동여부 : 원래 코드 그대로 사용(0 : 운동x, 1 : 운동o)
df_daily['exer'].value_counts()

df_daily.info()

##6. 성별 : 원래 코드 그대로 사용(1 : 남, 2 : 여)

df_daily.to_csv('C:/portfolio/code/week7/data/df_daily.csv')

##new 데이터셋 생성
df_new=df_daily.drop(columns=['region','age','incm','edu','BS1_1','sm_presnt'])

df_new.info()
df_new.to_csv('C:/portfolio/code/week7/data/df_new.csv')

df_new=pd.read_csv('C:/portfolio/code/week7/data/df_new.csv')
df_new.info()


#이상치 제거
## BMI <10 or BMI > 60 이면 이상치
## sbp < 70 or sbp > 250 / dbp < 40 or dbp > 150 이면 이상치
## chol < 70 or chol > 400이면 이상치
## TG < 30 or TG > 1000이면 이상치
# 조건 정의
import numpy as np
df_new.info()

cond = (
    (df_new['HE_BMI'].between(10, 60)) &   # BMI
    (df_new['HE_sbp'].between(70, 250)) &  # 수축기혈압
    (df_new['HE_dbp'].between(40, 150)) &  # 이완기혈압
    (df_new['HE_chol'].between(70, 400)) &    # 총콜레스테롤
    (df_new['HE_TG'].between(30, 1000))       # 중성지방
)

# sub_g 부여
df_new['sub_g'] = np.where(cond, 1, 0)

# 이상치 행 제거
df_clean = df_new[df_new['sub_g'] == 1].copy()

df_clean.info()
## 최종적으로 4818명 데이터

df_clean.to_csv('C:/portfolio/code/week7/data/df_clean.csv')