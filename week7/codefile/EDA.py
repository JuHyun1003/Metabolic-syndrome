import pandas as pd
df_clean=pd.read_csv('C:/portfolio/code/week7/data/df_clean.csv')
df_clean.info()

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
import seaborn as sns

#DIS_SCORE에 따른 건강지표들의 분포정도 시각화

##1. BMI
sns.boxplot(data=df_clean, x='dis_score',y='HE_BMI')
plt.show()

##2. 수축기혈압
sns.boxplot(data=df_clean, x='dis_score', y='HE_sbp')
plt.show()

##3. 이완기혈압
sns.boxplot(data=df_clean, x='dis_score', y='HE_dbp')
plt.show()

##4. 콜레스테롤
sns.boxplot(data=df_clean, x='dis_score',y='HE_chol')
plt.show()

##5. 중성지방
sns.boxplot(data=df_clean, x='dis_score',y='HE_TG')
plt.show()


# DIS_SCORE별 범주형 변수들의 분포 테이블
from scipy.stats import chi2_contingency

##1. 성별 분포
pd.crosstab(df_clean['dis_score'], df_clean['sex'], normalize='index')
###카이제곱 검정
ct_sex=pd.crosstab(df_clean['dis_score'], df_clean['sex'])
chi2, p, dof, expected = chi2_contingency(ct_sex)
p #p<0.001


##2. 평생음주여부
pd.crosstab(df_clean['dis_score'], df_clean['BD1'], normalize='index')
###카이제곱 검정
ct_bd1=pd.crosstab(df_clean['dis_score'], df_clean['BD1'])
chi2, p, dof, expected = chi2_contingency(ct_bd1)
p #p<0.001


##3. 소득수준
pd.crosstab(df_clean['dis_score'], df_clean['incm_new'], normalize='index')
###카이제곱 검정
ct_incm=pd.crosstab(df_clean['dis_score'], df_clean['incm_new'])
chi2, p, dof, expected = chi2_contingency(ct_incm)
p #p=0.38


##4. 학력수준
pd.crosstab(df_clean['dis_score'], df_clean['edu_new'], normalize='index')
###카이제곱 검정
ct_edu=pd.crosstab(df_clean['dis_score'], df_clean['edu_new'])
chi2, p, dof, expected = chi2_contingency(ct_edu)
p #p<0.001


##5. 흡연여부
pd.crosstab(df_clean['dis_score'], df_clean['smoking'], normalize='index')
###카이제곱 검정
ct_smok=pd.crosstab(df_clean['dis_score'], df_clean['smoking'])
chi2, p, dof, expected = chi2_contingency(ct_smok)
p #p<0.001


##6. 운동여부
pd.crosstab(df_clean['dis_score'], df_clean['exer'], normalize='index')
###카이제곱 검정
ct_exer=pd.crosstab(df_clean['dis_score'], df_clean['exer'])
chi2, p, dof, expected = chi2_contingency(ct_exer)
p #p=0.44


##7. 나이 분포
pd.crosstab(df_clean['dis_score'], df_clean['age_group'], normalize='index')
###카이제곱 검정
ct_age=pd.crosstab(df_clean['dis_score'], df_clean['age_group'])
chi2, p, dof, expected = chi2_contingency(ct_age)
p #p<0.001