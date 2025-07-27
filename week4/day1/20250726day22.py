import pandas as pd
df_student_per_teacher=pd.read_excel('C:/portfolio/code/week4/day1/교원1인당학생수.xlsx', skiprows=1)
df_student_per_teacher

df_private=pd.read_excel('C:/portfolio/code/week4/day1/인구천명당사설학원수.xlsx', skiprows=1)
df_private

df_student = pd.read_excel('C:/portfolio/code/week4/day1/학급당학생수.xlsx', skiprows=1)
df_student

df_a=pd.read_excel('C:/portfolio/code/week4/day1/학생의학교생활만족도.xlsx', skiprows=1)
df_a

#학교생활만족도 총점 구하기
df_a.columns = df_a.columns.str.strip().str.replace(" ", "_")

df_a['불만족지수']=(
    df_a['매우만족'] * 1 +
    df_a['약간만족'] * 2 +
    df_a['보통'] * 3 +
    df_a['약간불만족'] * 4+
    df_a['매우불만족'] *5
)
#문자열 덧셈이라는 오류 발생
df_a.info()

score_cols = ['매우만족', '약간만족', '보통', '약간불만족', '매우불만족']
df_a[score_cols] = df_a[score_cols].apply(pd.to_numeric, errors='coerce')

df_a.info()
df_a['약간불만족'].isnull().sum()
df_a['매우불만족'].isnull().sum()
#약간 불만족에 null 1개, 매우불만족에 null 9개

#평균으로 결측치 매우기
df_a['약간불만족']=df_a['약간불만족'].fillna(df_a['약간불만족'].mean())
df_a['매우불만족']=df_a['매우불만족'].fillna(df_a['매우불만족'].mean())
df_a.info()
df_a.describe()
df_a
df_a.drop('계(재학생)', axis=1, inplace=True)
df_a.drop(['특성별(1)','특성별(2)'], axis=1, inplace=True)
df_a.dropna(axis=0, inplace=True)
df_a

#나머지 dataframe의 행정구역 이름 전환
region_rename = {
    '서울특별시': '서울',
    '부산광역시': '부산',
    '대구광역시': '대구',
    '인천광역시': '인천',
    '광주광역시': '광주',
    '대전광역시': '대전',
    '울산광역시': '울산',
    '세종특별자치시': '세종',
    '경기도': '경기',
    '강원특별자치도': '강원',
    '충청북도': '충북',
    '충청남도': '충남',
    '전북특별자치도': '전북',
    '전라남도': '전남',
    '경상북도': '경북',
    '경상남도': '경남',
    '제주특별자치도': '제주'
}
df_student_per_teacher['행정구역별']=df_student_per_teacher['행정구역별'].map(region_rename).fillna(df_student_per_teacher['행정구역별'])
df_private['행정구역별']=df_private['행정구역별'].map(region_rename).fillna(df_private['행정구역별'])
df_student['행정구역별']=df_student['행정구역별'].map(region_rename).fillna(df_student['행정구역별'])

df_student_per_teacher.columns = df_student_per_teacher.columns.str.strip().str.replace(" ", "_")
df_student_per_teacher.drop(['재적학생수(A)', '교원수(B)'], axis=1, inplace=True)
df_student_per_teacher.rename(columns={'교원1인당_학생수(A/B)' : '교원1인당학생수'},inplace=True)
df_student_per_teacher=df_student_per_teacher.set_index('행정구역별')

df_private.columns = df_private.columns.str.strip().str.replace(" ", "_")
df_private.drop(['사설학원수(A)_(개)', '주민등록인구(B)_(명)'], axis=1, inplace=True)
df_private.rename(columns={'인구천명당사설학원수(A/B*1,000)(개)' : '인구천명당사설학원수'},inplace=True)
df_private=df_private.set_index('행정구역별')
df_private

df_student.drop(['유치원','초등학교','중학교','고등학교'], axis=1, inplace=True)
df_student.rename(columns={'전체' : '학급당학생수'},inplace=True)
df_student=df_student.set_index('행정구역별')
df_student

df_a.rename(columns={'행정구역별(1)':'행정구역별'}, inplace=True)
df_a.drop(['매우만족','약간만족','보통','약간불만족','매우불만족'], axis=1, inplace=True)
df_a=df_a.set_index('행정구역별')
df_a

df_merged=pd.concat([df_a, df_private, df_student, df_student_per_teacher], axis=1)
df_merged

df_merged.corr()['불만족지수']

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def heatmap1(df_merged):
    plt.figure(figsize=(8,6))
    matplotlib.rc('font', family='Malgun Gothic')
    corr=df_merged.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('변수간 히트맵')
    plt.tight_layout()
    plt.show()
heatmap1(df_merged)
#불만족 지수는 대다수 변수와 상관관계x, 단, 학급당 학생수와 약한 음의 상관관계를 나타냄.

sns.pairplot(df_merged)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

x = df_merged[['교원1인당학생수', '인구천명당사설학원수', '학급당학생수']]
y = df_merged['불만족지수']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

def linear(df_merged):
    rmse=np.sqrt(mean_squared_error(y_test, y_pred))
    r2=r2_score(y_test,y_pred)
    print('RMSE :', rmse)
    print('R2 :', r2)
linear(df_merged)

df_merged['불만족지수'].agg({'max','min','mean'})

for col, coef in zip(x.columns, model.coef_):
    print(f"{col}: {coef:.3f}")

def linear_visual(df_merged):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='이상적 회귀선')
    plt.xlabel('실제 불만족지수')
    plt.ylabel('예측 불만족지수')
    plt.title('불만족지수 회귀결과')
    plt.tight_layout()
    plt.show()
    plt.savefig('불만족지수회귀결과.png')
linear_visual(df_merged)

import statsmodels.api as sm

x = df_merged[['교원1인당학생수', '인구천명당사설학원수', '학급당학생수']]
y = df_merged['불만족지수']
a = sm.add_constant(x)

model = sm.OLS(y, a).fit()
print(model.summary())