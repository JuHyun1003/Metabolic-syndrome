import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

df=pd.read_csv('C:/portfolio/code/week2/day7/gym_members.csv')
df=df.set_index('이름')
df['성별코드']=df['성별'].map({'남' : 0, '여' : 1})
df
# 기본 정보 요약
def eda_summary(df):
    print('[행 x 열]')
    print(df.shape)

    print('\n [기본정보]')
    print(df.info())

    print('\n [요약 정리]')
    print(df.describe())

    print('\n [결측치 비율]')
    print((df.isnull().sum() / len(df) * 100).sort_values(ascending=False))

    print('\n [범주형 통계]')
    for col in df.select_dtypes(include='object'):
        print(f'[{col} 통계]')
        print(df[col].value_counts())

eda_summary(df)

# 수치형 변수의 시각화
def eda_plot(df):
    matplotlib.rc('font', family=('Malgun Gothic'))
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols :

        #히스토그램
        plt.subplot(1,2,1)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} 히스토그램')

        #박스플롯
        plt.subplot(1,2,2)
        sns.boxplot(y=df[col])
        plt.title(f'{col} 박스플롯')

        plt.tight_layout()
        plt.show()
    
eda_plot(df)


# 상관관계 분석
def eda_corr(df):
    plt.figure(figsize=(10,6))
    numeric_df = df.select_dtypes(include='number')
    corr=numeric_df.corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('헬스장 상관분석')
    plt.show()

eda_summary(df)
eda_plot(df)
eda_corr(df)


#groupby 해석
df.select_dtypes(include='object').columns
df.select_dtypes(include='number').columns

#보충제 섭취 여부에 따른 해석
df['보충제 섭취 여부']=np.where(df['보충제']=='안 씀', '사용안함','사용함')
df['보충제 섭취 여부'].value_counts()
df.groupby('보충제 섭취 여부')['건강점수'].agg(['mean','min','max','std'])
df.groupby('보충제 섭취 여부')['근육량'].agg(['mean','min','max','std'])


def groupby_supplement(df) :
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.title('보충제 섭취 여부에 따른 근육량')
    sns.boxplot(data=df, x='보충제 섭취 여부', y='근육량')

    plt.subplot(1,3,2)
    plt.title('보충제 섭취 여부에 따른 건강점수')
    sns.boxplot(data=df, x='보충제 섭취 여부', y='건강점수')

    plt.subplot(1,3,3)
    plt.title('보충제 섭취 여부에 따른 체지방률')
    sns.boxplot(data=df, x='보충제 섭취 여부', y='인바디체지방률')

    plt.tight_layout()
    plt.show()
groupby_supplement(df)

#성별에 따른 해석
df['성별'].value_counts()

def groupby_sex(df):
    plt.figure(figsize = (10,4))

    plt.subplot(1,3,1)
    sns.boxplot(data=df, x='성별', y='근육량')
    plt.title('성별에 따른 근육량')

    plt.subplot(1,3,2)
    sns.boxplot(data=df, x='성별', y='건강점수')
    plt.title('성별에 따른 건강점수')

    plt.subplot(1,3,3)
    sns.boxplot(data=df, x='성별', y='인바디체지방률')
    plt.title('성별에 따른 체지방률')

    plt.tight_layout()
    plt.show()

groupby_sex(df)
