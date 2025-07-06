import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
data = {
    '고객ID': ['user_001', 'user_002', 'user_003', 'user_004', 'user_005',
              'user_006', 'user_007', 'user_008', 'user_009', 'user_010'],
    '성별': ['남', '여', '여', '남', '여', '남', '남', '여', '여', '남'],
    '연령': [25, 34, 45, 31, 29, 38, 42, 27, np.nan, 36],
    '직업': ['학생', '회사원', '주부', '회사원', '프리랜서', '공무원', '회사원', '자영업', '주부', '회사원'],
    '월소득': [120, 250, 200, 300, 180, np.nan, 270, 220, 190, 310],
    '포인트사용': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    '구매금액': [15000, 32000, 18000, 40000, 23000, 27000, 38000, 21000, 20000, 36000],
    '가입일': ['2022-01-01', '2021-12-15', '2020-06-30', '2022-02-18', 
              '2022-03-10', '2021-11-23', '2020-01-10', '2021-08-09', 
              '2022-01-20', '2020-12-31']
}

df = pd.DataFrame(data)
df['가입일'] = pd.to_datetime(df['가입일'])

def eda_summary(df):
    print('기본정보')
    print(df.info())

    print('\n [기술통계]')
    print(df.describe())

    print('\n [결측치 비율]')
    print((df.isnull().sum() / len(df)*100).sort_values(ascending=False))

    print('\n [범주형 비율]')
    for col in df.select_dtypes(include='object').columns:
        print(f'\n {col} 분포 :')
        print(df[col].value_counts())
eda_summary(df)

def eda_plot(df):
    matplotlib.rc('font', family='Malgun Gothic')
    numeric_cols = df.select_dtypes(include='number').columns

    plt.figure(figsize=(10,4))
    
    for col in numeric_cols:
     #히스토그램
        plt.subplot(1,2,1)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} - Histogram')

    #박스플롯
        plt.subplot(1,2,2)
        sns.boxplot(y= df[col])
        plt.title(f'{col} - Boxplot')
    
        plt.tight_layout()
        plt.show()

eda_plot(df)

def eda_corr(df):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('correlation heatmap')
    plt.show()
#문제발생
#첫 코드
def eda_corr(df):
    numeric_cols = df.select_dtypes(include='number').columns #이 코드는 Dataframe이 아니라 SeriesIndex이다.
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('correlation heatmap') 
    plt.show()

eda_corr(df)
