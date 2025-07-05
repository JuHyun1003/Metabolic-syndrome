import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

np.random.seed(42)

n = 100
data = {
    '이름': [f'학생{i+1}' for i in range(n)],
    '성별': np.random.choice(['남', '여'], size=n),
    '학년': np.random.choice([1, 2, 3, 4], size=n),
    '전공': np.random.choice(['생명과학', '컴퓨터공학', '경영학', '심리학'], size=n),
    '중간고사': np.random.randint(40, 101, size=n),
    '기말고사': np.random.randint(40, 101, size=n),
    '과제점수': np.random.randint(20, 101, size=n),
    '키': np.random.normal(170, 10, size=n).round(1),
    '몸무게': np.random.normal(65, 15, size=n).round(1),
    '월소비금액': np.random.normal(500000, 100000, size=n).astype(int)
}

df = pd.DataFrame(data)
df['총점'] = (df['중간고사'] * 0.4 + df['기말고사'] * 0.4 + df['과제점수'] * 0.2).round(1)
df=df.set_index('이름')
df

def grade(총점):
    if 총점 >= 80:
        return 'A'
    elif 총점 >= 70 :
        return 'B'
    elif 총점 >= 60:
        return 'C'
    elif 총점 >= 50:
        return 'D'
    else:
        return 'F'
df['학점']=df['총점'].apply(grade)
df

matplotlib.rc('font', family='Malgun Gothic')

#1. 학년별 평균 총점 구하기
# errorbar='sd' : 오차 막대, 길수록 널뛰기가 심하다. 짧을수록 데이터가 덜 흩어져있다.
sns.barplot(data=df, x='학년', y='총점', estimator = 'mean', errorbar='sd')
plt.title('학년별 평균 총점')
plt.show()
plt.savefig('ex1.png')
# 학년별 총점의 평균과 표준편차 수치화
df.groupby('학년')['총점'].agg(['mean','std'])

#2. 전공별 성적 분포
sns.boxplot(data=df, x='전공', y='총점')
plt.title('전공별 성적 분포')
plt.show()
plt.savefig('ex2.png')
# 전공별 성적 요약
df.groupby('전공')['총점'].describe()

#3. 성별에 따른 키 분포
sns.histplot(data=df, x='키', hue='성별', kde=True)
plt.title('성별 키 분포')
plt.show()
plt.savefig('ex3.png')

#4. 몸무게 vs 키 산점도
sns.scatterplot(data=df, x='키', y='몸무게', hue='성별')
plt.title('키 vs 몸무게 (성별)')
plt.show()
plt.savefig('ex4.png')

#5. 학점 분포 빈도
sns.countplot(data=df, x='학점', order=['A', 'B', 'C', 'D', 'F'])
plt.title('학점 분포')
plt.show()
plt.savefig('ex5.png')

#6. 총점 히스토그램 + KDE
# KDE : 데이터를 부드러운 곡선으로 추정해서 보여줌
sns.histplot(data=df, x='총점', kde=True)
plt.title('총점 분포')
plt.show()
plt.savefig('ex6.png')

# KDE는 hue랑 같이 쓰기도 한다.
sns.histplot(data=df, x='총점', kde=True, hue='성별')
plt.title('성별 간 총점 분포')
plt.show()
plt.savefig('ex7.png')

#7. pair plot : 성적간 상관관계 파악
sns.pairplot(df[['중간고사','기말고사','과제점수','총점']])
plt.show()
plt.savefig('ex8.png')


fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i, col in enumerate(['중간고사','기말고사','과제점수']):
    sns.histplot(data=df, x=col, kde=True, ax=axes[i])
    axes[i].set_title(f'{col} 분포')
plt.tight_layout()
plt.show()

plt.subplot(1,4,1)
sns.histplot(data=df, x='중간고사', kde=True)
plt.tight_layout()
plt.title('중간고사 성적 분포')

plt.subplot(1,4,2)
sns.histplot(data=df, x='기말고사', kde=True)
plt.tight_layout()
plt.title('기말고사 성적 분포')

plt.subplot(1,4,3)
sns.histplot(data=df, x='과제점수', kde=True)
plt.tight_layout()
plt.title('과제점수 성적 분포')

plt.subplot(1,4,4)
sns.histplot(data=df, x='총점', kde=True)
plt.tight_layout()
plt.title('총점 성적 분포')

plt.show()