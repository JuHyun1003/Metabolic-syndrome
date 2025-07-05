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
df.describe()

# 1. 학생별 총점을 기준으로 학점 부여 (np.where or apply)
# A: 80이상, B: 70이상, C: 60이상, D: 50이상, F: 나머지
# apply 또는 np.where로 직접 만들어보기
def grade(총점):
    if 총점 >= 80:
        return 'A'
    elif 총점 >= 70:
        return 'B'
    elif 총점 >= 60:
        return 'C'
    elif 총점 >= 50:
        return 'D'
    else:
        return 'F'
df['학점']=df['총점'].apply(grade)
df
# 2. '경영학' 전공 학생들만 필터링해서 중간고사 평균 구해봐
df[df['전공']=='경영학']['중간고사'].mean()

# 3. 성별별 과제점수 평균 구해봐 (groupby + mean)
df.groupby('성별')['과제점수'].mean()

# 4. 전공별 기말고사 평균과 표준편차를 동시에 구해봐 (groupby + agg)
df.groupby('전공')['기말고사'].agg(['mean','std'])

# 5. 키가 평균보다 큰 애들 중에서 과제점수 90 이상인 놈들 몇 명?
df[(df['키'] > df['키'].mean()) & (df['과제점수']>= 90)].shape[0]

# 6. 월소비금액을 구간(20만 단위)으로 잘라서 각 구간에 몇 명 있는지 빈도 세어봐 (pd.cut + value_counts)
df['소비구간']=pd.cut(df['월소비금액'], bins=range(0,1000001,200000))
df['소비구간'].value_counts().sort_index()

# 7. 중간고사 vs 기말고사 산점도 그리고, 과제점수를 컬러로 표현해봐 (scatterplot + hue)
matplotlib.rc('font', family='Malgun Gothic')
sns.scatterplot(data=df, x='중간고사', y='기말고사', hue='과제점수')
plt.title('중간 vs 기말')
plt.show()
plt.savefig('ex9.png')

# 8. 학년별 평균 총점 barplot 그리고 오차막대 없애기
sns.barplot(data=df, x='학년', y='총점', errorbar=None)
plt.title('학년별 평균 총점')
plt.show()
plt.savefig('ex10.png')

# 9. 성별 키 분포 KDE 곡선으로 그리기 (히스토그램 말고 kdeplot)
sns.kdeplot(data=df, x='키', hue='성별')
plt.title('성별 키 분포')
plt.show()
plt.savefig('ex11.png')

# 10. 중간/기말/과제점수 3개 subplot으로 각각 박스플롯 그려 (fig, ax 조합 사용)
plt.subplot(1,3,1)
plt.title('중간점수')
sns.boxplot(data=df, y='중간고사')

plt.subplot(1,3,2)
plt.title('기말점수')
sns.boxplot(data=df, y='기말고사')

plt.subplot(1,3,3)
plt.title('과제점수')
sns.boxplot(data=df, y='과제점수')

plt.show()
plt.savefig('ex12.png')

