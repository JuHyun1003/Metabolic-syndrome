import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
data = {
    '이름': ['학생1', '학생2', '학생3', '학생4', '학생5'],
    '키': [170, 165, 180, 175, 160],
    '몸무게': [65, 55, 80, 70, 50],
    '중간고사': [75, 60, 90, 85, 45],
    '기말고사': [80, 65, 85, 87, 50],
    '총점': [78.0, 63.5, 88.0, 86.2, 47.5],
    '월소비금액': [500000, 300000, 700000, 600000, 250000]
}

df = pd.DataFrame(data)
df=df.set_index('이름')

#1. 상관관계 해석
# r값이 1이랑 가까울 수록 더 상관관계가 있다.
# ex. 중간고사 <-> 총점 상관계수 : 0.9956 -> '중간고사 점수가 오르면 총점도 오른다'라는 해석
corr_matrix=df.corr()
print(corr_matrix)

#2. 히트맵 시각화
matplotlib.rc('font', family='Malgun Gothic')
# annot=True : 숫자가 보이게
# cmap='coolwarm' : 붉은-푸른 대비
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('변수 간 상관관계 히트맵')
plt.show()
plt.savefig('ex1.png')

#3. 특정 변수 기준 정렬
# 총점이랑 연관성 높은 순서대로
df.corr()['총점'].sort_values(ascending=False)
print(corr_matrix['총점'].sort_values(ascending=False))
#총점에 영향 미치는 요인
#총점 <-> 기말고사의 상관계수 0.9956 -> 매우 큰 상관관계.
#총점 <-> 중간고사의 상관계수 0.9946 -> 매우 큰 상관관계
#기말고사, 중간고사 점수가 높아질수록 총점도 커진다.