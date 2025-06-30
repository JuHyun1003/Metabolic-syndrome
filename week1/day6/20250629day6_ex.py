import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
data = {
    '이름': ['콩이', '땅콩', '뽀삐', '망치', '떡순이', '철구'],
    '종류': ['고양이', '개', '토끼', '고양이', '돼지', '개'],
    '나이': [2, 5, 1, 4, 3, 7],  # 나이는 년 단위
    '일일사료량(kg)': [0.3, 0.8, 0.2, 0.4, 2.5, 1.2],
    '건강점수': [92, 85, 88, 79, 95, 67]  # 100점 만점
}

df = pd.DataFrame(data)
#1. 건강등급 붙이기(90 이상 : A, 80이상 : B, 70이상 : C, 그 외: D)
def grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    else:
        return 'D'

df['건강등급']=df['건강점수'].apply(grade)
df

#2. 동물의 종에 따른 코드 부여
df['종류'].value_counts()
df['종코드']=df['종류'].map({'고양이' : 0, '개' : 1, '토끼' : 2, '돼지' : 3})
df

#3. 건강등급을 비율로써 나타내라
print(df['건강등급'].value_counts(normalize=True))

#4. 건강점수를 히스토그램으로 나타내기
df['건강점수'].plot(kind='hist')
plt.show()

#5. 종류별 사료량 평균을 히스토그램으로 나타내기
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False     
df.groupby('종류')['일일사료량(kg)'].mean().plot(kind='bar', title='종 별 일일 사료량')
plt.show()
plt.savefig('ex3.png')