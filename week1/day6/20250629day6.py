import pandas as pd
import matplotlib.pyplot as plt
data = {
    '이름': ['지훈', '민서', '서준', '예은'],
    '성별': ['남', '여', '남', '여'],
    '중간': [80, 90, 70, 60],
    '기말': [85, 95, 75, 65],
    '과제': [90, 80, 100, 95]
}
df = pd.DataFrame(data)

#apply() : 함수를 집어넣기 위한 코드
#axis =1 : 행 기준으로 처리. 반드시 넣어줘야 함
def calc_score(row):
    return row['중간']*0.4 + row['기말']*0.4 + row['과제']*0.2
df['총합']=df.apply(calc_score, axis=1)

#map() : 문자열 값을 숫자나 다른 값으로 바꾸어줌
#map()은 Series에만 사용한다
#df['성별'].map() : O
#df.map() : X
df['성별코드']=df['성별'].map({'남' : 0, '여' : 1})
df

#value_counts(normalize = True)
# 카테고리 분포를 볼 때 사용한다. 비율로써 제시됨
print(df['성별'].value_counts(normalize=True))

#간단한 시각화
#.plot()사용
df['중간'].plot(kind='hist')
plt.show()
plt.savefig('ex1.png')
df['성별코드'].value_counts().plot(kind='bar')
plt.show()
plt.savefig('ex2.png')
