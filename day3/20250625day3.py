import pandas as pd

data={
    '이름' : ['주현', '진혁', '기혁','민수','민재','호영','현우','주찬','호빈','남섭'],
    '나이' : [22, 21, 25, 27, 28, 24, 23, 28, 26, 24],
    '점수' : [85, 79, 67, 91, 55, 67, 82, 95, 42, 75]
}
df=pd.DataFrame(data)
print(df)

#1 기본 정보
print(df.head())#위 5개
print(df.tail())#아래 5개
print(df.info())#열에 따른 기본 정보들
print(df.describe()) #각종 정보들
print(df.shape)#(행(세로), 열(가로))
print(df.columns)#열 이름

#2 열 선택
print(df['이름']) #'이름'열 출력
print(df[['이름', '점수','나이']]) #여러 열 선택

#3 행 선택
# loc vs iloc
# loc[n] : index값이 n인 행 출력
# iloc[n] : n번째 행 출력
print(df.loc[0])
print(df.iloc[0])
print(df.loc[1:3])

print(df.loc[2])
print(df.iloc[2])

#4 조건 필터링
print(df[df['나이'] > 23])
print(df[df['점수'] >= 80])

#5 새로운 열 추가 & 삭제
df['합격여부'] = df['점수'] >= 80 #해당 조건에 만족하는 값은 True / 만족 안하면 False
print(df)

del df['합격여부'] #새로 만든 '합격여부'열의 삭제
print(df)

#6 정렬
print(df.sort_values(by='나이'))
print(df.sort_values(by='점수', ascending=False))

#7 인덱스 조작
df=df.set_index('이름') #index를 이름으로 변환
print(df)
print(df.iloc[0])

df=df.reset_index()
print(df)