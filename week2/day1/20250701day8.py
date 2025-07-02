import pandas as pd

data = {
    '이름': ['JiHoon', '박민서', '이서준', '최예은'],
    '이메일': ['jihoon@gmail.com', 'minseo@naver.com', 'seojoon@kakao.com', 'yeeun@daum.net'],
    '가입일': ['2025.07.15', '2025.06.20', '2025.06.08', '2025.06.01']
}
df = pd.DataFrame(data)
df
#1. 문자열 처리
df['이름'].str.upper() # 전부 대문자로 바꾸기
df['이름'].str.lower() # 전부 소문자로 바꾸기
df['이름'].str.strip() # 공백 제거
df['이메일'].str.contains('@') # @포함 여부
df['이메일'].str.split('@') # @를 기준으로 분리
df['이름'].str.replace('박','P.') # 문자 치환
df['이름'].str.startswith('J') # 시작 문자
#이름의 성만 출력
df['이름'].str[0]
#이메일의 도메인만 출력
df['이메일'].str.split('@').str[1]

#2. 날짜 데이터 처리
df['가입일']=pd.to_datetime(df['가입일'])
df
df['가입일'].dt.year
df['가입일'].dt.month
df['가입일'].dt.day
df['가입일'].dt.weekday # 0~6 : 차례대로, 월화수목금토일
df['가입일'].dt.day_name()

df[df['이메일'].str.contains('gmail') & (df['가입일']>'2025.05.01')]