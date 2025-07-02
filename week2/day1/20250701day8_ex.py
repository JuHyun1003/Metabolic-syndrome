import pandas as pd

data = {
    '고객ID': ['user001', 'user002', 'admin001', 'guest001', 'user003'],
    '이름': ['김수현', '이민정', '박정우', '최유리', '정윤호'],
    '이메일': ['soohyun@gmail.com', 'lee@naver.com', 'jungwoo@kakao.com', 'yuri@daum.net', 'yunho@naver.com'],
    '가입일': ['2024.01.01', '2024.03.15', '2024.04.20', '2024.05.01', '2024.06.30']
}

df = pd.DataFrame(data)

#1. 이메일 도메인 추출
df['도메인']=df['이메일'].str.split('@').str[1]
df

#2. 사용자 유형 추출
# r'([a-zA-Z]+)' : 알파벳으로만 이루어진 연속된 부분 추출
df['유형'] = df['고객ID'].str.extract(r'([a-zA-Z]+)')
df

#3. 가입일 datetime으로 변환
df['가입일']=pd.to_datetime(df['가입일'])

#4. 가입한 요일 추출
df['요일']=df['가입일'].dt.day_name()
df

#5. 주말여부 출력
df['주말여부']=df['가입일'].dt.weekday >= 5
df['주말여부']=df['주말여부'].map({True : '주말', False : '평일'})
df

#6. 네이버 사용자만 필터링
df[df['도메인'].str.contains('naver')]