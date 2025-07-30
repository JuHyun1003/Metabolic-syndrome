# 1. 범주형 변수 처리
## map() 사용
## pd.get_dummies() 사용(One-Hot Encoding)

#2. feature scaling
## MinMaxScaler
## StandardScaler

import pandas as pd

data = {
    '이름': ['민수', '서연', '지훈', '예지', '동현', '지우', '수빈', '준호'],
    '성별': ['남', '여', '남', '여', '남', '여', '여', '남'],
    '전공': ['생명과학', '경영학', '심리학', '화학', '물리학', '컴공', 'AI', '생명과학'],
    '나이': [21, 22, 20, 23, 21, 22, 21, 20],
    '수면시간': [6.5, 7.0, 5.0, 8.0, 6.0, 5.5, 7.5, 4.0],
    '하루공부시간': [2.0, 3.5, 1.0, 4.0, 2.5, 1.5, 3.0, 0.5],
    'SNS사용시간': [3.0, 1.0, 4.0, 0.5, 2.0, 5.0, 1.0, 6.0],
    '중간': [75, 82, 65, 90, 70, 60, 85, 55],
    '기말': [80, 88, 70, 93, 76, 66, 90, 50],
    '과제': [85, 90, 75, 95, 80, 70, 92, 60]
}

df = pd.DataFrame(data)

# 총점 계산 (중간30 + 기말40 + 과제30)
df['총점'] = df['중간'] * 0.3 + df['기말'] * 0.4 + df['과제'] * 0.3

# 학점 부여 (절대평가)
def grade(score):
    if score >= 90:
        return 'A+'
    elif score >= 85:
        return 'A0'
    elif score >= 80:
        return 'A-'
    elif score >= 75:
        return 'B+'
    elif score >= 70:
        return 'B0'
    elif score >= 65:
        return 'C+'
    elif score >= 60:
        return 'C0'
    else:
        return 'F'

df['학점'] = df['총점'].apply(grade)

df['A등급여부'] = df['학점'].isin(['A+', 'A0', 'A-']).astype(int)
df



from sklearn.preprocessing import StandardScaler

# 범주형 처리
df['성별코드'] = df['성별'].map({'남': 0, '여': 1})
df = pd.get_dummies(df, columns=['전공'], drop_first=True)
df.info()
df

#수치형 스케일링
scaler = StandardScaler()
df[['수면시간', '하루공부시간', 'SNS사용시간', '나이']] = scaler.fit_transform(
    df[['수면시간', '하루공부시간', 'SNS사용시간', '나이']]
)
df.describe()

df.head()

# 입력변수와 라벨 나누기
x = df.drop(columns=['이름', '성별', '중간', '기말', '과제', '총점', '학점', 'A등급여부'])  # 필요없는 거 제거
y= df['A등급여부']  # 맨 끝에 붙여서 확인용

print(x.head())
y.head()