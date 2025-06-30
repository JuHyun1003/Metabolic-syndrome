import pandas as pd
import numpy as np
data={
    '이름' : ['주현', '진혁', '기혁','민수','민재','호영','현우','주찬','호빈','남섭'],
    '나이' : [22, 21, 25, 27, 28, 24, 23, 28, 26, 24],
    '점수' : [85, 79, 67, 91, 55, 67, 82, 95, 42, 75]
}
df=pd.DataFrame(data)

df['합격 여부']=np.where(df['점수']>=80, '합격', '불합격')
print(df)
print(df[['이름', '합격 여부']])

print(df[df['합격 여부']=='합격'])

#원본 데이터(아래 코드는 모두 실패함)
dcell = pd.read_excel(r'C:\Users\kgh44\Desktop\경희대\2025년(2학년)\2-1\세포생물학1\중간고사.xlsx')
print(dcell)
print(dcell.info)
print(dcell.describe())
cutoff=int(len('총합')*0.45)
print(dcell)
dcell['등급']=np.where(dcell['총합'] < cutoff, 'A','B')
print(dcell)
del dcell['등급']
len('총합')
#가공하기 위한 수정 데이터
dcell_sorted=dcell.sort_values(by='총합', ascending=False)
n=len(dcell_sorted)
cut_aplus= int(n*0.15)
cut_azero= int(n*0.30)
cut_aminus= int(n*0.45)
print(dcell_sorted)
dcell_sorted['등급'] = np.where(dcell_sorted['총합'] < cut_aplus, 'A+',
                              dcell_sorted['총합'] < cut_azero, 'A0',
                              dcell_sorted['총합'] < cut_aminus,'A-')
#오류 이유 : np.where(조건, 참, 거짓) 이렇게 3개 인자만 받는다.
#해결 방안
dcell_sorted['학점'] = np.where(dcell_sorted['총합'] < cut_aplus, 'A+',
                              np.where(dcell_sorted['총합'] < cut_azero, 'A0',
                                       np.where(dcell_sorted['총합'] < cut_aminus, 'A-','B')))
print(dcell_sorted)
print(dcell_sorted['총합'].dtype)
#이러니까 밑에서부터 높은 학점으로 들어감
#dcell_sorted['총합'] < cut_aplus 이런 식은 말 그대로 '총합' 점수가 15점 이하면 A+라고 인식하게됨. 따라서 인덱스로 바꿔야함.
dcell_sorted['학점'] = np.where(dcell_sorted.index < cut_aplus, 'A+',
                              np.where(dcell_sorted.index < cut_azero, 'A0',
                                       np.where(dcell_sorted.index < cut_aminus, 'A-','B')))
print(dcell_sorted)
print(dcell_sorted[dcell_sorted['학점']=='A+'])
print(dcell_sorted[dcell_sorted['학점']=='A0'])
print(dcell_sorted.iloc[13:29])