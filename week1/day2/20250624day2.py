#1번. 사용자로부터 나이를 입력받고, 다음 조건에 따라 문장 출력
#0세 미만: “아기"

#0~12세: “어린이”

#13~18세: “청소년”

#19~64세: “성인”

#65세 이상: “노인”
age = int(input('나이 입력: '))

if age < 0:
    print('세상에 없음')
elif age < 12:
    print('어린이')
elif age < 18:
    print('청소년')
elif age < 64:
    print('성인')
else :
    print('노인')


#2. 1~100까지 중에 3의 배수이면서 5의 배수인 것 출력
for i in range(1,101):
    if i%3 == 0 and i%5==0:
        print(i)

#3. icd_list = ['E11', 'I10', 'J18', 'E11', 'I10', 'I10', 'A41'] 이 리스트에서 코드 별 등장 횟수 정리
icd_list = ['E11', 'I10', 'J18', 'E11', 'I10', 'I10', 'A41']
for code in set(icd_list):
    print(f'{code} = {icd_list.count(code)}')

#4. ICD 코드가 'E11'이면 '당뇨', 'I10'이면 '고혈압', 'J18'이면 '폐렴'으로 바꾸는 함수를 짜라.
def icd_name(icd_list):
    if icd_list == 'E11':
        return '당뇨'
    elif icd_list == 'I10':
        return '고혈압'
    elif icd_list == 'J18':
        return '폐렴'
    else:
        return '기타'

icd_name('E11')

def my_sum(n):
    s=0
    for e in range(n):
        s=s+(e+1)
    print(s)
my_sum(10)