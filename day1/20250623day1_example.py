#1. 과일 목록에서 "딸기"가 있는지 확인하고 출력
# fruits = ["apple", "banana", "딸기", "귤"]
fruits = ["apple", "banana", "딸기", "귤"]
if "딸기" in fruits:
    print('딸기 있음')
else:
    print('딸기 없음')


#2 60점 이상이면 '합격', 아니면 '불합격'
# score = {"철수": 80, "영희": 55, "민수": 100, "지수": 30}
score = {"철수": 80, "영희": 55, "민수": 100, "지수": 30}
for name in score:
    if score[name] >= 60:
        print(name,': 합격')
    else:
        print(name, ': 불합격')


#3 리스트 안에 있는 과일 개수 세기
basket = ["apple", "banana", "apple", "apple", "귤", "banana"]
for name in set(basket):
    print(f'{name} : {basket.count(name)}개')

for name in basket:
    print('사과 :', basket.count('apple'),'개')
    print('바나나 :', basket.count('banana'),'개')
    print('귤 :', basket.count('귤'),'개')


#4 1부터 10까지 짝수면 '짝수', 홀수면 '홀수'라고 출력
for i in range(1,11,1):
    if i%2==1:
        print(f'{i}는 홀수입니다')
    else:
        print(f'{i}는 짝수입니다')


#5 이름과 나이 출력
people = [("철수", 23), ("영희", 21), ("민수", 25)]
for name, age in people:
    print(f'{name}은 {age}살입니다')


#6 2차원 리스트 출력
A=[[0 for j in range(3)] for i in range(2)]
A[0][0]='A'; A[0][1]='B'; A[0][2]='C'
A[1][0]='D'; A[1][1]='E'; A[1][2]='F'

for i in range(2):
    for j in range(3):
        print(f'i={i} j={j}, 값={A[i][j]}')