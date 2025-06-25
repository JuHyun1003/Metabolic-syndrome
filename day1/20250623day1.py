# for 문
fruits=["apple","banana"]
for b in fruits:
    print(b)
#1 1부터 10까지 더하기
s=0
for i in range(1,11,1):
    s=s+i
    print(i,s)

#2 1부터 10까지 짝수만 더하기
s=0
for i in range(2,12,2):
    s=s+i
    if i==10:
        print(s)
#2-1
s=0
for i in range(0,1000,2):
    i=i+2
    s=s+i
    if i==10:
        print(i,s)
        break

#3 1~3까지, 1~5까지 모든 조합 print
for i in range(1,4):
    for j in range(1,6):
        print(i,j)
#While문
#1 1~10까지 짝수만 출력
s=0
i=0
while i<10:
    i=i+2
    s=s+i
    print(i,s)
#IF문
score = {"철수": 80, "영희": 55}
for name in score:
    if score[name] >= 60:
        print(name, score[name])

a=1
b=2
c=3
if a>b:
    print('a is bigger than b')
if a<b:
    print('a is smaller than b')
else:
    print('a is smaller or equal to b')

if a>b:
    print('a is smaller than b')
elif a<c:
    print('a is smaller or equal to b and a is smaller than c')
else:
    print('a is smaller or equal to b and a is bigger or equal to c')

#행렬
B=[[0 for j in range(3)] for i in range(2)]
print(B)

for i in range(2):
    for j in range(3):
        print(i,j,B[i][j])