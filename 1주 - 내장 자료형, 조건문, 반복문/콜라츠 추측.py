# 콜라츠 추측 계산기
# 1부터 MAXnum 까지의 콜라츠 수열을 계산한다.
# 제한조건 : print 함수는 numlist 를 출력하는 데에만 사용할 수 있다.

MAXnum = 10000

for i in range(1, MAXnum+1):
    numlist = []
    num = i
    numlist.append(num)
    while num != 1:
        if num % 2 == 0:
            num = int(num / 2)
        else:
            num = num*3 + 1
        numlist.append(num)
    print(numlist)
