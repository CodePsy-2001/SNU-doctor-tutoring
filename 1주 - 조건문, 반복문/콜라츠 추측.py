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

# 콜라츠 추측 계산기