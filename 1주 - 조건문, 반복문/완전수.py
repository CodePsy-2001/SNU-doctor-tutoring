MAX = 500
result = []

for num in range(1, MAX+1):

    sum = 0
    for i in range(1, num):
        if num % i == 0:
            sum += i

    if num == sum:
        result.append(num)

print(result)
