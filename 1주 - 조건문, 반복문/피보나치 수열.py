n = 30

# 예제1: 행렬 연산으로 구현
import numpy as np
sequence = []
A = np.matrix( [ [1,1], [1,0] ] )

for i in range(n):
    sequence.append( (A**i)[0,1] )

print(sequence)


# 예제2: 일반식으로 구현
sequence = []
sqrt5 = 5 ** 0.5

for i in range(n):
    sequence.append(int(
        ( ((1+sqrt5)/2)**i  - ((1-sqrt5)/2)**i ) / sqrt5
    ))

print(sequence)
