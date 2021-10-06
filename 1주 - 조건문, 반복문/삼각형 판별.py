
tests = [
    [60, 60, 60],
    [30, 60, 90],
    [20, 40, 120],
    [0, 90, 90],
    [60, 70, 80],
    [40, 40, 50, 50]
]


def triangle(angles: list) -> int:
    if len(angles) != 3: return 0
    if sum(angles) != 180: return 0

    m = max(angles)

    if m < 90: return 1
    if m == 90: return 2
    if m > 90: return 3


for angles in tests:
    result = triangle()
    names = {0:"삼각형이 아님", 1:"예각삼각형", 2:"직각삼각형", 3:"둔각삼각형"}

    print(angles, names[result])
