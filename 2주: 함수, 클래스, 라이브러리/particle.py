
# 이데아
class Atom():
    # 초기화
    def __init__(self, place=0, velocity=0, acc=0):
        self.place = place
        self.velocity = velocity
        self.acc = acc

    # 가속도
    def accel(self):
        self.velocity += self.acc

    # 움직임
    def move(self):
        self.accel()
        self.place += self.velocity



class Human(Atom): # 상속, inheritance

    def __init__(self, place=0, velocity=0, acc=0, name="James", mass=60):
        super().__init__(place, velocity, acc)
        self.mass = mass
        self.name = name

    def die(self): # 단말마
        print(f"{self.name} 사망했다...")

    def energy(self): # 에너지
        return 0.5 * self.mass * (self.velocity ** 2)

    def crash(self): # 충돌
        print("땅바닥에 닿았다!")

        if self.energy() >= 5000:
            self.die()

        self.acc = 0
        self.velocity = 0


    def checkCrash(self): # crash 메소드의 실행조건 체크
        if self.place <= 0:
            self.crash()


    def move(self): # 1 단위마다 실행되는 함수
        super().move()
        self.checkCrash()



if __name__ == "__main__":

    humanA = Human(place=8849, velocity=1, acc=-0.98, name="철수") # 에베레스트
    humanB = Human(place=6, velocity=0, acc=-0.98, name="영희") # 2층
    
    
    time = 0
    while humanA.place >= 0:
        time += 0.1
        humanA.move()
        print(f"현재위치: {humanA.place}")
    
    print(f"총 걸린 시간: 약{time}초")
    
    print("=" * 10)
    
    time = 0
    while humanB.place >= 0:
        time += 0.1
        humanB.move()
        print(f"현재위치: {humanB.place}")
    
    print(f"총 걸린 시간: 약{time}초")