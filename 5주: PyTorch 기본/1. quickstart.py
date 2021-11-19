"""
Quickstart - 머신러닝의 기본적인 API를 익혀보자!
===================
PyTorch has two `primitives to work with data <https://pytorch.org/docs/stable/data.html>`_:
``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.
``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
the ``Dataset``.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

######################################################################
# Torchvision 토치비전, Torchaudio 토치오디오, Torchtext 토치텍스트 등의 교육용 데이터셋을 받아서 실습해볼 수 있다.

# 토치비전의 데이터셋에서 FashionMNIST 를 다운받는다 - 테스트용 데이터와 검증용 데이터가 따로 있다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


######################################################################
# "데이터셋"을 "데이터로더"에 매개변수로 전달한다.
# batch 크기, 샘플링, 셔플링, 멀티프로세싱(병렬연산)을 지원한다.

batch_size = 64

# 매개변수에 값을 넣어서 "데이터로더"를 만든다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape) # shape란? 행렬의 모양새, 차원을 의미함. (예: 200x200 사진을 1x40000 배열 shape로 변형해서 학습 가능)
    print("Shape of y: ", y.shape, y.dtype)
    break

# 데이터로더 더 자세히 알아보기: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html



######################################################################
# 모델 만들기
# 신경망(neural network, nn)을 만들기 위해 상속시킬 클래스를 만든다.
# nn 모듈 더 자세히 알아보기: https://pytorch.org/docs/stable/generated/torch.nn.Module.html

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu" # 가능하면 GPU, 없으면 CPU
print(f"Using {device} device")

# 모델 정의하기
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__() # 상속받은 클래스 먼저 init 하고
        self.flatten = nn.Flatten() # 납작한 레이어 하나
        self.linear_relu_stack = nn.Sequential( # relu stack 으로 레이어 3개 (레이어 3개 사이를 ReLU 신경으로 이동함)
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x): # 학습단계
        x = self.flatten(x) # 매개변수 x를 납작이 레이어에 넣고
        logits = self.linear_relu_stack(x) # 그걸 다시 relu stack에 넣어서
        return logits # 나온 결과를 반환!

model = NeuralNetwork().to(device) # 만들어낸 모델을 특정 디바이스에 물림
print(model)

# 모델 만들기 더 자세히 알아보기: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html


#####################################################################
# 매개변수 최적화하기
# 모델을 학습시키기 위해, 손실 함수(https://pytorch.org/docs/stable/nn.html#loss-functions)와
# 옵티마이저(https://pytorch.org/docs/stable/optim.html) 를 설정한다
# 옵티마이저: https://gomguard.tistory.com/187

loss_fn = nn.CrossEntropyLoss() # 이번 예제에서는 cross entroy 손실함수랑
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # SGD 옵티마이저를 사용해보자


#######################################################################
# 훈련 루프 1바퀴에서, 모델은 training 데이터셋에 대해 예측을 수행하고, 예측 오류를 역전파해 모델의 매개변수를 조정한다.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

##############################################################################
# test 데이터셋과 비교해, 모델의 성능을 확인해 보자 - 학습이 제대로 되고 있나 확인해보자

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

##############################################################################
# training 과정을 여러 번 반복(epochs)한다.
# 각 epoch 동안 모델은 매개변수를 학습한다. 각 epoch마다 모델의 정확도와 손실을 출력해보자.
# 목표: epoch마다 정확도는 증가하고, 손실이 감소하기를...!

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# 모델 최적화 더 자세히 알아보기: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html


######################################################################
# 학습한 모델 저장하기
# 보통은 state_dict 형태로 저장한다

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



######################################################################
# 저장한 모델 다시 로딩하기
# 모델 구조를 다시 만들고, state_dict 를 불러오면 된다.

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

#############################################################
# This model can now be used to make predictions.

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


# 세이브 & 로드 더 자세히 알아보기: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
