import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# < 기본 이미지 분류 >
# https://www.tensorflow.org/tutorials/keras/classification?hl=ko


# MNIST 패션 데이터셋을 사용한다
# https://github.com/zalandoresearch/fashion-mnist
fashion_mnist = keras.datasets.fashion_mnist

# load_data() 함수를 호출하면 네 개의 numpy 배열을 반환한다.
# train~ : 모델 학습에 사용되는 훈련 셋트
# test~ : 모델 검증에 사용되는 테스트 셋트
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ["T-shirt", "Trouse", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(train_images.shape) # 60000, 28, 28 - 28x28 픽셀의 이미지 6만개
print(train_labels, len(train_labels)) # 레이블 6만개

plt.figure()
plt.imshow(train_images[31])
plt.colorbar()
plt.grid(False)
plt.pause(0.5) #plt.show(), plt.close()
# 픽셀별 값이 0부터 255까지 들어가있다는 걸 알 수 있다.
# 0 ~ 1 사이의 실수로 정규화해주자.

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.pause(0.5)

# 모델 구성 단계
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # 28x28 이미지 포맷을 1x784 1차원 배열로 변환
    # 데이터만 받아들이는 평평하게 펼쳐진 레이어 하나

    keras.layers.Dense(128, activation="relu"), # 밀집연결 densely-connected 층.
    # 128개의 노드를 가지고, relu 모양으로 데이터를 증폭시킴

    keras.layers.Dense(10, activation="softmax"), 
    # 10개의 노드를 가지고, 입력받은 값을 0~1 사이의 기울기로 정규화함 (활성화 함수)
    # => 각 노드가 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력함
])

# 모델 컴파일 단계
model.compile(
    loss='sparse_categorical_crossentropy', # 손실 함수 - 훈련 동안 모델의 오차를 측정함
    optimizer='adam', # 옵티마이저 - 데이터와 손실함수를 바탕으로 모델의 업데이트 방법을 결정함
    metrics=['accuracy'] # 지표 - 훈련단계와 테스트단계를 모니터링하기 위해 사용함. 여기서는 accuracy, 정확도 기준 모니터링.
    # 침고로 정확도 = 올바르게 분류된 이미지의 비율
)

# 모델 훈련 단계
model.fit(
    train_images,
    train_labels, # 훈련용 데이터 주입
    epochs=5
)

test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print("\n테스트 정확도:", test_accuracy)

# 테스트 셋트에서의 정확도가 훈련 셋트에서의 정확도보다 조금 낮음
# overfitting(과적합) - 훈련셋트에 과하게 적응해버려서 테스트 셋트(실제)에서 성능이 낮아짐


# 모델의 예상 결과를 행렬로 - 데이터는 그냥 test_images 를 다시 삽입했음.
predictions = model.predict(test_images)
print(len(predictions)) # 당연히 예측 결과는 6만개.
print(predictions[31], len(predictions[31])) # 그중 31번째 이미지의 예측 결과는 신뢰도값 10개
print("모델의 예상:", np.argmax(predictions[31]), "실제:", test_labels[31]) # 가장 예상

