from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.preprocessing.image import *

import numpy as np
from tqdm import tqdm
import cv2

np.random.seed(2017)

n = 25000
X = np.zeros((n, 299, 299, 3), dtype=np.uint8)
y = np.zeros((n, 1), dtype=np.uint8)

for i in tqdm(range(n/2)):
    X[i] = cv2.resize(cv2.imread('train/cat.%d.jpg' % i), (299, 299))
    X[i+n/2] = cv2.resize(cv2.imread('train/dog.%d.jpg' % i), (299, 299))

y[n/2:] = 1

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# 下面内容为ypw内容，如果不行，按照keras文章在调整；
# 如果还是不行，则进行标签内的keras文章进行修改

base_model = InceptionResNetV2(input_tensor=Input((299, 299, 3)), weights='imagenet', include_top=False)

for layers in base_model.layers:
    layers.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.25)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, x)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 应该有比较高的acc
model.fit(X_train, y_train, batch_size=16, epochs=5, validation_data=(X_valid, y_valid))

for layer in model.layers[140:]:
    layer.trainable = True

model.fit(X_train, y_train, batch_size=16, epochs=5, validation_data=(X_valid, y_valid))

