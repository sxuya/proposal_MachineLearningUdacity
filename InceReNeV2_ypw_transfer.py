import zipfile

zip_ref = zipfile.ZipFile("all.zip", 'r')
zip_ref.extractall(".")
zip_ref.close()

zip_ref = zipfile.ZipFile("train.zip", 'r')
zip_ref.extractall(".")
zip_ref = zipfile.ZipFile("test.zip", 'r')
zip_ref.extractall(".")
zip_ref.close()

from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout, Lambda
from keras.applications import inception_resnet_v2
# from keras.preprocessing.image import *

import numpy as np
from tqdm import tqdm
import cv2

np.random.seed(2017)

n = 25000
img_height, img_width = 299, 299
X = np.zeros((n, img_height, img_width, 3), dtype=np.uint8)
y = np.zeros((n, 1), dtype=np.uint8)

for i in tqdm(range(12500)): # n/2
    X[i] = cv2.resize(cv2.imread('train/cat.%d.jpg' % i), (img_height, img_width))
    X[i+12500] = cv2.resize(cv2.imread('train/dog.%d.jpg' % i), (img_height, img_width)) # n/2

y[12500:] = 1

from sklearn.model_selection import train_test_split
# 需要把集合打乱么？
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# from keras.applications.resnet50 import ResNet50

# base_model = ResNet50(input_tensor=Input((img_height, img_width, 3)), weights='imagenet', include_top=False)

i_input_tensor = Input((img_height, img_width, 3))
# x_input_tensor = inception_resnet_v2.preprocess_input(i_input_tensor)	# 错误的处理
x_input_tensor = Lambda(inception_resnet_v2.preprocess_input)(i_input_tensor)

base_model = inception_resnet_v2.InceptionResNetV2(input_tensor=x_input_tensor, weights='imagenet', include_top=False)
# 上面的 input_tensor 没有进行处理吧，参考里面的 ResNet50 是不用预处理的。
# 但是上面会找不到数据
# 下面的可以进行计算，但是结果。。。训练集还不错，loss：0.16；acc：0.93。但是验证集，loss：5.9197；acc：0.5622
# base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False)


for layers in base_model.layers:
    layers.trainable = False

i_output = GlobalAveragePooling2D()(base_model.output)
i_output = Dropout(0.5)(i_output)
i_predictions = Dense(1, activation='sigmoid')(i_output)
model = Model(base_model.input, i_predictions)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=5, validation_data=(X_valid, y_valid))
# 训练集还好，但是验证集就不好了。