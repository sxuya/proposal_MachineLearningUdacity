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
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

i_input_tensor = Input((img_height, img_width, 3))
# x_input_tensor = inception_resnet_v2.preprocess_input(i_input_tensor)	# 错误的处理
x_input_tensor = Lambda(inception_resnet_v2.preprocess_input)(i_input_tensor)

base_model = inception_resnet_v2.InceptionResNetV2(input_tensor=x_input_tensor, weights='imagenet', include_top=False)

for layers in base_model.layers:
    layers.trainable = False

i_output = GlobalAveragePooling2D()(base_model.output)
i_output = Dropout(0.25)(i_output) # 0.25 比 0.5 好
i_predictions = Dense(1, activation='sigmoid')(i_output)
model = Model(base_model.input, i_predictions)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=20,     # defalt 16
	epochs=8, 	# defalt 5
	validation_data=(X_valid, y_valid))

for i in range(len(model.layers)):
    print(model.layers[i].name, i)

for layer in model.layers[770:]:	# total 783 layers
    layer.trainable = True


from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
# 第一次没有进行上面的compile，程序报错也没有仔细看，第二天详细看了弹出的红色的提示后，
# 在回看了ypw的融合的例子的情况，然后添加了上面这个
# 然后结果就好很多，验证集上达到loos：0.038；acc：0.9908

model.fit(X_train, y_train, batch_size=20, 	# first: 16
	epochs=8, 	# first: 5 
	validation_data=(X_valid, y_valid))

model.save('modelInceptionResNet_transfer_0_07.h5')

X_test = np.zeros((n, img_height, img_width, 3), dtype=np.uint8)
for i in tqdm(range(12500)):
    j = i+1
    X_test[i] = cv2.resize(cv2.imread('test/%d.jpg' % j), (img_height, img_width))
#     X_test[i] = cv2.resize(cv2.imread('test/%d.jpg' % j, (img_height, img_width))) the wrong code


y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)

import pandas as pd
df = pd.read_csv("sample_submission.csv")

for i in range(12500):
	df.set_value(i, 'label', y_pred[i])

df.to_csv('pred.csv', index=None)


########
# def rmrf_mkdir(dirname):
#     if os.path.exists(dirname):
#         shutil.rmtree(dirname)
#     os.mkdir(dirname)

# import os
# rmrf_mkdir('testRun')
# os.symlink('../test/', 'testRun/test')

# import pandas as pd
# from keras.preprocessing import image

# df = pd.read_csv("sample_submission.csv")

# image_size = (224, 224)
# gen = image.ImageDataGenerator()
# test_generator = gen.flow_from_directory("testRun", image_size, shuffle=False, 
#                                          batch_size=16, class_mode=None)

# for i, fname in enumerate(test_generator.filenames):
#     index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
#     df.set_value(index-1, 'label', y_pred[i])
########

df.to_csv('pred.csv', index=None)
df.head(10)