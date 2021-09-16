import zipfile

zip_ref = zipfile.ZipFile("all.zip", 'r')
zip_ref.extractall(".")
zip_ref.close()

zip_ref = zipfile.ZipFile("train.zip", 'r')
zip_ref.extractall(".")
zip_ref = zipfile.ZipFile("test.zip", 'r')
zip_ref.extractall(".")
zip_ref.close()

import os
import shutil

train_filenames = os.listdir('train')
train_cat = filter(lambda x:x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x:x[:3] == 'dog', train_filenames)

def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    
rmrf_mkdir('trainRun')
os.mkdir('trainRun/cat')
os.mkdir('trainRun/dog')

rmrf_mkdir('testRun')
os.symlink('../test/', 'testRun/test')

for filename in train_cat:
    os.symlink('../../train/'+filename, 'trainRun/cat/'+filename)

for filename in train_dog:
    os.symlink('../../train/'+filename, 'trainRun/dog/'+filename)


#############################################################################


from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image

import h5py

base_model = InceptionResNetV2(weights='imagenet', include_top=False)
model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

img_width, img_height = 299, 299
train_datagen = image.ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory("trainRun", target_size=(img_height, img_width), shuffle=False, batch_size=16)
test_generator = test_datagen.flow_from_directory("testRun", target_size=(img_height, img_width), shuffle=False, batch_size=16, class_mode=None)

train_h5py = model.predict_generator(train_generator)
test_h5py = model.predict_generator(test_generator)

with h5py.File("init_weights_InceptionRestNetV2.h5") as h:
        h.create_dataset("train", data=train_h5py)
        h.create_dataset("test", data=test_h5py)
        h.create_dataset("label", data=train_generator.classes)

import numpy as np
from sklearn.utils import shuffle
np.random.seed(2017)

X_train = []
X_test = []

with h5py.File("init_weights_InceptionRestNetV2.h5", 'r') as h:
    X_train.append(np.array(h['train']))
    X_test.append(np.array(h['test']))
    y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)

from keras.layers import Input, Dropout, Dense
np.random.seed(2017)

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model_run = Model(input_tensor, x)

model_run.compile(optimizer='adadelta',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

model_run.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2)

y_pred = model_run.predict(X_test, verbose=1)
# y_pred = y_pred.clip(min=0.003, max=0.997)

import pandas as pd

df = pd.read_csv("sample_submission.csv")

test_run_datagen = image.ImageDataGenerator()
test_run_generator = test_run_datagen.flow_from_directory("testRun", 
                                                          (img_height, img_width), 
                                                         shuffle=False, 
                                                         batch_size=16, 
                                                         class_mode=None)

for i, fname in enumerate(test_run_generator.filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv('pred_IceRNV2.csv', index=None)
df.head(10)