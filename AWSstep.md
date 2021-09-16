1. 配置 putty

	

1. download zip file

```
import urllib.request

urllib.request.urlretrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/sample_submission.csv", "sample_submission.csv")
urllib.request.urlretrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/test.zip", "test.zip")
urllib.request.urlretrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/train.zip", "train.zip")
```

the failed one processing
```
import urllib

testfile = urllib.URLopener()
testfile.retrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/sample_submission.csv", "sample_submission.csv")
testfile.retrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/test.zip", "test.zip")
testfile.retrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/train.zip", "train.zip")
```

	"""
	This downloads a file from a website and names it ***.**
	"""

1. unzip the zipFile

```
import zipfile

zip_ref = zipfile.ZipFile("all.zip", 'r')
zip_ref.extractall(".")
zip_ref.close()
```

1. classify the images to different fold

```
import os
import shutil

train_filenames = os.listdir('train')
train_cat = filter(lambda x:x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x:x[:3] == 'dog', train_filenames)
```

```
def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
```

```
rmrf_mkdir('train2')
os.mkdir('train2/cat')
os.mkdir('train2/dog')

rmrf_mkdir('test2')
os.symlink('../test', 'test2/test')

for filename in train_cat:
    os.symlink('../../train/'+filename, 'train2/cat/'+filename)

for filename in train_dog:
    os.symlink('../../train/'+filename, 'train2/dog/'+filename)
```

```
上述需要修改。
```

