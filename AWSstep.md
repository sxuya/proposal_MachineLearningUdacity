1. 配置 putty

	两篇文章的出入站规则都建立。
	```
	screen -S jupyter(仅仅是标记用的名字)
	jupyter notebook --ip:0.0.0.0
	```

2. download zip file


```
import urllib

~~testfile = urllib.URLopener()~~
testfile.retrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/sample_submission.csv", "sample_submission.csv")
testfile.retrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/test.zip", "test.zip")
testfile.retrieve("https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/train.zip", "train.zip")
```
~~no use~~

	"""
	This downloads a file from a website and names it ***.**
	"""
3. classify the images to different fold

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

