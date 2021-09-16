# MachineLearningUdacity
## dog_vs_cat

猫狗的数据、评价信息，可以见 [kaggle dog_vs_cat](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) 页面。

## 运行环境
- p2.xlarge on AWS
- Python 3.6.0
- Tensorflow-gpu
- keras
- sklearn
- jupyter notebook

## 主要文件

可视化代码：visualization.ipynb

模型代码：model_transfer_InceptionRestNetV2_succefull_loss0-056.ipynb

报告文件：report.pdf


## AWS 设置

- 进入 tensorflow_p36 环境进行操作
- 安装 Pillow，tqdm，scikit-learn
- 卸载默认的 tnesorflow，重新安装 Tensorflow-gpu
- 其他步骤参考[这篇文章](https://zhuanlan.zhihu.com/p/33176260)

## 模型
- InceptionResNetV2

## 中间文件

模型第一次 fit 训练产生的 h5 文件可以通过下面链接下载：

链接：https://pan.baidu.com/s/1G9B34w1TDAQdeBH6jSIEbg 密码：el03

## 结果

模型一共需要运行30分钟左右.

![result](https://note.youdao.com/yws/public/resource/5ec4832fbee107e8f067dd386dd72a8a/xmlnote/01EBD793AC624717B8E21F8B2C6A8536/21705)

![result2](https://note.youdao.com/yws/public/resource/5ec4832fbee107e8f067dd386dd72a8a/xmlnote/2B8ED995EEE746758A940F1C9CE18A8B/21708)