# OpenNRE-PyTorch

An open-source framework for neural relation extraction implemented in PyTorch.

Contributed by [Shulin Cao](https://github.com/ShulinCao), [Tianyu Gao](https://github.com/gaotianyu1350), [Xu Han](https://github.com/THUCSTHanxu13), [Lumin Tang](https://github.com/Tsingularity), [Yankai Lin](https://github.com/Mrlyk423), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/)

## Overview

It is a PyTorch-based framwork for easily building relation extraction models. We divide the pipeline of relation extraction into four parts, which are embedding, encoder, selector and classifier. For each part we have implemented several methods.

* Embedding
  * Word embedding
  * Position embedding
  * Concatenation method
* Encoder
  * PCNN
  * CNN
* Selector
  * Attention
  * Maximum
  * Average
* Classifier
  * Softmax loss function
  * Output
  
All those methods could be combined freely. 

We also provide fast training and testing codes. You could change hyper-parameters or appoint model architectures by using Python arguments. A plotting method is also in the package.

This project is under MIT license.

## Requirements

- Python (>=2.7)
- PyTorch (==0.3.1)
- CUDA (>=8.0)
- Matplotlib (>=2.0.0)
- scikit-learn (>=0.18)

## Installation

1. Install PyTorch
2. Clone the OpenNRE repository:
  ```bash
  git clone https://github.com/ShulinCao/OpenNRE-PyTorch
  ```
3. Download NYT dataset from [Google Drive](https://drive.google.com/file/d/1g95gbMUsGfeEmihZSb0kXPbMTuRA4lid/view?usp=sharing)
4. Extract dataset to `./raw_data`
```
unzip raw_data.zip
```
## Dataset

### CCKS 2019人物关系抽取数据集

下载[链接](https://www.biendata.com/competition/ccks_2019_ipre/)

- 远程监督数据集，不精确
- 句子已分词，有标点

### 数据集错误示例
- 外国人名只取了一部分：TRAIN_SENT_ID_000856├──┤米莉├──┤燕妮├──┤朵 依 燕妮 则 以 “ 玛琪 ” 来 称呼 玛琪 米莉 耶 。

### Training Data & Testing Data

### 词嵌入

网上下载的微博词嵌入。19万词，300维向量。含标点、人名、特殊符号等，按词频降序排序。

首行是个数与维度信息，一行一个词，每行301列以空格分隔，第一列为词，后300列为向量浮点值(-1.0, 1.0)，样例如下。
```
195202 300
， 0.094386 -0.200944 xx xx
的 -0.124044 -0.053688 xx xx
```

### Relation-ID Mapping Data

This file indicates corresponding IDs for relations to make sure during each training and testing period, the same ID means the same relation. Its format is as follows

```
{
    'NA': 0,
    'relation_1': 1,
    'relation_2': 2,
    ...
}
```

**IMPORTANT**: Make sure the ID of `NA` is always 0.

## Quick Start

### Process Data

```bash
python gen_data.py
```
The processed data will be stored in `./data`

### Train Model
```
python train.py --model_name pcnn_att
```

The arg `model_name` appoints model architecture, and `pcnn_att` is the name of one of our models. All available models are in `./models`. About other arguments please refer to `./train.py`. Once you start training, all checkpoints are stored in `./checkpoint`.

### Test Model
```bash
python test.py --model_name pcnn_att
```

Same usage as training. When finishing testing, the best checkpoint's corresponding pr-curve data will be stored in `./test_result`.

### Plot
```bash
python draw_plot.py PCNN_ATT
```

The plot will be saved as `./test_result/pr_curve.png`. You could appoint several models in the arguments, like `python draw_plot.py PCNN_ATT PCNN_ONE PCNN_AVE`, as long as there are these models' results in `./test_result`.

## Build Your Own Model

Not only could you train and test existing models in our package, you could also build your own model or add methods to the four basic modules. When adding a new model, you could create a python file in `./models` having the same name as the model and implement it like following:

```python
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.embedding import *
from networks.encoder import *
from networks.selector import *
from networks.classifier import *
from .Model import Model
class PCNN_ATT(Model):
  def __init__(self, config):
    super(PCNN_ATT, self).__init__(config)
    self.encoder = PCNN(config)
    self.selector = Attention(config, config.hidden_size * 3)
```

Then you can train, test and plot!

