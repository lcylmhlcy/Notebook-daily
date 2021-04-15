# 元学习
任务：少样本学习、单样本学习、零样本学习  
方法：元学习
  
元学习的类型：
- 学习度量空间
  - 孪生网络、原型网络、匹配网络、关系网络
- 学习初始化
  - NAML、Repitle、Meta-SGD
- 学习优化器

数据集以task划分 
Meta-train：支撑集、查询集  
Meta-test：支撑集、查询集  
  
### 学习梯度下降
- [NIPS2016] [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)
- [ICLR2017] [Optimization as a model for few-shot learning](https://openreview.net/pdf?id=rJY0-Kcll)

### 孪生网络
- [ICMLW2015] [Siamese neural networks for one-shot image recognition](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)
- [ECCV2016] [Fully-convolutional siamese networks for object tracking](https://arxiv.org/abs/1606.09549)

### 原型网络
变体：高斯原型网络、半原型网络
- [NIPS2017] [Prototypical networks for few-shot learning](https://arxiv.org/abs/1703.05175)
- [2017] [Gaussian prototypical networks for few-shot learning on omniglot](https://arxiv.org/abs/1708.02735)
- [ICLR2018] [Meta-Learning for Semi-Supervised Few-Shot Classification](https://arxiv.org/abs/1803.00676)

### 匹配网络/关系网络
- [NIPS2016] [Matching networks for one shot learning](https://arxiv.org/abs/1606.04080)
- [CVPR2018] [Learning to compare: Relation network for few-shot learning](https://arxiv.org/abs/1711.06025)

### 记忆增强网络
- [2014] [Neural turing machines](https://arxiv.org/abs/1410.5401)
- [NIPS2016] [One-shot learning with memory-augmented neural networks](https://arxiv.org/abs/1605.06065)

### 模型无关元学习
MAML、ADML、CAML
- [ICML2017] [Model-agnostic meta-learning for fast adaptation of deep networks](https://arxiv.org/abs/1703.03400)
- [2018] [Adversarial meta-learning](https://arxiv.org/abs/1806.03316)
- [ICML2019] [Fast context adaptation via meta-learning](https://arxiv.org/abs/1810.03642)