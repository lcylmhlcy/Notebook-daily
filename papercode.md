## Few-shot/meta/openset
- [Demo] [Code for few shot](https://github.com/oscarknagg/few-shot)
- [Demo] [Hands-On-Meta-Learning-With-Python](https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python)

### distance metric
- [ICMLW2015] [Siamese neural networks for one-shot image recognition](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)
- [NIPS2016] [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
- [NIPS2017] [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
- [CVPR2018] [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) [[Code](https://github.com/floodsung/LearningToCompare_FSL)]
- [CVPR2018] [Dynamic Few-Shot Visual Learning without Forgetting](https://arxiv.org/abs/1804.09458) [[Code](https://github.com/gidariss/FewShotWithoutForgetting)]
- [CVPR2018] [Low-Shot Learning with Imprinted Weights](https://arxiv.org/abs/1712.07136)
- [ICLR2018] [Meta-Learning for Semi-Supervised Few-Shot Classification](https://arxiv.org/abs/1803.00676)
  - 在原型网络上优化
- [NIPS2019] [Cross Attention Network for Few-shot Classification](https://arxiv.org/abs/1910.07677)
  - 基于relation网络，套SENet
- [NIPS2019] [Adaptive Cross-Modal Few-Shot Learning](https://arxiv.org/abs/1902.07104)
  - 把词语作为文本输入，非常简单的多模态融合，基于原型网络
- [ICML2019] [Infinite Mixture Prototypes for Few-Shot Learning](https://arxiv.org/abs/1902.04552)
  - 基于原型网络，原先一个类别一个聚类，变成一个类别一组聚类
- [ICLR2019] [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232) [[Code](https://github.com/wyharveychen/CloserLookFewShot)]
  - Baseline and Baseline++ 
- [ICLR2019] [Meta-Learning with Latent Embedding Optimization](https://arxiv.org/abs/1807.05960)
- [ICLR2019] [Meta-learning with differentiable closed-form solvers](https://arxiv.org/abs/1805.08136) [[Code](https://github.com/bertinetto/r2d2)]
- [ICCV2019] [Diversity with Cooperation: Ensemble Methods for Few-Shot Classification](https://arxiv.org/abs/1903.11341)
- [ICCV2019] [Learning Compositional Representations for Few-Shot Recognition](https://arxiv.org/abs/1812.09213) [[Code](https://drive.google.com/file/d/12Hn9pmBjYKGCWzumUmsbdi7viq-L3-IU/view)]
  - 使用属性标签，提出一种concept拼接方法
- [CVPR2019] [Meta-Learning with Differentiable Convex Optimization](https://arxiv.org/abs/1904.03758) [[Code](https://github.com/kjunelee/MetaOptNet)]
- [ICLR2020] [Meta-dataset: A dataset of datasets for learning to learn from few examples](https://arxiv.org/abs/1903.03096) [[Code](https://github.com/google-research/meta-dataset)]
- [CVPR2020] [DeepEMD: Differentiable Earth Mover's Distance for Few-Shot Learning](https://arxiv.org/abs/2003.06777) [[Code](https://github.com/icoz69/DeepEMD)]
  - 基于image regions，三种情况FCN、Grid、Sampling
- [IJCAI2020] [Multi-attention meta learning for few-shot fine-grained image recognition](http://vipl.ict.ac.cn/homepage/jsq/publication/2020-Zhu-IJCAI-PRICAI.pdf)
  - 基于MAML，引入CBAM注意力机制
- [ICLR2021] [Concept Learners for Few-Shot Learning](https://arxiv.org/pdf/2007.07375.pdf) [[Code](https://github.com/snap-stanford/comet)]
  - 基于原型网络，随机生成concept

### graph
- [NIPS2019] [Learning to Propagate for **Graph** Meta-Learning](https://arxiv.org/abs/1909.05024) [[Code](https://github.com/liulu112601/Gated-Propagation-Net)]

### others based
- [NIPS1996] [Is learning the n-th thing any easier than learning the first?](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/thrun-1996.pdf)
- [TPAMI2006] [One-shot learning of object categories](https://ieeexplore.ieee.org/abstract/document/1597116) [**Feifei Li**]
- [2014] [Tinkering Under the Hood: Interactive Zero-Shot Learning with Net Surgery](https://arxiv.org/abs/1612.04901)
- [ICCV2015] [One Shot Learning via Compositions of Meaningful Patches](https://ieeexplore.ieee.org/abstract/document/7410499)
- [ECCV2016] [Learning to learn: Model regression networks for easy small sample learning](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_37)
- [ICLR2017] [Optimization as a Model for Few-Shot Learning](https://openreview.net/forum?id=rJY0-Kcll)
- [ICML2017] [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [NIPS2018] [TADAM: Task dependent adaptive metric for improved few-shot learning](https://arxiv.org/abs/1805.10123) [[Code](https://github.com/ElementAI/TADAM)]

### openset
- [CVPR2021] [Towards Open World Object Detection](https://arxiv.org/abs/2103.02603) [[Code](https://github.com/JosephKJ/OWOD)]
- [CVPR2021] [Counterfactual Zero-Shot and Open-Set Visual Recognition](https://arxiv.org/abs/2103.00887) [[Code](https://github.com/yue-zhongqi/gcm-cf)]



## Concept learning/Compositional representations/part-based
- [MIT Technical report 1970] [Learning structural descriptions from examples](https://dspace.mit.edu/handle/1721.1/6884)
- [Biological Sciences 1978] [Representation and recognition of the spatial organization of three-dimensional shapes](https://royalsocietypublishing.org/doi/abs/10.1098/rspb.1978.0020)
- [MIT Press 1982] [Vision: A computational investigation into the human representation and processing of visual information](http://papers.cumincad.org/cgi-bin/works/Show?fafahttp://papers.cumincad.org/cgi-bin/works/Show?fafa)
- [Cognition 1984] [Parts of recognition](https://www.sciencedirect.com/science/article/pii/0010027784900222)
- [Psychological review 1987] [Recognition-by-components: a theory of human image understanding](https://psycnet.apa.org/record/1987-20898-001)
- [CVPR2000] [Learning from one example through shared densities on transforms](https://people.cs.umass.edu/~elm/papers/cvpr2000.pdf)
- [AMCSS2011] [One shot learning of simple visual concepts](https://escholarship.org/content/qt4ht821jx/qt4ht821jx.pdf)
- [CVPR2007] [Towards scalable representations of object categories: Learning a hierarchy of parts](http://www.mobvis.org/publications/cvpr07_fidler_leonardis.pdf)
- [TPAMI2009] [Object Detection with Discriminatively Trained Part Based Models](http://vision.stanford.edu/teaching/cs231b_spring1415/papers/dpm.pdf)
- [CVPR2009] [Learning To Detect Unseen Object Classes by Between-Class Attribute Transfer](http://ftp.idiap.ch/pub/courses/EE-700/material/28-11-2012/lampert-cvpr2009.pdf)
- [CVPR2010] [Part and Appearance Sharing: Recursive Compositional Models for Multi-View Multi-Object Detection](http://www.cs.jhu.edu/~alanlab/Pubs10/zhu2010part.pdf)
- [CVPR2011] [Shared Parts for Deformable Part-based Models](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.225.6520&rep=rep1&type=pdf)
- [Philosophical Psychology 2012] [Concept possession, experimental semantics, and hybrid theories of reference](https://www.tandfonline.com/doi/abs/10.1080/09515089.2011.627538)
- [Science2015] [Human-level concept learning through probabilistic program induction](https://science.sciencemag.org/content/350/6266/1332/)
- [Behavioral and brain sciences 2017] [Building machines that learn and think like people](https://core.ac.uk/download/pdf/141473153.pdf)
- [CVPR2017] [From Red Wine to Red Tomato: Composition with Context](https://www.cs.cmu.edu/~imisra/data/composing_cvpr17.pdf)
- [CVPR2017] [Teaching compositionality to cnns](https://arxiv.org/abs/1706.04313)
- [ICLRW2018] [Concept Learning with Energy-Based Models](https://arxiv.org/abs/1811.02486)
- [ICLR2019] [Measuring Compositionality in Representation Learning](https://arxiv.org/abs/1902.07181)
- [NIPS2019] [Visual Concept-Metaconcept Learning](http://vcml.csail.mit.edu/data/papers/2019NeurIPS-VCML.pdf) [[Code](https://github.com/Glaciohound/VCML)]
  - concept跟metaconcept的定义，制定推理过程
- [Nature2020] [Concept whitening for interpretable image recognition](https://www.nature.com/articles/s42256-020-00265-z) [[Code](https://github.com/zhiCHEN96/ConceptWhitening)]


## Self-supervised
- [2018] [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [ECCV2018] [Deep clustering for unsupervised learning of visual features](https://arxiv.org/abs/1807.05520) 
- [ICLR2019] [Learning deep representations by mutual information estimation and maximization](https://arxiv.org/abs/1808.06670) [**Bengio**]
- [CVPR2020] [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) [**Kaiming He**] [**MOCO**] [[Code](https://github.com/facebookresearch/moco)]
- [CVPR2020] [Self-Supervised Learning of Pretext-Invariant Representations](https://arxiv.org/abs/1912.01991) [**PIRL**]
- [NIPS2020] [Unsupervised learning of visual features by contrasting cluster assignments](https://arxiv.org/abs/2006.09882) [**SWAV**]
- [NIPS2020] [Bootstrap your own latent: A new approach to self-supervised learning](https://arxiv.org/abs/2006.07733) [DeepMind] [**BYOL**]
- [2020] [AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries](https://arxiv.org/abs/2011.08435)
- [ICML2020] [A simple framework for contrastive learning of visual representations](https://arxiv.org/abs/2002.05709) [**Hinton**] [**SIMCLR**]
- [2020] [Improved baselines with momentum contrastive learning](https://arxiv.org/abs/2003.04297) [**Kaiming He**] [**MOCO v2**]
- [2020] [Big self-supervised models are strong semi-supervised learners](https://arxiv.org/abs/2006.10029) [**Hinton**] [**SIMCLR v2**] [[Code](https://github.com/google-research/simclr)]
- [2020] [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566) [**Kaiming He**] [**SIMSIAM**]
- [AAAI2021] [Train a One-Million-Way Instance Classifier for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2102.04848) [Alibaba]
- [2021] [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) [**LeCun**] [[Code](https://github.com/facebookresearch/barlowtwins)]
- [2021] [An Empirical Study of Training Self-Supervised Visual Transformers](https://arxiv.org/abs/2104.02057) [**Kaiming He**] [**MOCO v3**]
- [2021] [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/pdf/2105.04906.pdf) [**LeCun**]


## Interpret
- [NIPS2009] [Reading tea leaves: How humans interpret topic models](https://www.cs.ubc.ca/~rjoty/Webpage/nips2009-rtl.pdf)
- [ICLRW2014] [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
- [2014] [Methods and Models for Interpretable Linear Classification](https://arxiv.org/abs/1405.4047)
- [ECCV2014] [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
- [NIPS2014] [The bayesian case model: A generative approach for case-based reasoning and prototype classification](https://proceedings.neurips.cc/paper/2014/file/390e982518a50e280d8e2b535462ec1f-Paper.pdf)
- [CVPR2015] [Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035)
- [AISTATS2015] [Falling Rule Lists](https://arxiv.org/abs/1411.5899)
- [2016] [Model-Agnostic Interpretability of Machine Learning](https://arxiv.org/abs/1606.05386)
- [KDD2016] ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- [2017] [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
- [NIPS2017] [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [NIPS2017] [Real Time Image Saliency for Black Box Classifiers](https://arxiv.org/abs/1705.07857)
- [ICML2017] [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)
- [ICML2017] [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)
- [ICCV2017] [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [NIPS2018] [Towards Robust Interpretability with Self-Explaining Neural Networks](https://arxiv.org/abs/1806.07538)
- [NIPS2018] [Sanity checks for saliency maps](https://arxiv.org/abs/1810.03292) [**Goodfellow**]
- [ICML2018] [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://arxiv.org/abs/1711.11279) [Google]
  - TCAV importance score，量化concept对模型影响度
  - concept: high-level, human-friendly; feature: low-level
  - prodeces estimates of how important that a concept was for the prediction
- [AAAI2018] [Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions](https://arxiv.org/abs/1710.04806)
- [ECCV2018] [Interpretable basis decomposition for visual explanation](http://bzhou.ie.cuhk.edu.hk/publication/eccv18-IBD.pdf)
  - decomposes the prediction of one image into human-interpretable **conceptual components**
- [CVPR2018] [Deep Image Prior](https://arxiv.org/abs/1711.10925) [[Code](https://github.com/DmitryUlyanov/deep-image-prior)]
- [CVPR2019] [Interpretable and Fine-Grained Visual Explanations for Convolutional Neural Networks](https://arxiv.org/abs/1908.02686)
- [TPAMI2018] [Interpreting Deep Visual Representations via Network Dissection](https://ieeexplore.ieee.org/abstract/document/8417924)
- [NIPS2019] [This Looks Like That: Deep Learning for Interpretable Image Recognition](https://arxiv.org/abs/1806.10574)
- [AAAI2019] [Interpretation of Neural Networks is Fragile](https://arxiv.org/abs/1710.10547)
- [AISTATS2019] [Knockoffs for the mass: new feature importance statistics with false discovery guarantees](https://arxiv.org/abs/1807.06214)
- [ExplainableAI2019] [The (un) reliability of saliency methods](https://arxiv.org/abs/1711.00867)
- [NIPS2019] [Towards Automatic Concept-based Explanations](https://arxiv.org/abs/1902.03129) [[Code](https://github.com/amiratag/ACE)]
  - 找到响应最高的concept，基于TCAV
  - without human supervision
  - concept-based explanation: meaningfulness, coherency, and importance
- [2021] [Manipulating and Measuring Model Interpretability](https://arxiv.org/abs/1802.07810)
- [ICML WHI 2020] [Robust Semantic Interpretability: Revisiting Concept Activation Vectors](https://arxiv.org/abs/2104.02768) [[Code](https://github.com/keiserlab/rcav)]

## Semi-supervised/Unsupervised
- [ICCV2015] [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)
- [CVPR2018] [Unsupervised discovery of object landmarks as structural representations](https://arxiv.org/abs/1804.04412) [[Code](https://github.com/YutingZhang/lmdis-rep)]
- [CVPR2021] [SimPLE: Similar Pseudo Label Exploitation for Semi-Supervised Classification](https://arxiv.org/pdf/2103.16725.pdf) [[Code](https://github.com/zijian-hu/SimPLE)]


## Dataset and preprocess
- [CVPR2011] [Unbiased look at dataset bias](https://ieeexplore.ieee.org/abstract/document/5995347)
- [ACPR2017] [A Deeper Look at Dataset Bias](https://link.springer.com/chapter/10.1007/978-3-319-58347-1_2)
- [Arxiv2018] [Why do deep convolutional networks generalize so poorly to small image transformations?](https://www.jmlr.org/papers/volume20/19-519/19-519.pdf)
- [ICML2019] [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486) [[Code](https://github.com/adobe/antialiased-cnns)]
- [CVPR2019] [Destruction and Construction Learning for Fine-grained Image Recognition](https://openreview.net/forum?id=HibvKgQe_pH) [[Code](https://github.com/JDAI-CV/DCL)]
- [CVPR2021] [MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition](https://arxiv.org/abs/2103.12579) [[Code](https://github.com/BIT-DA/MetaSAug)]
- [CVPR2021] [Learning Continuous Image Representation with Local Implicit Image Function](https://arxiv.org/abs/2012.09161) [[Code](https://github.com/yinboc/liif)]
- [CVPR2021] [CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning](https://arxiv.org/abs/2102.09559)
- [CVPR2021] [Contrastive Learning based Hybrid Networks for Long-Tailed Image Classification](https://arxiv.org/pdf/2103.14267.pdf) [[Code](https://k-han.github.io/HybridLT)]


## Reasoning
- [ICLR2020] [CLEVRER: CoLlision Events for Video REpresentation and Reasoning](https://arxiv.org/abs/1910.01442) [**DeepMind**] [[Project](http://clevrer.csail.mit.edu/)]
- [CVPR2020] [Graph-Structured Referring Expression Reasoning in The Wild](https://arxiv.org/pdf/2004.08814.pdf) [[Github](https://github.com/sibeiyang/sgmn)]
- [ICCV2017] [Inferring and Executing Programs for Visual Reasoning](https://arxiv.org/abs/1705.03633) [[Code](https://github.com/facebookresearch/clevr-iep)]
- [CVPR2021] [Transformation driven Visual Reasoning](https://arxiv.org/pdf/2011.13160) [[Code](https://github.com/hughplay/TVR)]


## Causal
- [CVPR2020] [Visual Commonsense R-CNN](https://arxiv.org/abs/2002.12204) [[zhihu](https://zhuanlan.zhihu.com/p/111306353)] [[Github](https://github.com/Wangt-CN/VC-R-CNN)]
- [2021] [A Survey of Learning Causality with Data: Problems and Methods](https://arxiv.org/abs/1809.09337v4)
- [2021] [Extracting Causal Viusal Features for Limited Lable Classification](https://arxiv.org/pdf/2103.12322.pdf)


## Bayes
- [2021] [A Survey on Bayesian Deep Learning](https://arxiv.org/abs/1604.01662v4) 
- [Demo] [Bayesian neural network using Pyro and PyTorch on MNIST dataset](https://github.com/paraschopra/bayesian-neural-network-mnist)


## multimodal
- [Dataset] [OpenVQA](https://github.com/MILVLG/openvqa)
- [CVPR2019] [MUREL: Multimodal Relational Reasoning for Visual Question Answering](https://arxiv.org/abs/1902.09487) [[Code](https://github.com/Cadene/murel.bootstrap.pytorch)]
- [CVPR2019] [Composing Text and Image for Image Retrieval](https://arxiv.org/abs/1812.07119) [[Code](https://github.com/google/tirg)]
- [ICCV2019] [Zero-Shot Grounding of Objects from Natural Language Queries](https://arxiv.org/abs/1908.07129) [[Code](https://github.com/TheShadow29/zsgnet-pytorch)]
- [CVPR2020 Tutorial] [Recent Advances in Vision-and-Language Research](https://rohit497.github.io/Recent-Advances-in-Vision-and-Language-Research/)
- [Survey on Deep Multi-modal Data Analytics: Collaboration, Rivalry and Fusion](https://arxiv.org/abs/2006.08159)
- [CVPR2020] [X-Linear Attention Networks for Image Captioning](https://arxiv.org/pdf/2003.14080.pdf) [[Code](https://github.com/JDAI-CV/image-captioning)]
- [CVPR2020] [Say As You Wish: Fine-grained Control of Image Caption Generation with Abstract Scene Graphs](https://arxiv.org/pdf/2003.00387.pdf)
- [CVPR2020] [Spatio-Temporal Graph for Video Captioning with Knowledge Distillation](https://arxiv.org/pdf/2003.13942.pdf)
- [CVPR2020] [Object Relational Graph with Teacher-Recommended Learning for Video Captioning](https://arxiv.org/pdf/2002.11566.pdf)
- [CVPR2020] [Counterfactual Samples Synthesizing for Robust VQA](https://arxiv.org/pdf/2003.06576.pdf) [[Code](https://github.com/yanxinzju/CSS-VQA)]
- [CVPR2020] [Hierarchical Conditional Relation Networks for Video Question Answering](https://arxiv.org/abs/2002.10698) [[Code](https://github.com/thaolmk54/hcrn-videoqa)]
- [ICLR2020] [VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530)
- [ACMMM2020] [KBGN: Knowledge-Bridge Graph Network for Adaptive Vision-Text Reasoning in Visual Dialogue](https://arxiv.org/abs/2008.04858v2)
- [NIPS2020] [Deep Multimodal Fusion by Channel Exchanging](https://papers.nips.cc/paper/2020/file/339a18def9898dd60a634b2ad8fbbd58-Paper.pdf) [[Code](https://github.com/yikaiw/CEN)]
- [CVPR2021] [VirTex: Learning Visual Representations from Textual Annotations](https://arxiv.org/abs/2006.06666) [[Code](https://github.com/kdexd/virtex)]
- [AAAI2021] [SMIL: Multimodal Learning with Severely Missing Modality](https://arxiv.org/pdf/2103.05677.pdf) [[Code](https://github.com/mengmenm/SMIL)]
- [ICLR2021] [Iterated learning for emergent systematicity in VQA](https://openreview.net/pdf?id=Pd_oMxH8IlF)


## Video/Action
- [ACMMM2020] [Dual Temporal Memory Network for Efficient Video Object Segmentation](https://arxiv.org/abs/2003.06125)
- [ICCV2019] [Compositional Video Prediction](https://arxiv.org/abs/1908.08522) [[Code](https://github.com/JudyYe/CVP)]
- [2021] [Reformulating HOI Detection as Adaptive Set Prediction](https://arxiv.org/pdf/2103.05983.pdf)
- [ICLR2021] [AdaFuse: Adaptive Temporal Fusion Network for Efficient Action Recognition](https://arxiv.org/pdf/2102.05775.pdf) [[Code](https://github.com/mengyuest/AdaFuse)]


## Graph NN
- [CVPR2020 Tutorial] [Learning Representations via Graph-structured Networks](https://xiaolonw.github.io/graphnnv2/)
- [KDD2020] [Xgnn: Towards model-level explanations of graph neural networks](https://arxiv.org/abs/2006.02587)
- [ICML2020] [Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/pdf/2006.05582.pdf)
- [NIPS2020 Tutorial] [Graph Mining and Learning](https://gm-neurips-2020.github.io/)
- [2020] [Contrastive Learning of Structured World Models](http://arxiv.org/abs/1911.12247) [[Code](https://github.com/tkipf/c-swm)]


## 1D data
- [KDD2020] [Hybrid Spatio-Temporal Graph Convolutional Network: Improving Traffic Prediction with Navigation Data](https://arxiv.org/abs/2006.12715)


## Others
- [ECCV2018] [Partial Convolution Layer for Padding and Image Inpainting](https://arxiv.org/pdf/1811.11718.pdf) [[Code](https://github.com/NVIDIA/partialconv)]
- [ICLR2021] [The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers](https://arxiv.org/pdf/2010.08127.pdf) [[Dataset](https://github.com/preetum/cifar5m)]
- [CVPR2021] [Dynamic Metric Learning: Towards a Scalable Metric Space to Accommodate Multiple Semantic Scales](https://arxiv.org/pdf/2103.11781v1.pdf) [[Code](https://github.com/SupetZYK/DynamicMetricLearning)]
- [亚利桑那州立大学周纵苇：视觉的目的是什么？](https://hub.baai.ac.cn/view/6777)
- [2021] [Embodied Intelligence via Learning and Evolution](https://arxiv.org/pdf/2102.02202.pdf) [**Feifei Li**]