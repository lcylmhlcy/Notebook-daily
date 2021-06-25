## metric-based meta
- [ICMLW2015] [Siamese neural networks for one-shot image recognition](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)
- [NIPS2016] [**baseline**] [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
- [NIPS2017] [**Strong baseline**] [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
- [CVPR2018] [**baseline**] [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) [[Code](https://github.com/floodsung/LearningToCompare_FSL)]
- [CVPR2018] [Dynamic Few-Shot Visual Learning without Forgetting](https://arxiv.org/abs/1804.09458) [[Code](https://github.com/gidariss/FewShotWithoutForgetting)]
- [CVPR2018] [Low-Shot Learning with Imprinted Weights](https://arxiv.org/abs/1712.07136)
- [ICLR2018] [Meta-Learning for Semi-Supervised Few-Shot Classification](https://arxiv.org/abs/1803.00676)
  - 在原型网络上优化
- [NIPS2019] [Adaptive Cross-Modal Few-Shot Learning](https://arxiv.org/abs/1902.07104)
  - 把词语作为文本输入，非常简单的多模态融合，基于原型网络
- [NIPS2019] [Cross Attention Network for Few-shot Classification](https://arxiv.org/abs/1910.07677)
  - 基于relation网络，套SENet
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
- [CVPR2019] [Few-Shot Learning with Localization in Realistic Settings](https://arxiv.org/abs/1904.08502) [[Code](https://github.com/daviswer/fewshotlocal)]
- [ICLR2020] [Meta-dataset: A dataset of datasets for learning to learn from few examples](https://arxiv.org/abs/1903.03096) [[Code](https://github.com/google-research/meta-dataset)]
- [IJCAI2020] [Multi-attention meta learning for few-shot fine-grained image recognition](http://vipl.ict.ac.cn/homepage/jsq/publication/2020-Zhu-IJCAI-PRICAI.pdf)
  - 基于MAML，引入CBAM注意力机制
- [CVPR2020] [DeepEMD: Differentiable Earth Mover's Distance for Few-Shot Learning](https://arxiv.org/abs/2003.06777) [[Code](https://github.com/icoz69/DeepEMD)]
  - 基于image regions，三种情况FCN、Grid、Sampling
- [CVPR2020] [Adaptive Subspaces for Few-Shot Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Simon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.html) [[Code](https://github.com/chrysts/dsn_fewshot)]
- [CVPR2020] [Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions](https://arxiv.org/abs/1812.03664) [[Code](https://github.com/Sha-Lab/FEAT)]
- [CVPR2020] [Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition](https://arxiv.org/abs/2004.00705) [[Code](https://github.com/Tsingularity/PoseNorm_Fewshot)]
- [ICLR2021] [Concept Learners for Few-Shot Learning](https://arxiv.org/pdf/2007.07375.pdf) [[Code](https://github.com/snap-stanford/comet)]
  - 基于原型网络，随机生成concept
- [CVPR2021] [Few-Shot Classification with Feature Map Reconstruction Networks](https://arxiv.org/abs/2012.01506) [[Code](https://github.com/Tsingularity/FRN)]



## Self-supervised
- [2018] [**CPC**] [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) [**InfoNCE**]
- [ECCV2018] [Deep clustering for unsupervised learning of visual features](https://arxiv.org/abs/1807.05520) 
- [ICLR2019] [Learning deep representations by mutual information estimation and maximization](https://arxiv.org/abs/1808.06670) [**Bengio**]
- [CVPR2020] [**MOCO**] [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) [[Code](https://github.com/facebookresearch/moco)] [**Kaiming He**]
- [CVPR2020] [**PIRL**] [Self-Supervised Learning of Pretext-Invariant Representations](https://arxiv.org/abs/1912.01991)
- [NIPS2020] [**SWAV**] [Unsupervised learning of visual features by contrasting cluster assignments](https://arxiv.org/abs/2006.09882)
- [NIPS2020] [**BYOL**] [Bootstrap your own latent: A new approach to self-supervised learning](https://arxiv.org/abs/2006.07733) [DeepMind]
- [2020] [AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries](https://arxiv.org/abs/2011.08435)
- [ICML2020] [Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272)
- [ICML2020] [**SIMCLR**] [A simple framework for contrastive learning of visual representations](https://arxiv.org/abs/2002.05709) [**Hinton**]
- [2020] [**MOCO v2**] [Improved baselines with momentum contrastive learning](https://arxiv.org/abs/2003.04297) [**Kaiming He**]
- [2020] [**SIMCLR v2**] [Big self-supervised models are strong semi-supervised learners](https://arxiv.org/abs/2006.10029) [[Code](https://github.com/google-research/simclr)] [**Hinton**]
- [2020] [**SIMSIAM**] [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566) [**Kaiming He**]
- [ICML2021] [Whitening for Self-Supervised Representation Learning](https://arxiv.org/abs/2007.06346)
- [2021] [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) [[Code](https://github.com/facebookresearch/barlowtwins)] [**LeCun**]
- [2021] [**MOCO v3**] [An Empirical Study of Training Self-Supervised Visual Transformers](https://arxiv.org/abs/2104.02057) [**Kaiming He**]
- [2021] [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/pdf/2105.04906.pdf) [**LeCun**]



## Concept-based Interpret
- [x] [ICML2018] [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://arxiv.org/abs/1711.11279) [**Been Kim**]
  - TCAV importance score，量化concept对模型影响度
  - concept: high-level, human-friendly; feature: low-level
  - prodeces estimates of how important that a concept was for the prediction
- [x] [ECCV2018] [Interpretable basis decomposition for visual explanation](http://bzhou.ie.cuhk.edu.hk/publication/eccv18-IBD.pdf) [[Code](https://github.com/CSAILVision/IBD)]
  - decomposes the prediction of one image into human-interpretable **conceptual components**
- [x] [NIPS2019] [Towards Automatic Concept-based Explanations](https://arxiv.org/abs/1902.03129) [[Code](https://github.com/amiratag/ACE)] [**Been Kim**]
  - 找到响应最高的concept，基于TCAV
  - without human supervision
  - concept-based explanation: meaningfulness, coherency, and importance
- [2019] [EDUCE: Explaining model Decisions through Unsupervised Concepts Extraction](https://arxiv.org/abs/1905.11852)
- [2019] [Explaining Classifiers with Causal Concept Effect (CaCE)](https://arxiv.org/abs/1907.07165) [**Been Kim**]
  - Make TCAV causal
  - propose a different causal prior graph to model the spurious correlations among the concepts and remove them using conditional variational autoencoders.
- [ICMLA2019] [Concept Saliency Maps to Visualize Relevant Features in Deep Generative Models](https://arxiv.org/abs/1910.13140) [[Code](https://github.com/lenbrocki/concept-saliency-maps)]
- [ACMMM2020] [Concept-based Explanation for Fine-grained Images and Its Application in Infectious Keratitis Classification](https://dl.acm.org/doi/10.1145/3394171.3413557) [[Code](https://github.com/createrfang/VisualConceptMining)]
- [ICMLW2020] [Robust Semantic Interpretability: Revisiting Concept Activation Vectors](https://arxiv.org/abs/2104.02768) [[Code](https://github.com/keiserlab/rcav)]
- [ ] [ICML2020] [Concept Bottleneck Models](http://proceedings.mlr.press/v119/koh20a/koh20a.pdf) [[SUP](http://proceedings.mlr.press/v119/koh20a/koh20a-supp.pdf)] [[Code](https://github.com/yewsiang/ConceptBottleneck)] [**Been Kim**]
  - Build a model where concepts are built-in so that you can control influential concepts.
- [NIPS2020] [On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://arxiv.org/abs/1910.07969) [[Code](https://github.com/chihkuanyeh/concept_exp)] [**Been Kim**]
  - find set of concepts that are "sufficient" to explain predictions.
- [ ] [ICLR2021] [Debiasing Concept-based Explanations with Causal Analysis](https://arxiv.org/abs/2007.11500)
- [ ] [ICLRW2021] [Do Concept Bottleneck Models Learn as Intended?](https://arxiv.org/abs/2105.04289)
- [CVPR2021] [A Peek Into the Reasoning of Neural Networks: Interpreting with Structural Visual Concepts](https://arxiv.org/pdf/2105.00290.pdf) [[Code](https://github.com/gyhandy/Visual-Reasoning-eXplanation)] [[Blog](https://mp.weixin.qq.com/s/FhQsi7twHkGskcE5fshOxA)]



## Concept learning/Compositional representations/part-based
- [MIT Technical report 1970] [Learning structural descriptions from examples](https://dspace.mit.edu/handle/1721.1/6884)
- [Biological Sciences 1978] [Representation and recognition of the spatial organization of three-dimensional shapes](https://royalsocietypublishing.org/doi/abs/10.1098/rspb.1978.0020)
- [MIT Press 1982] [Vision: A computational investigation into the human representation and processing of visual information](http://papers.cumincad.org/cgi-bin/works/Show?fafahttp://papers.cumincad.org/cgi-bin/works/Show?fafa)
- [Cognition 1984] [Parts of recognition](https://www.sciencedirect.com/science/article/pii/0010027784900222)
- [Psychological review 1987] [Recognition-by-components: a theory of human image understanding](https://psycnet.apa.org/record/1987-20898-001)
- [MIT 1999] [A Bayesian framework for concept learning](https://dspace.mit.edu/handle/1721.1/16714)
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