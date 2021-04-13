# Academic Notes
Only for Paper !!!

## New Method and task
- 图卷积
- 贝叶斯学习
- 对抗攻击
- 小样本/零样本 Few-shot/Zero-shot
- 数据集处理 Long-tailed 长尾分布
- 开集识别 Open-Set Recognition
- 推理 Reasoning
    - Visual
    - Relation
    - Multimodal
- 强化学习 Reinforcement Learning
- 元学习 Meta Learning/Learning to learn
    - 原型网络/matching net/relation net
    - 元模仿学习 meta imitation learning (MIL)
- 迁移学习 Transfer Learning
    - Domain 自适应
- 因果学习 Causal Learning
    - 反事实 counterfactual
    - 因果干预 causal intervention
- 持续学习 Continual/Life-long Learning、增量学习 Incremental Learning
- 无监督 Unsupervised
    - 自监督 Self-supervised
        - 对比学习 Contrastive Learning
- 半监督 Semi-supervised
- 概念学习 Concept Learning
- 主动学习 Active Learning

## AI3.0 Vision 
**from recognition to cognition (Bottom to Top)**
- 抽象概况(比如场景图)
- 事件推理
- 因果推理
- 物理规律的推理和总结
- 对未来意图的预判
- 构建常识体系
- 规划

### Some opinions about vision future
- 目前貌似第一步的效果还达不到要求（仅从paper中总结）。
- 图像转成多个语言标签或图结构形式，有助于检索、分析和与其他模态融合，可能是未来趋势。
- 度量学习可能会成为主流，目前小样本、自监督、迁移等任务都会使用度量。人在看见新知识或者新事物时，会根据以前的知识来判断新事物与原有知识库中的事物的区别，从而判断其类别。“举一反三”，既具备推理能力，也具备度量能力。
- 持续学习/增强学习是目前大部分模型欠缺的能力。迁移学习后会忘记原来的数据，不利于在线学习。例如，分类模型需要不断加入新类的数据训练。
- 推理能力仍处理较差。由于神经网络本身缺乏推理性，而设计的模块大多没有严谨的数学证明，因此大部分论文都是根据实验结果讲故事。其中，多模态融合（视觉语言）缺乏推理，只是单纯的点乘或者拼接，这跟提取特征没有可解释性有很大关系。
- 数据集问题受到关注，个人以为是一个好的趋势。深度学习像是一个大数据驱动的统计学习方法，对数据集的设计和分析尤为重要，网络设计也应从数据和任务本身出发，然而现在大部分模型设计感觉跟数据没什么关系。大家只关注总体提升多少metric，很少分析难例。例如，样本数量失衡的长短尾问题、人为标注错误问题等。
- 传统的算法质量评价在深度学习方法上很难评测。例如，超分目前产出大量论文，指标多为PSNR和SSIM，不知道该方法在哪类数据上表现较好，比较贴合哪种应用环境。研究通用的方法有价值，但是也应处理某些经常误判的类别群体。
- 超大数据训练的超大模型霸榜，很正常，不知道是否对目前的深度学习科研环境造成冲击，变成富人的垄断。
- 视觉两个很明显的方向：云（大规模分布式）和边（嵌入式机器）。云太耗钱了，一般人玩不起，买点nano、tx2还是可以的。
  
