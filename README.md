# 我的日常科研笔记

## 模版
1. **题目**：[。。。]() [【Github】]()  
  **作者及接受期刊/会议**：  
  **摘要**：

---

## 笔记
1. **题目**：[Hybrid Spatio-Temporal Graph Convolutional Network: Improving Traffic Prediction with Navigation Data](https://arxiv.org/abs/2006.12715)  
  **作者及接受期刊/会议**：高德机器学习团队，KDD2020  
  **摘要**：时空预测在天气预报、运输规划等领域有着重要的应用价值。交通预测作为一种典型的时空预测问题，具有较高的挑战性。以往的研究中主要利用通行时间这类交通状态特征作为模型输入，很难预测整体的交通状况，本文提出的混合时空图卷积网络，利用导航数据大大提升了时空预测的效果。
  
2. **题目**：[Recent Advances in Vision-and-Language Research](https://rohit497.github.io/Recent-Advances-in-Vision-and-Language-Research/)  
  **作者及接受期刊/会议**：微软和facebook的Licheng Yu, Yen-Chun Chen, Linjie Li，CVPR2020 workshop  
  **摘要**：视觉和语言(V+L)研究是计算机视觉和自然语言处理之间联系的一个有趣的领域，并迅速吸引了这两个领域的关注。各种各样的V+L任务，以大规模的人类注释数据集为基准，已经推动了联合多模态表示学习的巨大进步。本教程将重点介绍该领域中最近流行的一些任务，如视觉描述、视觉基准、视觉问题回答和推理、文本到图像的生成以及通用图像-文本表示的自监督学习。我们将涵盖这些领域的最新方法，并讨论集中体现多模态理解、推理和生成的核心挑战和机遇的关键原则。
  
3. **题目**：[Survey on Deep Multi-modal Data Analytics: Collaboration, Rivalry and Fusion](https://arxiv.org/abs/2006.08159)  
  **作者及接受期刊/会议**：合肥工业大学  
  **摘要**：随着web技术的发展，多模态或多视图数据已经成为大数据的主要流，每个模态/视图编码数据对象的单个属性。不同的模态往往是相辅相成的。这就引起了人们对融合多模态特征空间来综合表征数据对象的研究。大多数现有的先进技术集中于如何融合来自多模态空间的能量或信息，以提供比单一模态的同行更优越的性能。最近，深度神经网络展示了一种强大的架构，可以很好地捕捉高维多媒体数据的非线性分布，对多模态数据自然也是如此。大量的实证研究证明了深多模态方法的优势，从本质上深化了多模态深特征空间的融合。在这篇文章中，我们提供了从浅到深空间的多模态数据分析领域的现有状态的实质性概述。在整个调查过程中，我们进一步指出，该领域的关键要素是多模式空间的协作、对抗性竞争和融合。最后，我们就这一领域未来的一些方向分享我们的观点。
  
4. **题目**：[Learning Representations via Graph-structured Networks](https://xiaolonw.github.io/graphnnv2/)  
  **作者及接受期刊/会议**：CVPR 2020, The 2nd Tutorial  
  **摘要**：近年来，卷积神经网络(ConvNets)在大量计算机视觉任务中的应用出现了戏剧性的增长。卷积结构在许多任务中都是非常强大的，它可以从图像像素中提取相关性和抽象概念。然而，当面对一些更困难的计算机视觉任务时，ConvNets在建模中也有相当多的属性方面存在缺陷。这些属性包括成对关系、全局上下文和处理超越空间网格的不规则数据的能力。一个有效的方向是根据手头的任务重新组织要用图处理的数据，同时构建网络模块，在图内的视觉元素之间关联和传播信息。我们将这种具有传播模块的网络称为图网络结构。在本教程中，我们将介绍一系列有效的图网络结构，包括非局部神经网络、空间广义传播网络、面向对象和多主体行为建模的关系网络、面向3D领域的视频和数据的图网络。我们还将讨论如何利用图神经网络结构来研究连接模式。最后，我们将讨论在许多视觉问题中仍然存在的相关开放挑战。
  
5. **题目**：[XGNN-可解释图神经网络，从模型级解释构建可信赖GNN](https://xiaolonw.github.io/graphnnv2/)  
  **作者及接受期刊/会议**：KDD2020  
  **摘要**：图神经网络通过聚合和结合邻居信息来学习节点特征，在许多图的任务中取得了良好的性能。然而，GNN大多被视为黑盒，缺乏人类可理解的解释。因此，如果不能解释GNN模型，就不能完全信任它们并在某些应用程序域中使用它们。在这项工作中，我们提出了一种新的方法，称为XGNN，在模型级别上解释GNN。我们的方法可以为GNNs的工作方式提供高层次的见解和一般性的理解。特别地，我们提出通过训练一个图生成器来解释GNN，使生成的图模式最大化模型的某种预测。我们将图形生成表述为一个强化学习任务，其中对于每一步，图形生成器预测如何向当前图形中添加一条边。基于训练后的GNN信息，采用策略梯度方法对图生成器进行训练。此外，我们还加入了一些图规则，以促使生成的图是有效的。在合成和真实数据集上的实验结果表明，我们提出的方法有助于理解和验证训练过的GNN。此外，我们的实验结果表明，所生成的图可以为如何改进训练的神经网络提供指导。
  
6. **题目**：[CLEVRER数据集，推动视频理解的因果逻辑推理](https://arxiv.org/abs/1910.01442) [【Project】](http://clevrer.csail.mit.edu/)  
  **作者及接受期刊/会议**：ICLR 2020 论文，麻省理工、DeepMind  
  **摘要**：提出了一种针对时间和因果推理问题的数据集，包含 20,000 个关于碰撞物体的合成视频以及 300,000 多个问题和答案，从互补的角度研究了视频中的时间和因果推理问题。
  
7. **题目**：[对视觉与语言的思考：从自洽、交互到共生](https://github.com/JDAI-CV/image-captioning)  
  **作者及接受期刊/会议**：CVPR 2020，京东AI研究院  
  **摘要**：纵观视觉与语言在这六年间的飞速发展史，它就仿佛是两种不同文化（计算机视觉与自然语言处理）的碰撞与交融。这里每一种文化最初的进化都是自洽的，即独立地演化形成一套完备的视觉理解或语言建模体系；演化至今，我们当前所迎来的则是两种文化间的交互，自此视觉理解和语言建模不再是简单串联的两个模块，而是通过互相的信息传递成为共同促进的一个整体；对于视觉与语言的未来，则一定是聚焦于两者更为本质和紧密的共生，它所渴望的，将是挣脱开数据标注的桎梏，在海量的弱监督甚至于无监督数据上找寻两者间最为本质的联系，并以之为起源，如「道生一，一生二，二生三，三生万物」一般，赋予模型在各种视觉与语言任务上的生命力。
  
8. **题目**：[Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions](https://arxiv.org/pdf/2004.03967.pdf)  
  **作者及接受期刊/会议**：CVPR2020  
  **摘要**：场景理解（scene understanding）一直是计算机视觉领域的研究热点。它不仅包括识别场景中的对象，还包括识别它们在给定上下文中的关系。基于这一目标，最近的一系列工作解决了3D语义分割和场景布局预测问题。在我们的工作中，我们关注场景图，这是一种在图中组织场景实体的数据结构，其中对象是节点，它们的关系建模为边。我们利用场景图上的推理作为实现3D场景理解、映射对象及其关系的一种方式。特别地，我们提出了一种从场景的点云回归场景图的学习方法。我们的新体系结构是基于PointNet和图卷积网络(GCN)的。此外，我们还介绍了一个半自动生成的数据集3DSSG，它包含了语义丰富的三维场景图。我们展示了我们的方法在一个领域无关的检索任务中的应用，其中图作为3D-3D和2D-3D匹配的中间表示。
  
9. **题目**：[Graph-Structured Referring Expression Reasoning in The Wild](https://arxiv.org/pdf/2004.08814.pdf) [【Github】](https://github.com/sibeiyang/sgmn)    
  **作者及接受期刊/会议**：CVPR2020  
  **摘要**：Grounding referring expressions的目标是参照自然语言表达式在图像中定位一个目标。指代表达式（referring expression）的语言结构为视觉内容提供了推理的布局，并且该结构对于校准和共同理解图像与指代表达式是十分重要的。本文提出了一种场景图引导的模块化网络(SGMN)，它在表达式的语言结构指导下，用神经模块对语义图和场景图进行推理。特别地，我们将图像（image）建模为结构化语义图，并将表达式解析为语言场景图。语言场景图不仅对表达式的语言结构进行解码，而且与图像语义图具有一致的表示。除了探索指代表达式基础的结构化解决方案外，我们还提出了Ref-Reasning，一个用于结构化指代表达式推理的大规模真实数据集。我们使用不同的表达式模板和函数式程序自动生成图像场景图上的指代表达式。该数据集配备了真实世界的可视化内容以及具有不同推理布局的语义丰富的表达式。实验结果表明，SGMN不仅在新的Ref-Reasning数据集上的性能明显优于现有的算法，而且在常用的基准数据集上也超过了最先进的结构化方法。它还可以为推理提供可解释的可视化证据。
  
10. **题目**：[Say As Y ou Wish: Fine-grained Control of Image Caption Generation with Abstract Scene Graphs](https://arxiv.org/pdf/2003.00387.pdf)  
  **作者及接受期刊/会议**：CVPR2020  
  **摘要**：人类能够随心所欲地用粗到细的细节来描述图像内容。然而，大多数图像字幕模型是意图不可知的（intention-agnostic），不能主动根据不同的用户意图生成各种描述。在这项工作中，我们提出了抽象场景图(ASG)结构来在细粒度层次上表示用户意图，并控制生成的描述应该是什么和有多详细。ASG是一个由三种类型的抽象节点(对象、属性、关系)组成的有向图，它们以图像为基础，没有任何具体的语义标签。因此，这些节点可以很容易通过手动或自动获得。与在VisualGenome和MSCOCO数据集上精心设计的基线相比，我们的模型在ASG上实现了更好的可控性条件。它还通过自动采样不同的ASG作为控制信号，显著提高了字幕多样性。
  
11. **题目**：[Semantic Image Manipulation Using Scene Graphs](https://www.researchgate.net/publication/340523427_Semantic_Image_Manipulation_Using_Scene_Graphs) [【Github】](https://he-dhamo.github.io/SIMSG/)  
  **作者及接受期刊/会议**：CVPR2020  
  **摘要**：图像处理可以被认为是图像生成的特例，其中要生成的图像是对现有图像的修改。在很大程度上，图像生成和处理都是对原始像素进行操作的任务。然而，在学习丰富的图像和对象表示方面的显著进展已经为主要由语义驱动的诸如文本到图像或布局到图像生成之类的任务开辟了道路。在我们的工作中，我们解决了从场景图进行图像处理的新问题，在该问题中，用户可以仅通过对从图像生成的语义图的节点或边进行修改来编辑图像。我们的目标是对给定constellation中的图像信息进行编码，然后在此基础上生成新的constellation，例如替换对象，甚至改变对象之间的关系，同时尊重原始图像的语义和样式。我们引入了空间语义场景图网络，该网络不需要直接监督constellation变化或图像编辑。这使得从现有的现实世界数据集中训练系统成为可能，而无需额外的注释工作。
  
12. **题目**：[Spatio-Temporal Graph for Video Captioning with Knowledge Distillation](https://arxiv.org/pdf/2003.13942.pdf)  
  **作者及接受期刊/会议**：CVPR2020  
  **摘要**：视频描述生成是一项具有挑战性的任务，需要对视觉场景有深刻的理解。最先进的方法使用场景级或对象级信息生成字幕，但没有显式建模对象交互。因此，它们通常无法做出基于视觉的预测，并且对虚假相关性敏感。在本文中，我们为视频字幕提出了一种新颖的时空图模型，该模型利用了时空中的对象交互作用。我们的模型建立了可解释的连接，并且能够提供明确的视觉基础。为了避免对象数量变化带来的性能不稳定，我们进一步提出了一种对象感知的知识提炼机制，该机制利用局部对象信息对全局场景特征进行正则化。通过在两个基准上的广泛实验证明了我们的方法的有效性，表明我们的方法在可解释的预测上产生了具有竞争力的性能。
  
13. **题目**：[ContrastiveMulti-ViewRepresentationLearningonGraphs](https://arxiv.org/pdf/2006.05582.pdf)  
  **作者及接受期刊/会议**：ICML2020，对比多视角表示学习   
  **摘要**：我们介绍了一种通过对比图的结构视图来学习节点和图表示的自监督方法。我们表明，与视觉表示学习不同，增加视图数量到两个以上或对比多尺度编码不会提高性能，而最佳性能是通过对比一阶邻居编码和图扩散来实现的。在线性评估协议下，我们在8个节点中的8个和图分类基准上实现了新的最先进的自监督学习结果。例如，在Cora(节点)和reddy - binary(图形)分类基准上，我们实现了86.8%和84.5%的准确率，相对于之前的最先进水平分别提高了5.5%和2.4%。与监督基准相比，我们的方法在8个基准中有4个优于监督基准。
  
14. **题目**：[Object Relational Graph with Teacher-Recommended Learning for Video Captioning](https://arxiv.org/pdf/2002.11566.pdf)  
  **作者及接受期刊/会议**：CVPR2020，ORG-TRL：基于物体关系图和教师推荐学习的视频描述   
  **摘要**：视频描述任务作为一种结合视觉与语言的高层任务，近年来得到研究者的广泛关注。该任务的关键包括如何充分提取视频中关键的视觉信息和如何构建从视觉到语言的映射关系。自动化所视频内容安全团队张子琦、史雅雅博士等人针对当前视频描述任务中，视觉层面缺少物体之间的交互关系，以及语言层面缺乏与内容相关单词的充分训练等问题，提出了基于物体关系图和教师推荐学习的视频描述系统。图1为该系统的框架图，该系统利用图卷积神经网络获取视频中物体间的相互关联，同时利用知识蒸馏的方法将外部语言模型的知识传授给视频描述模型。团队所提出的模型和训练策略，能够充分挖掘物体之间的关联，有效吸纳外部语言模型的语言学知识。在三个大型视频描述数据集（MSVD、MSR-VTT和VATEX）上均验证了该系统的有效性。
  
15. **题目**：[赛尔笔记 | 多模态信息抽取简述](https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247524098&idx=6&sn=4585f6a1e00a3d3b9360476db7002a2d&chksm=fc867e11cbf1f707211c4ab8d72f31deff7cb51dfb2975aaf58f172c720b577025da613ac8ed&scene=0&xtrack=1&key=f74daa37e4bd8ddc7a8d4ba238316b5106a21002f8de5f7b70419d2c5820ae0d849c8ef87a602717bd98d49d9df41d7406a36a7553bd29a0fa57fb03e267239f0f7415278b3252e7a4ace34706a96cd2&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10+x64&version=62090523&lang=zh_CN&exportkey=A5Fa2JDTtV0iJble5bUXTLY%3D&pass_ticket=j%2FylNa%2FNKHDOVnthS3zCYteKHaC%2FBaJihtPsKLSPzLN4M6KsOQOucf7E4PKqTuc6)

16. **题目**：[BERT在多模态领域中的应用](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247503902&idx=1&sn=eb0e9a512c69c7cfd42ab89d4e8c202d&chksm=96ea0f9ea19d868886eef0798186a553da5dbb03000126acf4af992d9d8e9e5d0b4531ae453e&mpshare=1&scene=24&srcid=&sharer_sharetime=1583384019015&sharer_shareid=0091823d7fc0a9eb9f042a61c73cc17a&key=fb6555550f80da0ccd0d60a2f93f7d0ad9fd9e1ca66b9ef4df7c382711b9a93738e97198c3933bcbb186d139a6b66a3f44b6fcc6b530101560aba176ab1304b5f276baf81335222ef642d3b0a19ed06a&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10+x64&version=62090523&lang=zh_CN&exportkey=A5RLs5sWQxiLxWDS9rffUcg%3D&pass_ticket=j%2FylNa%2FNKHDOVnthS3zCYteKHaC%2FBaJihtPsKLSPzLN4M6KsOQOucf7E4PKqTuc6)

17. **题目**：[让机器读懂视频：亿级淘宝视频背后的多模态AI算法揭秘](https://mp.weixin.qq.com/s?__biz=MzIwMTE1NjQxMQ==&mid=2247490231&idx=1&sn=2b750b1143272344939e8f626778d5df&chksm=96f368e3a184e1f5f8a44db8a239b1d7952c39f0b319b885e57aae648a681029eaf9baf068ba&scene=0&xtrack=1&key=46828b3a71d4b504d8480d28c4a9597623bbac0c507736f14af212f8f08f27f756a2b8d4372fbceaa9ea6ddc885568ac833c33ca56ec98449943757a1434a9674bdfe4dbaef21da98bb6a638f5120e47&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10+x64&version=62090523&lang=zh_CN&exportkey=A6HPGxTEBiiM%2FEm7rp%2F%2Bb4Q%3D&pass_ticket=j%2FylNa%2FNKHDOVnthS3zCYteKHaC%2FBaJihtPsKLSPzLN4M6KsOQOucf7E4PKqTuc6)

18. **题目**：[多模态情感分析简述](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247490881&idx=5&sn=abed4c6bbf3de75889e29c8f1c19267d&chksm=ebb42595dcc3ac836f82182c48a0f147663e9bee5a853e447379bbdc03092f46ff0941ad039d&scene=0&xtrack=1&key=835f76b9f6281b48135da4ee5e5acc81347b450c040bc2dba259f38d955918f82e6d3bc4bb6e9320c3a8110de3fd04a2e58983fc973d17d1024d2b49a2ced1fc34a5f9008f153d149ecf256106b1e513&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10+x64&version=62090523&lang=zh_CN&exportkey=A6YYvKl8zQPcoG9N7jV8%2F5I%3D&pass_ticket=j%2FylNa%2FNKHDOVnthS3zCYteKHaC%2FBaJihtPsKLSPzLN4M6KsOQOucf7E4PKqTuc6)

19. **题目**：[VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530)  
  **作者及接受期刊/会议**：ICLR 2020，微软亚洲研究院  
  **摘要**：研究员提出了一种新的通用的多模态预训练模型VL-BERT，该模型采用简单而强大的Transformer模型作为主干网络，并将其输入扩展为同时包含视觉与语言输入的多模态形式，适用于绝大多数视觉语义下游任务。为了让VL-BERT模型利用更为通用的特征表示，研究员在大规模图片描述生成数据集Conceptual Captions中进行VL-BERT的预训练，实验证明此预训练过程可以显著提高下游的视觉语义任务的效果，包含视觉常识推理、视觉问答与引用表达式理解等。
  
20. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
21. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
22. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
23. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
24. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
25. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
26. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
27. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
28. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
29. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
  
30. **题目**：[]()  
  **作者及接受期刊/会议**：  
  **摘要**：
