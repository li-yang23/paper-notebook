1. [Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization](https://www.aminer.cn/pub/5f5f378a91e0117a861e8942/transfer-learning-of-graph-neural-networks-with-ego-graph-information-maximization?conf=NeurIPS%202021):
   
   * 图神经网络 迁移学习（预训练）
   
   * 使用局部图拉普拉斯独描述了原图和目标图之间的可迁移性

2. [Adaptive Diffusion in Graph Neural Networks](https://www.aminer.cn/pub/61a881436750f87bf8702032/adaptive-diffusion-in-graph-neural-networks?conf=NeurIPS%202021):
   
   * 图神经网络 图扩散卷积
   
   * 解决图扩散卷积方法需要手动调整扩散域的问题，设计方法为每个节点自动选择最适合的扩散域

3. [Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration](https://www.aminer.cn/pub/61552aed5244ab9dcb23ec5f/be-confident-towards-trustworthy-graph-neural-networks-via-confidence-calibration?conf=NeurIPS%202021):
   
   * 图神经网络 可信度矫正（？这是啥）
   
   * 解决图神经网络模型预测结果欠置信度问题，设计模型进行置信度矫正

4. [Robust Counterfactual Explanations on Graph Neural Networks](https://www.aminer.cn/pub/60ebec3491e0119571373fd1/robust-counterfactual-explanations-on-graph-neural-networks?conf=NeurIPS%202021):
   
   * 图神经网络 可解释性
   
   * 解决已有可解释性方法对于噪声不稳定的问题，从反事实角度进行解释，设计方法生成稳定的反事实解释。

5. [Graph Neural Networks with Local Graph Parameters](https://www.aminer.cn/pub/60caa52991e011b329373e9d/graph-neural-networks-with-local-graph-parameters?conf=NeurIPS%202021):
   
   * 图神经网络 表达能力
   
   * 解决高阶GNN学习时因保存高阶邻接节点导致的高内存需求所以应用性不高的问题，使用局部图参数保存节点的高阶信息，从而避免保存大量节点

6. [Decoupling the Depth and Scope of Graph Neural Networks](https://www.aminer.cn/pub/6197105b6750f82d7d73b020/decoupling-the-depth-and-scope-of-graph-neural-networks?conf=NeurIPS%202021):
   
   * 图神经网络 可伸缩性 设计原则
   
   * 解决深度图神经网络出现的过平滑导致的表达能力欠缺和邻域爆炸导致的高计算消耗问题
   
   * 给每个节点（或边）提取一个子图，并只在这个子图上进行传播

7. [Dirichlet Energy Constrained Learning for Deep Graph Neural Networks](https://www.aminer.cn/pub/60e565dadfae54c432544012/dirichlet-energy-constrained-learning-for-deep-graph-neural-networks?conf=NeurIPS%202021):
   
   * 图神经问题 过平滑问题 设计原则

8. [DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks](https://www.aminer.cn/pub/618ddb455244ab9dcbda8ece/dropgnn-random-dropouts-increase-the-expressiveness-of-graph-neural-networks?conf=NeurIPS%202021):
   
   * 图神经网络 训练方法
   
   * 解决已有GNN框架的缺陷（还不知道是什么），训练多次同一个GNN，每次独立随机抛掉一部分节点

9. [SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks](https://www.aminer.cn/pub/6023dddb91e0119b5fbd9902/slaps-self-supervision-improves-structure-learning-for-graph-neural-networks?conf=NeurIPS%202021):
   
   * 图生成  图构建 图神经网络 自监督
   
   * 解决大量节点时根据指定任务监督生成节点结构困难的问题，使用自监督同时学习节点结构和GNN参数

10. [Robustness of Graph Neural Networks at Scale](https://www.aminer.cn/pub/617a16075244ab9dcbdb3924/robustness-of-graph-neural-networks-at-scale?conf=NeurIPS%202021):
    
    * 图攻击
    
    * 解决在大规模网络上的图攻击问题，提出了两种攻击方法，设计了一种稳定聚合函数

11. [On the Universality of Graph Neural Networks on Large Random Graphs](https://www.aminer.cn/pub/60b19b5491e0115374595764/on-the-universality-of-graph-neural-networks-on-large-random-graphs?conf=NeurIPS%202021):
    
    * 图神经网络 表达能力
    
    * 研究在隐位置随机网络上的拟合能力，**结构GNN**在大规模随机图上的表达能力

12. [GemNet: Universal Directional Graph Neural Networks for Molecules](https://www.aminer.cn/pub/60cd6b7691e011329faa227b/gemnet-universal-directional-graph-neural-networks-for-molecules?conf=NeurIPS%202021):
    
    * 图神经网络 链接预测 位置敏感图神经网络
    
    * 填补GNN在分子反应任务上的理论分析，提出了几何信息传递网络

13. [Subgroup Generalization and Fairness of Graph Neural Networks](https://www.aminer.cn/pub/60dd3f2b91e011cc85cbcd08/subgroup-generalization-and-fairness-of-graph-neural-networks?conf=NeurIPS%202021):
    
    * 图神经网络 泛化（生成）能力 理论分析
    
    * 提出GNN半监督学习下的PAC-贝叶斯分析，分析了不同子群组上的泛化能力区别。子群组和训练集之间的距离会影响泛化能力，因此需要谨慎选择训练节点

14. [Neo-GNNs: Neighborhood Overlap-aware Graph Neural Networks for Link Prediction](https://www.aminer.cn/pub/61a887006750f87bf870220b/neo-gnns-neighborhood-overlap-aware-graph-neural-networks-for-link-prediction?conf=NeurIPS%202021):
    
    * 图神经网络 链接预测
    
    * 解决GNN在对结构信息敏感的链接预测任务上表现不如启发式方法的问题，提出了邻域重叠敏感GNN，从邻接矩阵学习结构特征，并评估邻域重叠程度（用LINE就行？）

15. [Graph Neural Networks with Adaptive Residual](https://www.aminer.cn/pub/61a88c016750f8304711e98d/graph-neural-networks-with-adaptive-residual?conf=NeurIPS%202021):
    
    * 图神经网络 新模型 残差链接
    
    * 解决残差链接对异常特征很脆弱的问题，设计新的自适应消息传递机制，和有自适应残差链接的GNN

16. [VQ-GNN: A Universal Framework to Scale up Graph Neural Networks using Vector Quantization](https://www.aminer.cn/pub/617a16125244ab9dcbdb3f26/vq-gnn-a-universal-framework-to-scale-up-graph-neural-networks-using?conf=NeurIPS%202021):
    
    * 图神经网络 可伸缩性
    
    * 解决基于采样的方法无法获得多跳之外的节点的问题，提出使用向量离散化方式，通过学习和更新一小部分离散化的全局节点表示向量保存传入一个mini batch的所有信息

17. [Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning](https://www.aminer.cn/pub/61a886156750f87bf87021b7/labeling-trick-a-theory-of-using-graph-neural-networks-for-multi-node?conf=NeurIPS%202021):
    
    * 图神经网络 多节点表示学习（给节点集合学习表示，比如边）
    
    * 解决池化方法忽略节点集合内节点的依赖关系的问题，使用节点标签标记聚合方式，统一已有方法，提出一个泛化形式

18. [Metropolis-Hastings Data Augmentation for Graph Neural Networks](https://www.aminer.cn/pub/61a884d96750f87bf87020f3/metropolis-hastings-data-augmentation-for-graph-neural-networks?conf=NeurIPS%202021):
    
    * 图神经网络 数据增强
    
    * 解决因为标签稀少导致的泛化性能不佳问题，提出数据增强方法MH-Aug，为半监督学习任务从目标分布中获得增强后的图数据

19. [Automorphic Equivalence-aware Graph Neural Network](https://www.aminer.cn/pub/61a886b86750f87bf87021fd/automorphic-equivalence-aware-graph-neural-network?conf=NeurIPS%202021):
    
    * 图神经网络 自同构检测
    
    * 解决GNN的自同构检测问题，引入自中心自同构检测，然后使用自同构敏感的聚合函数区分节点不同邻接节点的自中心自同构

20. [Representing Long-Range Context for Graph Neural Networks with Global Attention](https://www.aminer.cn/pub/61a8843a6750f87bf87020bf/representing-long-range-context-for-graph-neural-networks-with-global-attention?conf=NeurIPS%202021):
    
    * 图神经网络 模型设计 长距离依赖
    
    * 解决图神经网络无法有效捕捉远距离依赖的问题，根据transformer注意力机制捕捉远距离关系，用新的readout函数获得全局图嵌入

21. 


