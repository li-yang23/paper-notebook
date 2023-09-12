1. [Graph Entropy Guided Node Embedding Dimension Selection for Graph Neural Networks](https://www.aminer.cn/pub/6099004f91e011aa8bcb6de9/graph-entropy-guided-node-embedding-dimension-selection-for-graph-neural-networks?conf=IJCAI%202021)：
   
   * 图神经网络 理论分析（嵌入维度选择）
   
   * 解决已有维度选择方法（网格搜索，先验知识）计算量大，表现差的问题。从最小熵角度考虑维度选择问题，并提出了最小化图熵算法。

2. [The Surprising Power of Graph Neural Networks with Random Node Initialization](https://www.aminer.cn/pub/5f7c344591e0117ac2a78854/the-surprising-power-of-graph-neural-networks-with-random-node-initialization?conf=IJCAI%202021):
   
   * 图神经网络 理论分析
   
   * 超出1-WL能力的GNN（高阶等变和不变GNN）计算量大，普通GNN实用性强。研究随机初始化节点特征的GNN，证明随机初始化可以达到高阶GNN的效果
   
   * 等变：对输入施加的变换会同样反应在输出上。不变：输入施加的变换不会影响到输出

3. [On Self-Distilling Graph Neural Network](https://www.aminer.cn/pub/5fa3f83991e011a6939b81e2/on-self-distilling-graph-neural-network?conf=IJCAI%202021):
   
   * 图神经网络 知识蒸馏
   
   * 解决已有方法因为训练大规模模型困难很难得到满意的老师网络，师生网络容易训练不充分的问题。提出了自蒸馏网络

4. [Multi-hop Attention Graph Neural Networks](https://www.aminer.cn/pub/60da8fc20abde95dc965f83f/multi-hop-attention-graph-neural-networks?conf=IJCAI%202021):
   
   * 图注意力网络
   
   * 解决已有图注意力机制只关注直接相连的节点，而不管多跳外的关系的问题。**在注意力值上使用传播先验来有效计算非直接相连节点间的注意力**。

5. [Graph-Free Knowledge Distillation for Graph Neural Networks](https://www.aminer.cn/pub/60a3a42e91e01115219ffa36/graph-free-knowledge-distillation-for-graph-neural-networks?conf=IJCAI%202021):
   
   * 图神经网络 知识蒸馏
   
   * 解决知识蒸馏中数据样本难以获得的问题，提出不用图数据的GNN知识蒸馏，通过多项式分布建模迁移学习图结构信息

6. [CSGNN - Contrastive Self-Supervised Graph Neural Networkfor Molecular Interaction Prediction](https://www.aminer.cn/pub/60da8fc20abde95dc965f885/csgnn-contrastive-self-supervised-graph-neural-network-for-molecular-interaction-prediction?conf=IJCAI%202021):
   
   * 图神经网络 自监督 分子反应预测 泛化能力
   
   * 解决原有分子反应预测只关注某个领域，泛化能力差的问题，使用多跳邻居聚合+GNN+自监督正则+多任务学习框架学习泛化能力强的GNN分子反应预测模型

7. [Learning Unknown from Correlations - Graph Neural Networkfor Inter-novel-protein Interaction Prediction](https://www.aminer.cn/pub/60a2401291e0115ec77b9cd9/learning-unknown-from-correlations-graph-neural-network-for-inter-novel-protein-interaction?conf=IJCAI%202021):
   
   * 图神经网络 蛋白质反应预测 泛化能力
   
   * 解决当前评价指标忽视新蛋白质之间反应的预测导致模型在新数据集上表现差的问题，设计新框架和模型考虑新蛋白质反应和已有蛋白质之间关系的信息

8. [GraphMI - Extracting Private Graph Data from Graph Neural Networks](https://www.aminer.cn/pub/60c16bd191e0112cf43c1f08/graphmi-extracting-private-graph-data-from-graph-neural-networks?conf=IJCAI%202021):
   
   * 图神经网络 隐私分析 图攻击（有点像解释性，先解释再攻击）
   
   * 解决模型倒推技术在图上无法直接应用的问题，通过模型倒推分析隐私训练数据，提出编码解码模型分析图结构节点属性和边预测任务的模型参数

9. [TrafficStream - A Streaming Traffic Flow Forecasting Framework Based on Graph Neural Networks and Continual Learning](https://www.aminer.cn/pub/60c80a6191e0110a2be2393f/trafficstream-a-streaming-traffic-flow-forecasting-framework-based-on-graph-neural-networks?conf=IJCAI%202021):
   
   * 交通流预测 图神经网络
   
   * 解决已有方法忽视扩张和进化模式，只关注时空关系的问题，基于GNN和连续学习提出了流式交通流预测框架

10. [Node-wise Localization of Graph Neural Networks](https://www.aminer.cn/pub/60da8fc20abde95dc965f807/node-wise-localization-of-graph-neural-networks?conf=IJCAI%202021):
    
    * 图神经网络
    
    * 解决GNN聚合函数不能自适应考虑节点聚合范围的问题，提出了节点局部化GNN（node-wise localization），局部来看节点通过一个全局GNN中作为函数的模型聚合局部信息

11. [Pairwise Half-graph Discrimination - A Simple Graph-level Self-supervised Strategy for Pre-training Graph Neural Networks](https://www.aminer.cn/pub/60da8fc20abde95dc965f72d/pairwise-half-graph-discrimination-a-simple-graph-level-self-supervised-strategy-for?conf=IJCAI%202021):
    
    * 图神经网络 自监督 预学习
    
    * 解决预训练可迁移，泛化能力强且稳定的GNN模型的问题，提出新学习策略，判断两个半图是否来自同一个数据源

12. [Residential Electric Load Forecasting via Attentive Transfer of Graph Neural Networks](https://www.aminer.cn/pub/60da8fc20abde95dc965f8c3/residential-electric-load-forecasting-via-attentive-transfer-of-graph-neural-networks?conf=IJCAI%202021):
    
    * 电力负荷预测 多变量时间序列预测
    
    * 解决已有方法忽略同一区域内可能受相同因素影响的问题，

13. [Multi-Channel Pooling Graph Neural Networks](https://www.aminer.cn/pub/60da8fc20abde95dc965f89d/multi-channel-pooling-graph-neural-networks?conf=IJCAI%202021):
    
    * 图神经网络 图池化
    
    * 解决粗糙池化只考虑全局结构，下坠池化只考虑局部结构，没有进行权衡的问题，提出了新的池化方法，同时考虑全局和局部结构还有节点特征。

14. [GraphReach - Position-Aware Graph Neural Network using Reachability Estimations](https://www.aminer.cn/pub/60da8fc20abde95dc965f84b/graphreach-position-aware-graph-neural-network-using-reachability-estimations?conf=IJCAI%202021):
    
    * 图神经网络 位置敏感
    
    * 解决已有方法不关心节点位置只关心局部结构的问题，提出模型，根据锚点捕捉全局位置信息

15. [Blocking-based Neighbor Sampling for Large-scale Graph Neural Networks](https://www.aminer.cn/pub/60da8fc20abde95dc965f7cd/blocking-based-neighbor-sampling-for-large-scale-graph-neural-networks?conf=IJCAI%202021):
    
    * 图神经网络 可伸缩性
    
    * 解决训练大规模GNN时的计算量和存储暴增问题，设计基于块的采样方式来有效训练GNN，自适应地阻止即将出现的邻域暴增，并自适应地调整邻域内节点的权重

16. [Convexified Graph Neural Networks for Distributed Control in Robotic Swarms](https://www.aminer.cn/pub/60da8fc20abde95dc965f7e6/convexified-graph-neural-networks-for-distributed-control-in-robotic-swarms?conf=IJCAI%202021):
    
    * 图神经网络 机器人群 分布式控制

17. [Unsupervised Knowledge Graph Alignment by Probabilistic Reasoning and Semantic Embedding](https://www.aminer.cn/pub/609d02d191e01118a99b93d8/unsupervised-knowledge-graph-alignment-by-probabilistic-reasoning-and-semantic-embedding?conf=IJCAI%202021):
    
    * 知识图谱 图谱对齐 
    
    * 解决已有方法要么需要大量理想的已标记数据，或者无法有效利用图结构和实体语义的问题。提出基于概率推理和语义嵌入的框架合并两种方法的优势

18. [Graph Deformer Network](https://www.aminer.cn/pub/60da8fc20abde95dc965f865/graph-deformer-network?conf=IJCAI%202021):
    
    * 图卷积 卷积滤波
    
    * 解决图卷积只用加和/均值进行聚合容易导致结构损失和节点信号纠缠的问题，提出了一种新的过滤操作。

19. [Self-Guided Community Detection on Networks with Missing Edges](https://www.aminer.cn/pub/60da8fc20abde95dc965f878/self-guided-community-detection-on-networks-with-missing-edges?conf=IJCAI%202021):
    
    * 社团检测
    
    * 解决已有方法无法在边缺失的网络上进行社团检测的问题（链接预测和社团发现结合的方法有链接预测方法目的和社团检测方法不一致的问题）。提出一种自指导方法，同步生成网络并识别社团

20. [Graph Universal Adversarial Attacks - A Few Bad Actors Ruin Graph Learning Models](https://www.aminer.cn/pub/5e451e433a55acfaed738769/graph-universal-adversarial-attacks-a-few-bad-actors-ruin-graph-learning-models?conf=IJCAI%202021)：
    
    * 图攻击 图防御
    
    * 解决GNN在学习bad actor节点后易受攻击的问题，设计方法从图中识别bad actor节点

21. [Graph Consistency Based Mean-Teaching for Unsupervised Domain Adaptive Person Re-Identification](https://www.aminer.cn/pub/609ba97491e0113c3c76931b/graph-consistency-based-mean-teaching-for-unsupervised-domain-adaptive-person-re-identification?conf=IJCAI%202021)
