# GNN的可解释性

主要提供后验的解释，哪条边/节点/特征更加重要，哪种网络模式对指定类型的影响更大

## 个体级别的方法

研究根据输入确定的解释性，识别针对具体任务的图中的重要特征，根据对于重要性分数的计算方式进行具体分类

### 基于梯度/特征的方法

使用梯度或者特征值来表示不同输入特征的重要性，基于梯度的方法计算目标预测对于输入特征的梯度，梯度越大说明重要性越强。基于特征值的方法将隐层特征映射回输入空间来衡量重要性。

+ [x] [Deep inside convolutional networks: Visualising image classification models and saliency maps](https://arxiv.org/abs/1312.6034.pdf)：固定学习到的网络参数，计算输出概率和输入特征之间的梯度，得到类别显著性映射，用来反映图中每个像素对于特定类别识别结果的重要性

+ [x] [Smoothgrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825.pdf)：基于梯度的方法，随机从高斯分布中采样多个值，然后加在原输入上，并求梯度的平均值作为输入的重要性，用于降低噪声影响

  > 其实没太明白怎么加随机采样然后再计算梯度，说是从输入的领域中随机采样，然后计算平均敏感性映射

+ [x] [Learning deep features for discriminative localization](http://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html)：研究均值池化层如何使CNN可以捕捉局部特征，生成一个类别活动映射来表示CNN用于识别特定类别的图像区域。因为最后卷积层经过全局池化再接一个全连接层接softmax来输出类别，所以全联接层的权重代表了特定卷积单元的重要性，可以推出在$\sum_kw_k^cf_k(x,y)$代表了图片位置$(x,y)$对预测结果$c$的重要性

+ [x] [Grad-cam: Visual explanations from deep networks via gradient-based localization](http://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)：计算输出预测与特征图中每个元素的映射，并用全局均值作为系数，在与特征图进行加权后使用relu函数得到特征图中每个元素对于分类结果的影响程度。

+ [x] [Explainability techniques for graph convolutional networks](https://arxiv.org/abs/1905.13686)：提出了基于梯度和基于分解的方法。SA直接使用梯度的开方来衡量输入特征的重要性，Guided BP之反向传播正梯度并将负梯度归零。LRP将输出信号分解为输入特征的组合，并将组合权重作为重要性分数。

+ [x] [Explainability methods for graph convolutional neural networks](http://openaccess.thecvf.com/content_CVPR_2019/html/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.html)：将CAM和Grad-CAM引入了图卷积网络，计算的东西变成了每列特征值，最终得到节点每个特征对于指定分类的重要程度，进而得到每个节点的重要程度。

### 基于转换的方法

通过观察打乱输入后输出结果的变化来判断重要性，如果重要的特征没动的话最终结果应当相似。用一个遮罩层来遮盖重要特征，生成一个新图，然后将新图也送入同一GNN，根据最终得到的结果更新遮罩层。

> 这个适用于大型网络吗，遮罩层存的进内存么

+ [ ] [Real time image saliency for black box classifiers](https://arxiv.org/abs/1705.07857)：后面没有全看懂，但是方法是初始化一个遮罩层，然后遮罩掉一部分输入特征，并和另一个图的反向遮罩结果相加，最终通过优化遮罩层得到对于特征重要性的解释。被遮掉的就是不重要的特征。
+ [ ] [Interpreting image classifiers by generating discrete masks](https://ieeexplore.ieee.org/abstract/document/9214476/)：采用GAN结构，利用要解释的GN N做判别器，生成器用来得到遮罩层，使用一个概率图来采样得到离散遮罩层，然后得到输入图，用GNN学习输出判断结果。使用policy gradient来更新生成器，最终生成器得到的就是可以保留重要信息的遮罩层，可以过滤出重要的输入特征
+ [ ] 60
+ [ ] 42
+ [ ] 43
+ [ ] 51
+ [ ] 52
+ [ ] 53

### 基于分解的方法

通过将任务指定的指标逐层反向分解，得到特征的重要性分数

### 基于替代的方法

首先在样本处采样相邻节点，然后在采样的途中使用一个简单的可解释模型来解释原本的预测

## 模型级别的方法

研究与特定输入无关的对于GNN的解释。通过生成图模式的方式最大化特定任务的预测结果的方式来使用图模式来表示对对应类的重要程度

> 可以理解为这个GNN对于这个类的判断方式就是这个图模式

