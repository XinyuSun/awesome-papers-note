# Contents
- [Image](#image)
  - [Barlow Twins (2021-06)](#barlow-twins-self-supervised-learning-via-redundancy-reduction)
  - [VICReg (2021-10)](#vicreg-variance-invariance-covariance-regularization-for-self-supervised-learning)
  - [DirectCLR (2021-10)](#understanding-dimensional-collapse-in-contrastive-self-supervised-learning)
- [Video](#video)
  - [BE (2020-09)](#removing-the-background-by-adding-the-background-towards-background-robust-self-supervised-video-representation-learning)
  - [DSM (2020-09)](#enhancing-unsupervised-video-representation-learning-by-decoupling-the-scene-and-the-motion)
  - [ASCNet (ICCV 2021)](#ascnet-self-supervised-video-representation-learning-with-appearance-speed-consistency)
  - [CORP (ICCV 2021)](#contrast-and-order-representations-for-video-self-supervised-learning)
  - [MCN (ICCV 2021)](#self-supervised-video-representation-learning-with-meta-contrastive-network)
  - [LSFD (ICCV 2021)](#long-short-view-feature-decomposition-via-contrastive-video-representation-learning)
  - [TEC (ICCV 2021)](#time-equivariant-contrastive-video-representation-learning)
- [Multi-modal](#multi-modal)
  - [GDT (2021-03)](#on-compositions-of-transformations-in-contrastive-self-supervised-learning)
  - [STiCA (2021-03)](#space-time-crop--attend-improving-cross-modal-video-representation-learning)
  - [CM-ACC (2021-04)](#active-contrastive-learning-of-audio-visual-video-representations)
## Image

### Barlow Twins: Self-Supervised Learning via Redundancy Reduction
Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun

<div align="center">
<img src=resources/002.png width=100% />
</div>

- Making the __cross-correlation matrix__ between the output features of two identical networks as close to the identity matrix.
- Minimizing the __redundancy__ between the components of the output vectors.
- Does not require tricks used in recent CLR methods such as stopping gradient, momentem encoder or predictor network.

[$\to$back to contents](#contents)

### VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
Adrien Bardes, Jean Ponce, Yann LeCun
- Explicitly avoids the collapse problem in SSL methods with two regularizations term.
- __Invariance__: minimize distance between two views from one sample.
- __Variance__: maintain the distinct of each embedding in a batch.
- __Covariance__: decorrelating the variables between pairs of embedding variables.

<div align="center">
<img src=resources/001.png width=100% />
</div>

[$\to$back to contents](#contents)

### Understanding Dimensional Collapse in Contrastive Self-Supervised Learning
- Instead of complete collapse, contrastive methods still experience a __dimensional collapse__.

- The dimensional collapse is caused by (1)strong augmentation and (2)implicit regularization.

- Proposed a novel contrastive self-supervised learning method `DirectCLR` that calculate loss using only a part of the representation.

[$\to$back to contents](#contents)

## Video
### Removing the Background by Adding the Background: Towards Background Robust Self-supervised Video Representation Learning
Jinpeng Wang, Yuting Gao, Ke Li, Yiqi Lin

[$\to$back to contents](#contents)

### Enhancing Unsupervised Video Representation Learning by Decoupling the Scene and the Motion
- Some action categories are highly related with the __scene__ where the action happens, making the model tend to degrade to a solution where __only the scene information is encoded__.
- Proposed to decouple the scene and the motion with a clip pair that contain a __motion-broken clip and a scene-broken clip__.

[$\to$back to contents](#contents)

### ASCNet: Self-supervised Video Representation Learning with Appearance-Speed Consistency
#### Architecture
<div align="center">
<img src=resources/016.png width=100% />
</div>

#### Methods
- ACP taks: pull the appearance features from the same video with different playback speed together
- SCP task: 
  - __Retrieval the same speed videos with similar contents__
    - Top-1 similarity selection from __memory bank__
  - Pull the speed features together

#### Contributions
- Negative samples no longer affect the quality of learned representations
  - Only pull the positive pairs together
  - Using $L_2$ loss (actually is a predictor that predict one view from another view)
- Appearance-focus feature retrieval strategy to select more effctive positive samples for speed consistency perception

#### Results
- Finetuned on UCF-101, pretrained on Kinetics400
<div align="center">
<img src=resources/017.png width=100% />
</div>

[$\to$back to contents](#contents)

### Contrast and Order Representations for Video Self-supervised Learning
#### Motivation
- Recent proposed contrastive based methods simply utlize two augmented clips from the same videos and compare their distance without referring to their temporal relation
- There should be some other ways to capture temporal information across different frames
  
#### Proposal
- Contrast-and-order Representation (CORP)
  - First predict if two video clips come from the same input video
  - Then predict the temporal ordering of the clips if they come from the same video
  - 在对比学习任务之外加上一个分类任务，预测两个clip哪个先发生
- Decoupling attention method
  - Symmetric similarity (contrast) and anti-symmetric patterns (order)

#### Overview
<div align="center">
<img src=resources/023.png width=100% />
</div>

- 两种实现方式，第一种从数据集采样两个视频，从每个视频采样K个片段，一共有$2K(2K-1)$个pairs，每一对pair都要去预测是否属于同一个视频以及时间的顺序，即每一对pair的特征经过一个attention之后进行一个三分类任务
- 第二种实现方式中，从数据集采样$B$个视频，每个视频采样两个增强的clip，一共$2B$个clips，每一个clip都需要和剩下的$2B-1$个clip的特征去计算attention，然后进行一个$2(2B-1)$分类任务，找出和它属于相同视频的片段并判定先后次序

#### Methods
##### CORP$_m$ Model
- Possibile categories:
  - $\mathcal{P}_1$：两个片段采样来自不同的视频样本
  - $\mathcal{P}_1$：两个片段采样来自相同视频样本，并且第一个片段发生在第二个片段之前
  - $\mathcal{P}_1$：两个片段采样来自相同视频样本，并且第一个片段发生在第二个片段之后
- Multi-head attention:
  - $r(z_i,z_j)=[\langle U_1z_i,V_1z_j\rangle,\cdots,\langle U_hz_i,V_hz_j\rangle]\in \mathbb{R}^h$
- Probability distribution:
  - $\phi(z_i,z_j)=mlp(r(z_i,z_j))\in\mathbb{R}^3$
- Loss function:
  - Summation of the classification loss of all pairs
- Data processing
  - k个clip彼此之间不能太过靠近，否则会使得预测顺序的任务过于简单

##### CORP$_f$ Model
- Video Sampling
  - Sample B video as a batch, further sample two clips from each video
- (4B-2)-way classification:
  - 这里相当于是在contrastive loss的基础上多做了一个预测顺序的分类，所以总共有$2\times(2B-1)$种可能的预测值这里相当于是在contrastive loss的基础上多做了一个预测顺序的分类，所以总共有$2\times(2B-1)=4B-2$个类别
- Loss function
  - $\mathcal{L}_i=\mathcal{L}_{NCE}+\mathcal{L}_{2-way\ order\ classification}$

##### Decoupling Attention
- Symmetric patterns vs. non-symmetric patterns
  - **对称范式 Symmetric**: if x is similar to y, y is similar to x (contrast task)
  - **非对称范式 Non-symmetric**: if x is earlier than y, y is later than x (for order task, is anti-symmetric)

[$\to$back to contents](#contents)

### Self-Supervised Video Representation Learning with Meta-Contrastive Network
#### Motivation
- Conttastive based methods lack of categories information, which will lead to **hard-positive problem** that constrains the generalization ability
- 相同动作类别的视频可能包含截然不同的场景和物体，而不同动作类别的视频反而可能包含相似这些这些信号

[$\to$back to contents](#contents)

### Long Short View Feature Decomposition via Contrastive Video Representation Learning
#### Background
- Stationary features:
  - remain similar throughout the video, **enable the prediction of the video level action class**
- Non-stationary features:
  - represent temporal varying arttibute, more benificial for downstream tasks involving **more fine-grained temporal understanding** (action segmentation)

#### Motivation
- A single representation to capture both types of features is
sub-optimal, we can **decompose** the representation space into stationary and non-stationary features via contrastive learning **from long and short view**
- 对于长时和短时视频片段的表征空间，我们可以分解为长时视角的静态特征和短时视角的非静态特征，从而得到更好的表征


#### Overview
<div align="center">
<img src=resources/024.png width=100% />
</div>

- 在long view 和 short view中，stationary feature是相似的，他们可以直接作为正样本
- short view 的 non-stationary feature 可以组合起来得到long view的正样本

#### Methods
- Stationary and Non-Stationary Features
  - $f_\theta(x)=\xi=(\psi,\phi)$
  - $\psi$ is stationary feature, $\phi$ is non-stationary feature
  - Aggregation funciton:
    - `linear`, `MLP`, `RNN` 
- Training objective
  - $\mathcal{L}=\mathcal{L}_{\mathbf{\rm stationary}}+\mathcal{L}_{\rm non-stationary}+\mathcal{L}_{\rm instance}$
  - Negative set: consist of random videos

#### Experiments


#### Conclusion
- Stationary feature包含比较多静态的特征，对action recognition有用
- Non-stationary feature包含比较多在时间上不同的内容，对action segmentation比较有用

[$\to$back to contents](#contents)

### Time-Equivariant Contrastive Video Representation Learning
#### Motivation
#### Overview
<div align="center">
<img src=resources/026.png width=100% />
</div>

- 3种不同的temporal transformations：
  - 2倍速采样的片段，覆盖范围比1倍速大一倍
  - 正常速度片段，分为正向播放和逆向播放 
  - 片段之间是否发生重合

<div align="center">
<img src=resources/025.png width=100% />
</div>

- 将两个视频片段特征用于分类任务：
  - non-overlapping with correct order
  - overlapping
  - non-overlapping with incorrect order

[$\to$back to contents](#contents)

## Multi-modal
### On Compositions of Transformations in Contrastive Self-Supervised Learning
Mandela Patrick, Yuki M. Asano, Polina Kuznetsova, Ruth Fong
#### Architecture
<div align="center">
<img src=resources/019.png width=40% />
</div>

#### Methods
- Hierarchical sampling process of generalized data transformations(GDT)
  - A wider set of transformations for which either __invariance and distinctiveness__ is sought.
  - Each sample is process with the combination of __five__ transformations
    - data-sampling($t_1$), time-shift($t_2$), modality splicing($t_3$), time-reversal($t_4$), augmentation transformations($t_5$)

[$\to$back to contents](#contents)

### Space-Time Crop & Attend: Improving Cross-modal Video Representation Learning
Mandela Patrick, Po-Yao Huang, Ishan Misra

### Active Contrastive Learning of Audio-Visual Video Representations
Shuang Ma, Zhaoyang Zeng, Daniel McDuff, Yale Song

#### Overview
通过随机采样获取负样本的方式会得到非常冗余的负样本，特别是采样数量很大的时候，这对对比学习是有害的。作者提出了一种主动采样的方式来得到**多样**而**信息量丰富**的负样本

#### Background
- 互信息（MI）的下限与数据量 呈 指数上升的关系
- 提高负样本的数量能够改善学到的表征，但是过多的负样本有时候也会伤害模型的性能
- **Active Learning** 主动学习：通过某种方法获取到比较“难”分类的样本数据，再进行人工标注，用于有监督或半监督学习模型进行训练，从而降低标注的成本

#### Motivation
- **Random negative sampling** leads to a highly redundant dictionary that results in suboptimal representations for downstream tasks.
  - 随机采样构建的字典会包含很多“有偏差”的key，即和query属于相同类别的相似的key，以及“不够有效”的key，即这些key模型能够很容易判别出来
  - 这种情况会在字典大小很大的情况下急剧恶化
- 通过主动地选取具有信息量的负样本来构建一个不冗余的字典

#### Architecture
<div align="center">
<img src=resources/021.png width=100% />
</div>

#### Methods
- **Cross-Modal Contrastive Representation Learning**
  - 视频和音频模态各有一个query encoder和一个key encoder
  - 在visual-to-audio阶段，取一个视频片段$v^{query}$，需要在一个key队列中找到相应的音频$a^{key}$。在这个阶段只更新video的query encoder $f_v$，audio的key encoder由audio的query encoder $h_a$更新得来
  - 在audio-to-visual阶段，进行相反的操作

- **Active Sampling of Negative Instances: Uncertainty and Diversity**
  - 让learner选择最信息量最高的负样本来构建字典
  - 不确定性(Uncertainty)
    - 损失函数相对于模型置信度最高的预测值的梯度，能够近似样本的不确定性，因此可以选取最后一层的梯度值来衡量不确定性
    - 鼓励模型选取具有最高梯度值的样本来构建字典
  - 多样性(Diversity)
    - 模型可能会对某些特定的语义类别具有很高的不确定性，如果总是从这些类别中选取样本会严重偏置梯度，最终让模型趋于局部最优
    - 通过kmeans++来多样化地采样负样本

- **Cross-Modal Active Contrastive Coding**
  - 

#### Reference
- [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds](https://arxiv.org/abs/1906.03671)

[$\to$back to contents](#contents)

### Contrastive Learning of Global and Local Video Representations
Shuang Ma, Zhaoyang Zeng, Daniel McDuff, Yale Song

