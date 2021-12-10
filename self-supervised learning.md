# Contents
- [Image](#image)
  - [Barlow Twins (2021-06)](#barlow-twins-self-supervised-learning-via-redundancy-reduction)
  - [VICReg (2021-10)](#vicreg-variance-invariance-covariance-regularization-for-self-supervised-learning)
  - [DirectCLR (2021-10)](#understanding-dimensional-collapse-in-contrastive-self-supervised-learning)
- [Video](#video)
  - [BE (2020-09)](#removing-the-background-by-adding-the-background-towards-background-robust-self-supervised-video-representation-learning)
  - [DSM (2020-09)](#enhancing-unsupervised-video-representation-learning-by-decoupling-the-scene-and-the-motion)
  - [ASCNet (2021-03)](#ascnet-self-supervised-video-representation-learning-with-appearance-speed-consistency)
- [Multi-modal](#multi-modal)
  - [GDT (2021-03)](#on-compositions-of-transformations-in-contrastive-self-supervised-learning)
  - [STiCA (2021-03)](#space-time-crop--attend-improving-cross-modal-video-representation-learning)
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

#### Background
- 互信息（MI）的下限与数据量程指数上升的关系
- 提高负样本的数量能够改善学到的表征，但是过多的负样本有时候也会伤害模型的性能

#### Motivation
- **Random negative sampling** leads to a highly redundant dictionary that results in suboptimal representations for downstream tasks.
  - 随机采样构建的字典会包含很多“有偏差”的key，即和query属于相同类别的相似的key，以及“不够有效”的key，即这些key模型能够很容易判别出来
  - 这种情况会在字典大小很大的情况下急剧恶化

#### Architecture
<div align="center">
<img src=resources/021.png width=100% />
</div>

#### Methods
- **Cross-Modal Contrastive Representation Learning**
  - 一共有两个query encoder和两个key encoder
  - 在visual-to-audio阶段，取一个视频片段$v^{query}$，需要在一个key队列中找到相应的音频$a^{key}$。在这个阶段只更新video的query encoder $f_v$，audio的key encoder由audio的query encoder $h_a$更新得来。
  - 在audio-to-visual阶段，进行相反的操作

- Active Sampling of Negative Instances: Uncertainty and Diversity

[$\to$back to contents](#contents)

### Contrastive Learning of Global and Local Video Representations
Shuang Ma, Zhaoyang Zeng, Daniel McDuff, Yale Song

