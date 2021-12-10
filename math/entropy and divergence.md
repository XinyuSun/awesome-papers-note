### Background
最近在一些工作中发现，KL散度衡量两个不同分布之间的相似程度，常用在神经网络模型的知识蒸馏(如DeiT[[4]](#ref-4))、孪生网络的对比学习方法(如ReSSL[[5]](#ref-5))中。此外，KL散度在深度生成模型如VAE中也有非常广泛的应用。
<div align="center">
<img src=../resources/020.png width=80% /> 
</div>
<center>图1. ReSSL结构图</center>
<br>
有趣的是，在查看这些方法的代码实现中，笔者发现KL散度损失函数是调用交叉熵损失来实现的，这似乎提示我们KL散度与交叉熵之间有着千丝万缕的联系。在这个背景下，笔者对交叉熵和KL散度的原始数学含义和在深度学习中的应用进行了调研和总结。

### Cross-Entropy[[1]](#ref-1)
在信息学中， __信息(information)__ 表示用于编码和传输一个事件所需的比特数。

一个事件发生的概率越小，不确定性越大，其包含的信息越多，我们称其给我们带来的“惊喜”越大。同样地，一个事件越有可能发生，其给我们带来的“惊喜”也就没那么大，其包含的信息越少。
> The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred.[[6]](#ref-6)

因此，我们可以通过事件发生的概率来计算事件的信息数量，这种计算方法称为 __香农信息(Shannon information)__ ，事件$x$的信息通常写为$h(x)$
$$h(x)=-\log(P(x))$$

其中，$\log$是以2为底的对数，这说明用来计算信息的单位是信息的二进制比特数。负号确保信息量的值是大于等于0的。

在数学中，我们会更关心变量(variable)的信息量。我们常用 __熵(entropy)__ 来衡量一个随机变量包含信息的数量。对于一个包含$n$个离散状态$x$的随机变量$X$，我们可以用下面的公式来计算它的熵：
$$H(X)=\sum_{i=0}^nP(x_i)\log P(x_i)$$

__交叉熵(cross-entropy)__ 是在信息论中熵的思想上构建的。它可以用于计算与另一个分布相比时，用于表示和传输一个分布的平均事件所需的比特数。
> the cross entropy is the average number of bits needed to encode data coming from a source with distribution p when we use model q.[[7]](#ref-7)

考虑我们有一个真实的目标分布$P$，以及一个对目标分布进行拟合得到的近似分布$Q$。这时，交叉熵$H(P,Q)$可以用来衡量， **当我想要用分布$Q$来表示分布$P$时，所需要额外的信息的比特数。**
$$H(P,Q)=-\sum_{i=0}^nP(x_i)\log(Q(x_i))$$

需要注意的是，如果采用2为底的对数，信息的单位是比特(bits)；如果采用e为底的对数，信息的单位是奈特(nats)。

上式用于计算离散的概率分布，如果需要计算服从连续概率分布的随机变量，则应该采用积分$\int$而不是求和$\sum$。

### Kl Divergence[[2]](#ref-2)
__KL散度(Kullback-Leibler Divergence)__，或称 __相对熵(relative entropy)__，同样用于量化给定随机变量对于不同概率分布之间的差异。

同样假设我们拥有一个随机变量，以及两个不同的概率分布，即一个真实的目标分布$P$和对真实分布的近似$Q$。我们可以用 __统计距离(statistical distance)__ 来衡量两个分布之间的差异，其中一种做法是直接计算两个分布的距离，但是如何去解释这个度量是非常麻烦的。因此更常见的做法是计算两个分布之间的 __散度(divergence)__。

散度也可以视为一种度量的方式，但是其和距离不同，它相对于$P$和$Q$不是对称的，计算$P$和$Q$的散度和计算$Q$和$P$的散度得到的结果是不同的。这在一些复杂的建模过程中是非常有用的工具，比如优化GAN模型的时候。

在信息论中，有两种常用的散度分数，包括：
- Kullback-Leibler Divergence (KL散度)
- Jensen-Shannon Divergence (JS散度)

我们先来看KL散度。
$$KL(P|Q)=-\sum_{i=0}^n P(x_i)\log{\frac{Q(x_i)}{P(x_i)}}$$

当事件$x_i$在分布$P$中的概率很大，但在$Q$中的概率很小时，KL散度会很大。反之，KL散度也会很大，但是符号相反。同时，我们可以发现KL散度对于$P$和$Q$不是对称的。

JS散度利用KL散度来计算一个对称的归一化散度分数。
$$JS(P|Q)=JS(Q|P)=\frac{1}{2}KL(P|M)+\frac{1}{2}KL(Q|M)$$

其中
$$M=\frac{1}{2}(P+Q)$$

### Cross-Entropy and KL Divergence[[3]](#ref-3)
我们将KL散度进行分解：
$$\begin{aligned}KL(P|Q)=&\sum_{i=0}^n P(x_i)\log{P(x_i)}-P(x_i)\log{Q(x_i)}\\=&\sum_{i=0}^n P(x_i)\log{P(x_i)}+H(P,Q)\\=&-H(P)+H(P,Q)\end{aligned}$$

即
$$H(P,Q)=KL(P|Q)+H(P)$$

其中$H(P,Q)$是分布$P$和$Q$的交叉熵，$H(P)$是分布$P$的熵。如果$H(P)$是一个常数，优化交叉熵$H(P,Q)$就等同于优化$KL(P|Q)$。

对于机器学习任务，给定一个数据集$D$，我们希望模型$Q_\theta(x)$能够尽可能拟合一个任务的真实分布$P(x)$。这个数据集$D$可以在统计意义上一定程度地近似真实分布，但是在模型的训练过程中，我们会用一个`minibatch`的数据得到一个局部分布$P'$，其与真实分布和数据集分布是有一定的差异的。

由于优化KL散度需要假定真实分布$P(x)$的熵$H(P)$是一个常数，在一个`minibatch`中，我们并不能利用小规模的数据去推测一个稳定的熵值。因此，在这种情况下，使用交叉熵对模型进行优化比KL散度要更加鲁棒。


### Reference
- <span id="ref-1">[1]</span> [A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
- <span id="ref-2">[2]</span> [How to Calculate the KL Divergence for Machine Learning](https://machinelearningmastery.com/divergence-between-probability-distributions/)
- <span id="ref-3">[3]</span> [What is the difference Cross-entropy and KL Divergence?](https://stats.stackexchange.com/questions/357963/what-is-the-difference-cross-entropy-and-kl-divergence)
- <span id="ref-4">[4]</span> Hugo Touvron et al. Training data-efficient image transformers & distillation through attention. ICML 2021.
- <span id="ref-5">[5]</span> Mingkai Zheng et al. ReSSL: Relational Self-Supervised Learning with Weak Augmentation. NeurIPS 2021.
- <span id="ref-6">[6]</span> Page 73, Deep Learning, 2016.
- <span id="ref-7">[7]</span> Page 57, Machine Learning: A Probabilistic Perspective, 2012.