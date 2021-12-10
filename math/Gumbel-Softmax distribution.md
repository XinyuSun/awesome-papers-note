[origin page](https://towardsdatascience.com/what-is-gumbel-softmax-7f6d9cdcb90e)

## Background
在深度学习中，我们经常需要采样离散样本，比如：
- 通过GAN生成文本
- 基于离散隐空间的变分自编码器
- 具有离散动作空间的深度强化学习

然而，从类别分布中采样离散数据的过程是不可微的，这使得反向传播操作在这里罢工了。因此，有学者提出了`Gumbel-Softmax`分布来近似类别分布中的样本，这种分布是连续的，从而能够在反向传播过程中工作

## Methods
假设$Z$是服从类别分布$P(\pi_1,\pi_2,...,\pi_x)$的类别变量，$\pi_i$代表神经网络学习到的属于类别$i$的概率。通常我们采样得到$Z$的方法是取最大概率类别得到一个`one-hot`向量：
$$Z=\rm{onehot}(\max{\{i|\pi_1,...,\pi_i\}})$$

注意由于有$\max$函数的存在，采样的过程是不可导的。

#### 1. Gumbel-Max trick
