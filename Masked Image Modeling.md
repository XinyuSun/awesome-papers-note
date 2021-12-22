# Masked Image Modeling(MIM)
## Contents
- iGPT (ICML 2020)
- ViT (ICLR 2021)
- BEiT (Arxiv 2021.06.15)
- MAE (Arxiv 2021.11.11)
- iBOT (Arxiv 2021.11.15)
- SimMIM (Arxiv 2021.11.18)
- BEVT (Arxiv 2021.12.02)

## Zero-Shot Text-to-Image Generation (DALL-E)
Aditya Ramesh, Mikhail Pavlov
### Background
> 传统的 `Text-to-Image` 生成任务关注于在一个固定的数据集上寻找更好的模型假设，这些假设可能包含复杂的模型结构，额外的损失函数，或者在训练的过程中需要一些其他的边缘信息，包括目标的部分标签、分割掩膜。
#### History
- First work: Mansimov et al. (2015)
- Reed et al. (2016b)
  - Using a generative adversarial network rather than a recurrent variational auto-encoder improves image fidelity (保真度).
- Multi scale generator (Zhang et al., 2017; 2018).
- Integrating attention and auxiliary losses (Xu et al., 2018).
- Leveraging additional sources of conditioning information beyound just text  (Reed et al., 2016a; Li et al., 2019; Koh et al., 2021).
- Incorporate pretrained discriminative models.
- Pretrained cross-modal masked language model.

### Motivation
- 通过增加数据量和模型大小，transformer可以在多个领域得到很好的效果
- 数据大小和模型大小可能是现有方法的一大限制
- 通过一个具有120亿参数的自回归transformer在2亿5千万的图像-文本配对数据集上训练，能够得到灵活、高保真的自然语言-图像生成模型

### Methods
提出了一种基于`transformer`的简单方法，将图像和文本的`tokens`自回归地建模为单个数据流
- 采用具有8192个类别的离散向量作为每个图片patch的token
  - 直接使用像素作为图像的token，对高分辨图像需要过高的内存开销
  - 此外，重构像素的目标函数倾向于优先对像素的短程依赖关系进行建模，模型的大部分建模能力被用在捕捉 __高频细节__，比如图像的边缘、轮廓，而不是我们识别这幅图像更需要的物体的低频结构
  - 这种离散的token可以很好地应用在`visual transformer`的自监督预训练中

提出了一种两阶段(2-stage)的训练过程
- 阶段1
  - 训练一个离散的变分自编码器(`d-VAE`)，将$256\times256$大小的RGB图像编码为$32\times32$个grid的tokens，每一个token可能是8192个值中的一个。
    - 这能够大幅度降低上下文的大小(context size)，但不会给图像质量带来很大的下降
- 阶段2
  - 将256个BPE编码的文本tokens和$32\times32=1024$个图像tokens拼接到一起，用于训练一个自回归的transformer来建模文本和图像的联合分布。

优化目标：最大化模型对图像$x$，描述$y$，和tokens$z$之间的分布的联合似然的 __ELB__(Evidence Lower Bound)
联合分布：
$$p_{\theta,\phi}(x,y)=p_\theta(x|y,z)p_\phi(y,z)$$
下限：
$$\ln p_{\theta,\phi}(x,y)\ge\mathop{\mathbb{E}}\limits_{z\sim q_\Phi(z|x)}(\ln p_\theta(x|y,z)-\beta D_{KL}(q_\Phi(y,z|x),p_\phi(y,z)))$$

其中
- $q_\Phi$表示通过`d-VAE`从RGB图像$x^2$生成$32\times32$的图像tokens的分布
- $p_\theta$表示通过`d-VAE`解码器从给定的图像tokens生成RGB图像的分布
- $p_\phi$表示通过`transformer`建模的文本和图像 tokens之间的联合分布

#### Stage one: Learning the Visual Codebook

