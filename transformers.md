# contents
- [vision transformer](#vision-transformer)

- [video transformer](#video-transformer)
  - [Video Swin Transformer (Arxiv 2021-06)](#video-swin-transformer)
  - [TokenLearner (NeurIPS 2021)](#tokenlearner-adaptive-space-time-tokenization-for-videos)

## Vision Transformer


## Video Transformer
### Video Swin Transformer
<div align="center">
<img src=resources/029.png width=100% />
</div>
### TokenLearner: Adaptive Space-Time Tokenization for Videos

#### Background
- 在`ViT`等视觉`transformer`中，输入图像被分为$16\times16$甚至更小的`patches`
- 在`ViViT`和`TimeSformer`中，输入视频同样按照网格被分成多个2维的patches或3维的时-空立方体

#### Motivation
- 传统的视觉`transformer`使用手动划分的大量稠密`patches`来获取`tokens`，但是由于视觉图像中的空间冗余性，会引入大量重复的`tokens`，从而增大`self-attention`的计算量
- 自适应地区选取最重要的`tokens`来计算`attention`能够极大地减小计算开销
- 通过学习的方法来让模型决定选取更合适的图像区域作为`tokens`输入

#### Contribution
- 将每一帧的`tokens`数量减少到$8\sim16$个，(通常采用$200\sim500$个)，可以大幅度减小计算开销，在`FLOPs`减小一半的同时保持分类的准确率
- 视频分类结果取得`k400`，`k600`，`Charades`，`AViD`的`SOTA`

#### Overview
<div align="center">
<img src=resources/022.gif width=100% />
</div>

- 对每一帧计算多个`attention map`，将这个`map`与原输入帧相乘，并进行`spatial pooling`后得到对应的`tokens`

#### Methods


[$\to$back to contents](#contents)