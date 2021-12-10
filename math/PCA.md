## Principal Component Analysis (PCA)主成分分析
数据降维算法

### 主要思想
先考虑二维平面下的特殊情况。有一系列的数据按照一定的规律分布在$xoy$平面上。
<div align="center">
<img src=../resources/004.png width=50% />
</div>

如果能够从这些数据中找到一个合适的新坐标系$x'o'y'$，同时尽量使得数据点分布在新坐标系的某个坐标轴上，就可以用这个坐标轴来表征原来的数据，从而实现数据的降维。
此时，原来的二维数据点$D=\left[\begin{matrix}x_1&x_2&x_3&x_4\\y_1&y_2&y_3&y_4\end{matrix}\right]$可以用新的一维标量表示$D'=[x_1'\ x_2'\ x_3'\ x_4']$

PCA旨找到一个坐标系$x'o'y'$，使得数据损失过程的信息损失是最小的。因此，要找到一个数据分布最分散的方向，使沿这个方向能够保留最多的信息。我们可以通过最大化数据在这个方向上分布的方差来找到这个方向，作为 __主成分__。

### 步骤
- 去中心化
- 找到方差最大的方向

考虑一个白数据`White Data`，其服从相互独立的二维标准正态分布
<div align="center">
<img src=../resources/005.png width=50% />
</div>
假设在一般情况下，我们已有的数据服从$xy$方向上不同且相互独立的正态分布
<div align="center">
<img src=../resources/006.png width=50% />
</div>
白数据$D$可以通过沿某个方向拉伸、旋转得到我们已有的数据$D'$，即
$$D'=RSD]\tag{1}$$
主成分分析的过程其实也是将我们现有的数据$D'$反变换为白数据$D$的过程。只要能得到找到旋转的角度$R$，就可以找到分布方差最大的坐标系。

反变换的过程非常简单，即
$$D=S^{-1}R^{-1}D'\tag{2}$$

### 协方差矩阵
所以PCA的问题又进一步简化为找到这个旋转矩阵$R$。通过这个矩阵的旋转，我们可以得到一个分布方差最大的方向。对于数据呈二维分布的情况，

协方差可以用于衡量数据在不同维度上的相关程度。如果数据在两个维度$x,y$上是同时往相同的方向变化的，那么协方差$cov(x,y)>0$，如果是反方向变化，则$ccov(x,y)<0$。
$$cov(x,y)=\frac{\sum^n_{i=1}x_iy_i}{n-1}\tag{3}$$

按照二维平面的两个坐标排列协方差，可以得到协方差矩阵
$$C=\left[\begin{matrix}cov(x,x)&cov(x,y)\\cov(x,y)&cov(y,y)\end{matrix}\right]\tag{4}$$

在对角线方向上，是数据在某个坐标轴上的方差。在逆对角线方向上，则是数据在两个坐标轴上的协方差，显然方差越大，协方差越小的坐标系越符合我们的要求，此时协方差矩阵接近于对角矩阵。

联立公式(3)和(4)，我们可以得到
$$\begin{aligned}C=&\left[\begin{matrix}\frac{\sum^n_{i=1}x_i^2}{n-1}&\frac{\sum^n_{i=1}x_iy_i}{n-1}\\\frac{\sum^n_{i=1}x_iy_i}{n-1}&\frac{\sum^n_{i=1}y_i^2}{n-1}\end{matrix}\right]\\=&\frac{1}{n-1}\left[\begin{matrix}x_1&x_2&x_3&x_4\\y_1&y_2&y_3&y_4\end{matrix}\right]\left[\begin{matrix}x_1&y_1\\x_2&y_2\\x_3&y_3\\x_4&y_4\end{matrix}\right]\\=&\frac{1}{n-1}DD^T\end{aligned}\tag{5}$$

对于我们现有的数据$D'$，则有
$$\begin{aligned}C'=&\frac{1}{n-1}D'D'^T=\frac{1}{n-1}RSD(RSD)^T\\=&\frac{1}{n-1}RSDD^TS^TR^T=RS(\frac{1}{n-1}DD^T)S^TR^T\\=&RSCS^TR^T\end{aligned}\tag{6}$$

由于白数据的协方差矩阵$C=\left[\begin{matrix}1&0\\0&1\end{matrix}\right]$，代入式(6)可以得到
$$C'=RSS^TR^T\tag{7}$$

由于拉升变换矩阵是对角矩阵，即$S^T=S$，令$L=SS^T=\left[\begin{matrix}a^2&0\\0&b^2\end{matrix}\right]$，有
$$C'=RLR^T\tag{8}$$

到了这一步，我们可以欣喜的发现已经离我们需要求解的R非常接近了。再利用一点线性代数的知识我们就可以求得$R$。我们观察式(8)，给他右乘R，由于旋转矩阵$R$是正定矩阵，其转置等于自身的逆，可以得到
$$C'R=RLR^TR=RL\tag{9}$$

显然，$R$是协方差矩阵$C'$的特征向量，$L$是$C'$的特征值。

### 总结
<div align="center">
<img src=../resources/007.png width=50% />
</div>

### 缺点
`PCA`受离群点的影响较大。