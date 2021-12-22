## 关于矩阵求导的布局方式

考虑向量$\mathbf{x}\in R^{n}$和向量$\mathbf{y}\in R^m$，求偏导$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$
- 若采用分母布局，则$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\in R^{m\times n}$
- 若采用分子布局，则$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\in R^{n\times m}$
- 若是标量对矩阵求导，则可以采用混合布局，$\frac{\partial x}{\partial\mathbf{y}}$按照$\mathbf{y}$布局，$\frac{\partial \mathbf{x}}{\partial y}$按照$\mathbf{x}$布局

即列数与所采用的布局列数相同
参考[wikipedia](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions) 或[文件](../resources/matrix_vector_derivatives_for_machine_learning.pdf)

### 尝试推导`BN`的导数
`BN`算法的伪代码：
<div align="center">
<img src=../resources/008.png width=70% />
</div>

变量的依赖图：
<div align="center">
<img src=../resources/009.png width=50% />
</div>

我们需要求解参数$\gamma$和$\beta$的导数
- 首先，$\frac{\partial L}{\partial \gamma}=\frac{\partial L}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial \gamma}$，如果采用混合布局，则$\frac{\partial L}{\partial \gamma}=\hat{\mathbf{x}}^T \frac{\partial L}{\partial \mathbf{y}}\in R=\sum_i\frac{\partial L}{\partial y_i}\hat{x}_i$
- 同样的方法可以计算$\frac{\partial L}{\partial \beta}$

## 对DirectCLR中证明部分的理解

### 对`Lemma1`的理解

假设拥有一个单层的线性模型，对于两个增强输入$x_i$和$x_i'$，分别得到输出$y_i=Wx_i$和$y_i'=Wx_i'$

损失函数采用`InfoNCE loss`，公式为
$$L=-\sum_{i=1}^N{\log{\frac{\exp(z_i^Tz_i'/\tau)}{\sum_{j\ne i}\exp{(z_i^Tz_j/\tau)}+\exp(z_i^Tz_i'/\tau)}}}$$

采用随机梯度下降作为优化算法时，权重的更新方向为
$$\dot{W}=-G$$

其中$G$为梯度$\frac{dL}{dW}$，由于输入输出都是向量，线性模型的权重$W\in R^{n\times n}$，从$z_i$到$W$的偏导属于向量对矩阵的求导。

$$x(t)=(\frac{\sin20\pi t}{\pi t})^2\ \ \ x(j\omega)=(\frac{2\sin20\pi t}{t})^2$$