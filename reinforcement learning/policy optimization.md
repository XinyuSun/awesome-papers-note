- 最简单的公式来描述策略效果的梯度 (gradient of policy performance) 
- 如何去掉公式中没用的项
- 如何向公式中添加有用的项

### 关键概念
#### 状态和观测值 (States and Observations)
- 状态$s_t$描述在当前时刻环境的状态，我们并不知道隐藏在状态中的环境信息
- 观测值$o_t$是对状态 __部分__ 的描述

#### 动作空间 (Action Spaces)
一系列可能的动作
- 离散的动作空间：如`alpho go`
- 连续的动作空间：如`agent`控制的现实物理环境中的机器人
  
#### 策略 (Policies)
`agent`决定采取何种动作的依据。策略可以是确定的 (deterministic) ，这时候可以表示为
$$a_t=\mu(s_t)$$
或者是随机的 (stochastic) ，这时通常表示为
$$a_t\sim\pi(\cdot|s_t)$$
在`Reinforcement Learning`中，我们通常考虑参数化的策略 (parameterized policies) ，比如带参数的神经网络，通过优化算法可以改变`agent`的行为

通常将参数表示为$\theta$，并写在策略的表达式旁边:
$$\begin{aligned}&a_t=\mu_\theta(s_t)\\&a_t\sim\pi_\theta(\cdot|s_t)\end{aligned}$$

##### 确定性策略 (Deterministic Policies)
```python
# construct a mlp layer that take in obs_dim-tensor
# observation as input and output act_dim-tensor action
pi_net = MLP(obs_dim, act_dim)

obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
actions = pi_net(obs_tensor)
```
##### 随机策略 (Stochastic Policies)
- 分类策略 (Categorical Policies)
  - 类似于分类器，用于离散的动作空间
  - 多层网络$\to$fc层$\to$softmax层得到概率值
- 对角高斯策略 (Diagonal Guassian Policies)
  - 用于连续的动作空间

- 使用和训练随机策略的关键计算步骤
  - 从策略采样得到动作
  - 计算特定动作的对数似然值$\log_{\pi_\theta}(a|s)$

![](https://img.shields.io/badge/!-Categorical%20Policies-blue)
> __采样 (Sampling).__ 利用每个动作的概率，可以通过pytorch的内建工具进行采样。比如 [Categorical distributions in PyTorch](https://pytorch.org/docs/stable/distributions.html#categorical), [torch.multinomial](https://pytorch.org/docs/stable/torch.html#torch.multinomial)
> 
> __对数似然 (Log-Likelihood).__ 由于最后一层的概率$P_\theta(s)$代表各种动作，我们可以将每个动作视为这个概率（向量）的下标。
> $$\log\pi_\theta(a|s)=\log[P_\theta(s)]_a$$

![](https://img.shields.io/badge/!-Diagonal%20Gaussian%20Policies-blue)
> 多元高斯分布 (multivariate normal distribution) 可以通过均值向量,$\mu$,和协方差矩阵,$\Sigma$,来表示。对角高斯分布是其中的一种特殊情况，即协方差矩阵是一个对角矩阵（各个变量之间相互独立）。因此我们可以通过一个向量来表示方差。
> 对角高斯策略通常通过一个神经网络将观测值映射为均值动作$\mu_\theta(s)$，此外，有两种方法可以表示协方差矩阵。
> 
> __第一种方法:__ 对数标准差向量$\log\sigma$是一个独立的参数，其不是状态的函数。（如TRPO、PPO等方法采用这种方式）
> 
> __第一种方法:__ 通过一个神经网络将状态映射为对数标准差，$\log\sigma_\theta(s)$，其可能会和均值网络共享一些网络层。
> 
> 注意我们一般采用标准差的对数而不是直接得到标准差，这是因为标准差是非负的，而使用对数可以预测一个$(-\infty,+\infty)$的值，在不加约束的情况下可以更容易地训练这个参数。
> 
> __采样:__ 先通过一个标准正态分布$z\sim\mathcal{N}(0,1)$采样得到的噪声$z$，乘标准差加均值之后得到采样的结果：
> $$a=\mu_\theta(s)+\sigma_\theta(s)\odot z$$
> 其中$\odot$表示元素间的点乘。通过pytorch内建的方法可以采样得到随机高斯噪声：[torch.normal](https://pytorch.org/docs/stable/torch.html#torch.normal)，或者通过[torch.distributions.Normal](https://pytorch.org/docs/stable/distributions.html#normal)来构造一个分布类，这样可以在后续方便计算对数似然。
> 
> __对数似然:__ 对于一个$k$维度的动作$a$，其服从均值为$\mu=\mu_\theta(s)$和标准差$\sigma=\sigma_\theta(s)$的正态分布，通过下式计算对数似然：
> $$\log_{\pi_\theta}(a|s)=-\frac{1}{2}(\sum_{i=1}^k(\frac{(a_i-\mu_i)^2}{\sigma_i^2}+2\log\sigma_i)+k\log 2\pi)$$

#### 轨迹 (Trajectories)
世界中一系列的状态和动作
$$\tau=(s_0,a_0,s_1,a_1,...).$$
![](https://img.shields.io/badge/!-Tips-blue)
> `Trajectories` 通常也可称为 `episodes` 或 `rollouts`.

最早的状态$s_0$是从初始状态分布 (start-state distribution) 中随机采样得到，其可以表示为$\rho_0$：
$$s_0\sim\rho_0(\cdot)$$
状态转移由环境中的自然规则来确定，可以是确定的：
$$s_{t+1}=f(s_t,a_t)$$
也可以是随机的：
$$s_{t+1}\sim P(\cdot,a_t)$$

#### 奖励和收益 (Reward and Return)
奖励依赖于当期状态、当期时刻采取的动作、下一个状态来确定：
$$r_t=R(s_t,a_t,s_{t+1})$$
不过一般情况下，奖励只与当前时刻的状态或者状态和动作有关，即$r_t=R(s_t)$或$r_t=R(s_t,a_t)$

`agent`的目标在于 __在某个trajectory上最大化累积的奖励__。我们通过收益$R(\tau)$来表示可能的奖励概念。

其中一种收益是 __finite-horizon undiscounted return, 有限长度的未折扣回报__，其是对固定窗口步长得到的奖励的简单求和：
$$R(\tau)=\sum^T_{t=0}r_t$$

另外一种是 __infinite-horizon discounted return, 无限长度的折扣回报__，其是对`agent`每一步的所有奖励进行加权求和，这个权重称为`折扣系数 (discount factor)` $\gamma\in(0,1)$，其表示这项奖励随时间变化的重要程度：
$$R(\tau)=\sum^\infty_{t=0}\gamma^t r_t$$

加上折扣系数的原因是因为：
- 当前的收益是首要考虑的
- 数学上是为了防止 reward 不收敛

![](https://img.shields.io/badge/!-Tips-blue)
> 实际上在RL的应用中，通常会优化无折扣的收益函数，同时采用带折扣的收益来优化 __值函数 (value funtions)__

#### RL 问题
RL的目标在于选择一个合适的策略，当`agent`按照这个策略行动时，能够获得最大化的期望回报。

假设我们有随机转移的环境和随机策略，对于$T$个step的trajectory，我们有如下的概率分布：
$$P(\tau|\pi)=\rho_o(s_0)\Pi$$

### 最简单的 Policy Gradient 公式推导
考虑一个随机参数策略$\pi_\theta$，我们希望最大化预期的收益函数$J(\pi_\theta)=\mathop{E}\limits_{\tau\sim\pi_\theta}[R(\tau)]$
