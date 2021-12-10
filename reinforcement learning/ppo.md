Contributions in origin [PPO](https://arxiv.org/abs/1707.06347) paper:
> We have introduced [PPO], a family of policy optimization methods that use __multiple epochs of stochastic gradient ascent to perform each policy update.__ These methods have the stability and reliability of trust-region [[TRPO]](https://arxiv.org/abs/1502.05477) methods but are much simpler to implement, requiring __only a few lines of code change to a vanilla policy gradient implementation__, applicable in more general settings (for example, when using a joint architecture for the policy and value function), and have better overall performance.

### The Clipped Surrogate Objective
是Policy gradient的替代品，能够通过限制在每一步对policy的更改来提高训练时的稳定性。

- Vanilla Policy Gradient (refers to [openai wiki](https://spinningup.openai.com/en/latest/algorithms/vpg.html#key-equations))
$$L^{PG}(\theta)=\hat{\mathbb{E}}_t[\log\pi_{\theta}(a_t|s_t)\hat{A}_t]$$
$$\nabla_\theta J(\pi_\theta)=\mathop{E}\limits_{\tau\sim\pi_\theta}[\sum^T_{t=0}\nabla_\theta\log\pi_\theta(a_t|s_t)A^{\pi_\theta}(s_t,a_t)]$$

其中$\pi_\theta$表示参数为$\theta$的policy network，$J(\pi_\theta)$表示