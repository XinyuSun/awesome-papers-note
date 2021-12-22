[toc]
## 刚体运动
- 基础部分请参考[My BLOG](https://bigducktwist.club/?p=175)
- 补充知识：向量的内积和外积
  - 对于$a,b\in\mathbb{R}^3$
  - 内积
$$\boldsymbol{a}\cdot \boldsymbol{b}=\boldsymbol{a}^T \boldsymbol{b}=\sum^3_{i=1}a_ib_i=|\boldsymbol{a}||\boldsymbol{b}|\cos\langle\boldsymbol{a},\boldsymbol{b}\rangle$$
  - <span id="outer_product">外积：</span> 

$$a\times b=\left\Vert\begin{matrix}r_1&r_2&r_3\\a_1&a_2&a_3\\b_1&b_2&b_3\end{matrix}\right\Vert=\left[\begin{matrix}a_2b_3-a_3b_2\\a_3b_1-a_1b_3\\a_1b_2-a_2b_1\end{matrix}\right]=\left[\begin{matrix}0&-a_3&a_2\\a_3&0&-a_1\\-a_2&a_1&0\end{matrix}\right]\boldsymbol{b}\overset{\text{def}}{=}(\boldsymbol{a}^{\land})\boldsymbol{b}$$

- 这里的$\boldsymbol{a}^{\land}$表示将向量$\boldsymbol{a}$写为反对称矩阵的形式
- 外积不满足交换律，而满足反交换律
  $$a\times b=-b\times a$$
- 共线的向量，外积为0
  $$n\times n=n^\land n=0$$
- 外积满足分配率
  $$a\times (b+c)=a\times b+a\times c$$

### 旋转矩阵的性质
- 行列式为1的正交矩阵

定义$n$维旋转矩阵的集合
$$\text{SO}(n)=\{\boldsymbol{R}\in\mathbb{R}^{n\times n}|\boldsymbol{RR}^T=\boldsymbol{I},\rm{det}(\boldsymbol{R})=1\}.$$

- $\text{SO}(n)$表示**特殊正交群(Special Orthogonal Group)**
- $\text{SO}(3)$即表示3维旋转矩阵

定义$n+1$维变换矩阵
$$\text{SE}(3)=\{\boldsymbol{T}=\left[\begin{matrix}\boldsymbol{R}&\boldsymbol{t}\\\boldsymbol{0}^T&1\end{matrix}\right]\in\mathbb{R}^{4\times 4}|\boldsymbol{R}\in\text{SO}(3),\boldsymbol{t}\in\mathbb{R}^3\}$$

- $\text{SE}(3)$表示**特殊欧式群(Special Euclidean Group)**

### 旋转向量和欧拉角
变换矩阵表达旋转的方式有以下两个缺点
- 变换矩阵用16个变量来表示6自由度的变换，这种表达方式是冗余的
- 矩阵本身有约束：行列式为1的正交矩阵，这在估计或优化一个变换矩阵时会使求解变得困难

#### 旋转向量
- 使用一个向量表示绕一个旋转轴的旋转，其方向表示旋转轴，长度表示旋转角
- 称为**旋转向量**或**轴角(Axis-Angle)**
- 拆分为一个单位长度向量$n$和角度$\theta$

**罗德里格斯公式 (Rodrigues Formula)**
$$\boldsymbol{R}=\cos\theta\boldsymbol{I}+(1-\cos\theta)\boldsymbol{nn}^T+\sin\theta \boldsymbol{n}^{\land}$$

- $\boldsymbol{n}^{\land}$表示向量$\boldsymbol{n}$的反对称矩阵$\boldsymbol{n}^{\land}=\left[\begin{matrix}0&-a_3&a_2\\a_3&0&-a_1\\-a_2&a_1&0\end{matrix}\right]$

**从旋转矩阵到旋转向量**
$$\begin{aligned}\text{tr}(\boldsymbol{R})=&\cos\theta\text{tr}(\boldsymbol{I})+(1-\cos\theta)\text{tr}(\boldsymbol{nn}^T)+\sin\theta\text{tr}(\boldsymbol{n}^{\land})\\=&3\cos\theta+(1-\cos\theta)\\=&1+2\cos\theta\end{aligned}$$
即 <span id="quat1"></span>
$$\theta=\arccos\frac{\text{tr}(\boldsymbol{R})-1}{2}$$
对于转轴$n$，旋转轴上的向量经过旋转不发生变化，可以得到
$$\boldsymbol{Rn}=\boldsymbol{n}$$
显然，$\boldsymbol{n}$是旋转矩阵$\boldsymbol{R}$特征值1对应的特征向量，解这个特征方程，再归一化就得到$\boldsymbol{n}$

#### 欧拉角
欧拉角需要考虑定坐标系旋转和动坐标系旋转，具体参考[My BLOG](https://bigducktwist.club/?p=175/#eular-angles)
- 万向锁问题：当俯仰角为$\pm90\degree$时，第一次旋转和第三次旋转使用相同的转轴，**使得旋转过程丢失了一个自由度**
  - 理论上可以证明，使用三个实数表示旋转时都会遇到这种奇异性证明
  - 一般只用于人机交互

### 四元数 (Quaternion)
> 事实上，我们找不到不带奇异性的三维向量来表示三维旋转。因此我们只能折中地选取另一种方式，在牺牲直观性的同时，保证紧凑和非奇异地表示三维旋转

#### 复数与旋转
欧拉公式
$$e^{i\theta}=\cos\theta+i\sin\theta$$
表示一个单位复数。
- 在二维情况下，旋转可以用单位复数来表示，乘以$i$表示旋转90°
  - $$ie^{i\theta}=i\cos\theta-\sin\theta=e^{i(\theta+\pi)}$$
- 推广到三维情况，我们也可以用单位四元数来描述旋转

#### 四元数
$$\boldsymbol{q}=q_0+q_1 i+q_2 j+q_3 k$$
- $i,j,k$表示四元数的三个虚部
- 三个虚部满足：
  $$\left\{\begin{aligned}&i^2=j^2=k^2=-1\\&jk=i,kj=-i\\&ki=j,ik=-j\\&ij=k,ji=-k\end{aligned}\right.$$
  - 即虚部自己的乘法与复数一样($i^2=-1$)，虚部之间的乘法与[外积](#outer_product)一样
- $q_0$为实部，$q_1i+q_2j+q_3k$为虚部，可以写为$\boldsymbol{q}=[s,v]^T$，使$s=q_0\in\mathbb{R},\ v=[q_1,q_2,q_3]^T\in\mathbb{R}^3$
- 与复数不同的是，**四元数的虚部不能直接表示具体的旋转**

**四元数的运算**
常见的运算：四则运算、共轭、求逆、数乘

- 加法减法
  $$\boldsymbol{q}_a\pm \boldsymbol{q}_b=[s_a+s_b,v_a+v_b]^T$$
- 乘法
  $$\boldsymbol{q}_a\boldsymbol{q}_b=[s_as_b-v_a^Tv_b,s_av_b+s_bv_a+v_a\times v_b]^T$$
- 模长
  $$\Vert\boldsymbol{q}_a\Vert=\sqrt{s_a^2+x_a^2+y_b^2+z_c^2}=\sqrt{s_a^2+v_a^Tv_a}$$
- 共轭
  $$\boldsymbol{q}^*_a=[s_a,-v_a]^T$$
  
  $$\boldsymbol{q}^*\boldsymbol{q}=\boldsymbol{qq}^*=[\Vert \boldsymbol{q}\Vert^2,0]$$
  - 共轭和自身相乘，得到实部为模长平方的实四元数
- 逆
  $$\boldsymbol{q}^{-1}=\boldsymbol{q}^*/\Vert \boldsymbol{q}\Vert$$
  
  $$\boldsymbol{qq}^{-1}=\boldsymbol{q}^{-1}\boldsymbol{q}=1$$
  - 可由共轭的性质推出
- 数乘
  $$k\boldsymbol{q}=[ks,kv]^T$$

**用四元数表示旋转**
假设存在三维空间向量$\boldsymbol{p}=[x,y,z]\in\mathbb{R}^3$，通过一个单位四元数$\boldsymbol{q}$来指定这个向量的旋转
- 首先将三维空间点用一个**虚四元数**表示
$$\boldsymbol{p}=[0,x,y,z]=[0,v]^T$$

$$\boldsymbol{p}'=\boldsymbol{qpq}^{-1}$$
- 将计算结果的虚部取出，得到旋转后的向量
- 可以证明，计算结果也是一个虚四元数（见课后习题）

**四元数到其他旋转表示的转换**
我们可以将四元数乘法写成喜闻乐见的矩阵乘法
- 设$\boldsymbol{q}=[s,v]^T$，定义$\boldsymbol{q}^+,\boldsymbol{q}^\oplus$
  $$\boldsymbol{q}^+=\left[\begin{matrix}s&-v^T\\v&s\boldsymbol{I}+v^\land\end{matrix}\right], \boldsymbol{q}^\oplus=\left[\begin{matrix}s&-v^T\\v&s\boldsymbol{I}-v^\land\end{matrix}\right]$$
  
  $$\boldsymbol{q}_1^+\boldsymbol{q}_2=\left[\begin{matrix}s_1&-v_1^T\\v_1&s_1\boldsymbol{I}+v_1^\land\end{matrix}\right]\left[\begin{matrix}s_2\\v_2\end{matrix}\right]= \left[\begin{matrix}-v_1^Tv_2+s_1s_2\\s_1v_2+s_2v_1+v_1^\land v_2\end{matrix}\right]=\boldsymbol{q}_1\boldsymbol{q}_2 $$

- 同理，可证 (已知$v_1^\land v_2=-v_2^\land v_1$)
  $$\boldsymbol{q}_2^\oplus\boldsymbol{q}_2=\left[\begin{matrix}s_2&-v_2^T\\v_2&s_2\boldsymbol{I}+v_2^\land\end{matrix}\right]\left[\begin{matrix}s_1\\v_1\end{matrix}\right]= \left[\begin{matrix}-v_2^Tv_1+s_1s_2\\s_1v_2+s_2v_1-v_2^\land v_1\end{matrix}\right]=\boldsymbol{q}_1\boldsymbol{q}_2 $$

- 考虑前面用四元数表示旋转的公式，有
  $$\boldsymbol{p}'=\boldsymbol{qpq}^{-1}=\boldsymbol{q}^+\boldsymbol{p}^+\boldsymbol{q}^{-1}=\boldsymbol{q}^+(\boldsymbol{q}^{-1})^\oplus\boldsymbol{p}$$

$$\boldsymbol{q}^+(\boldsymbol{q}^{-1})^\oplus=\left[\begin{matrix}s&-v^T\\v&s\boldsymbol{I}+v^\land\end{matrix}\right]\left[\begin{matrix}s&-v^T\\-v&s\boldsymbol{I}+v^\land\end{matrix}\right]=\left[\begin{matrix}1&0\\0^T&vv^T+s^2\boldsymbol{I}+2sv^\land+(v^\land)^2\end{matrix}\right] $$

- 由于$\boldsymbol{p}'$和$\boldsymbol{p}$都是虚四元数，因此该矩阵右下角给出了**四元数到旋转矩阵的变换关系**
  $$\boldsymbol{R}=vv^T+s^2\boldsymbol{I}+2sv^\land+(v^\land)^2$$

- 对上式两侧求迹，求四元数到旋转向量的转换公式
  $$\begin{aligned}\text{tr}(\boldsymbol{R})=&\text{tr}(vv^T)+3s^2+2s\cdot0+\text{tr}((v^\land)^2)\\=&v_1^2+v_2^2+v_3^2+3s^2-2(v_1^2+v_2^2+v_3^2)\\=&4s^2-(s^2+v^Tv)\\=&4s^2-\Vert\boldsymbol{q}\Vert^2\\=&4s^2-1\end{aligned}$$

  - 由[旋转矩阵到旋转向量公式](#quat1)可得
  $$\begin{aligned}\theta=&\arccos\frac{\text{tr}(\boldsymbol{R})-1}{2}\\=&\arccos(2s^2-1)\end{aligned}$$

  - 根据三角函数的倍角公式，有
  $$\begin{aligned}\cos\theta&=2s^2-1=2\cos^2\frac{\theta}{2}-1\\&\to s=\cos\frac{\theta}{2}\\&\to\theta=2\arccos s\end{aligned}$$

  - 将$\boldsymbol{p}$的虚部用$\boldsymbol{q}$的虚部代替，经过$\boldsymbol{q}^+(\boldsymbol{q}^{-1})^\oplus$的旋转有

$$\begin{aligned}\left[\begin{matrix}1&0\\0^T&vv^T+s^2\boldsymbol{I}+2sv^\land+(v^\land)^2\end{matrix}\right]\left[\begin{matrix}0\\v\end{matrix}\right]=&\left[\begin{matrix}0\\(v^Tv+s^2\boldsymbol{I})v+2sv^\land v+v^\land(v^\land v)\end{matrix}\right]\\=&\left[\begin{matrix}0\\v\end{matrix}\right] \end{aligned}$$

- 即$\boldsymbol{q}$的虚部经过$\boldsymbol{q}$的旋转不发生变化，因此$\boldsymbol{q}$的虚部为转轴
  - $\Vert\boldsymbol{q}\Vert=\sqrt{s^2+v^Tv}=1$，故转轴的模长为
  $$\begin{aligned}\sqrt{v^Tv}=&\sqrt{1-s^2}\\=&\sqrt{1-\cos^2\frac{\theta}{2}}=\sin\frac{\theta}{2}\end{aligned}$$

  - 因此，四元数到旋转向量的转换公式为
  $$\left\{\begin{aligned}&\theta=2\arccos q_0\\&[n_x,n_y,n_z]^T=[q_1,q_2,q_3]^T/\sin\frac{\theta}{2}\end{aligned}\right.$$

### 相似，仿射和射影变换
- 相似变换
  - 与欧式变换相比增加了一个缩放的自由度，在三个坐标轴上均匀缩放
  $$\boldsymbol{T}_s=\left[\begin{matrix}s\boldsymbol{R}&t\\0^T&1\end{matrix}\right] $$
- 仿射变换
  - 只要求$\boldsymbol{A}$是一个可逆矩阵，不必是正交矩阵
  - 也称正交投影
  $$\boldsymbol{T}_A=\left[\begin{matrix}\boldsymbol{A}&\boldsymbol{t}\\0^T&1\end{matrix}\right] $$
- 射影变换
  - 可逆矩阵$\boldsymbol{A}$，平移向量$\boldsymbol{t}$，缩放系数向量$\boldsymbol{a}^T$
  $$\boldsymbol{T}_P=\left[\begin{matrix}\boldsymbol{A}&\boldsymbol{t}\\\boldsymbol{a}^T&v\end{matrix}\right] $$

  <div align="center">
  <img src=../resources/030.png width=70% />
  </div>

### 课后习题
[可以参考](https://blog.csdn.net/Night___Raid/article/details/105447274)
- 验证旋转矩阵是正交矩阵
  - 旋转矩阵$R_A^B$表示将A坐标系旋转到B坐标系，可以用B坐标系的基向量在A坐标系下的表示来组成该矩阵
  $$R_A^B=\left[\begin{matrix}\mid&\mid&\mid\\X_B&Y_B&Z_B\\\mid&\mid&\mid\end{matrix}\right] $$
  - 可见，旋转矩阵的列向量之间是两两正交的
<br>

- 推导罗德里格斯公式
<div align="center">
<img src=../resources/034.png width=50% />
</div>

首先，我们分析叉乘的性质。根据叉乘的分配率，有$\boldsymbol{k}\times \boldsymbol{v}=\boldsymbol{k}\times\boldsymbol{v_\perp}+\boldsymbol{k}\times\boldsymbol{v}_\parallel$，因为$\boldsymbol{k}$和$\boldsymbol{v}_\parallel$平行，$\boldsymbol{k}\times\boldsymbol{v}_\parallel=0$，故$\boldsymbol{k}\times\boldsymbol{v}=\boldsymbol{k}\times\boldsymbol{v_\perp}$。因为$\boldsymbol{k}$和$\boldsymbol{v_\perp}$垂直，$\boldsymbol{k}\times\boldsymbol{v}_\perp$相当于将$\boldsymbol{v}_\perp$绕$\boldsymbol{k}$旋转90度，如果再旋转90度，并取反，就得到$\boldsymbol{k}_\perp$本身，因此有

$$\begin{aligned}&\boldsymbol{v}_\parallel=(\boldsymbol{k}\cdot\boldsymbol{v})\boldsymbol{k}\\&\boldsymbol{v}_\perp=-\boldsymbol{k}\times(\boldsymbol{k}\times\boldsymbol{v})\end{aligned}$$

<div align="center">
<img src=../resources/031.png width=80% />
</div>

根据这个旋转示意图，可以发现$\boldsymbol{v}_{rot}$可以拆分为$\boldsymbol{v}$和两个位于旋转平面上的向量之和，根据三角函数关系，有
$$\begin{aligned}\boldsymbol{v}_{rot}=&\boldsymbol{v}+\boldsymbol{k}\times\boldsymbol{v}\sin\theta+\boldsymbol{k}\times(\boldsymbol{k}\times\boldsymbol{v})(1-\cos\theta)\\=&\boldsymbol{v}+\boldsymbol{k}\times\boldsymbol{v}\sin\theta-\boldsymbol{v}_\perp(1-\cos\theta)\\=&\boldsymbol{v}_\parallel+\boldsymbol{v}_\perp\cos\theta+\boldsymbol{k}\times\boldsymbol{v}\sin\theta\\=&\boldsymbol{v}_\parallel(1-\cos\theta)+(\boldsymbol{v}_\parallel+\boldsymbol{v}_\perp)\cos\theta+\boldsymbol{k}\times\boldsymbol{v}\sin\theta\\=&\boldsymbol{v}\cos\theta+(\boldsymbol{v}\cdot\boldsymbol{k})\boldsymbol{k}(1-\cos\theta)+\boldsymbol{k}\times\boldsymbol{v}\sin\theta\end{aligned}$$

<br>

- 验证四元数旋转某个向量后得到的结果仍然是一个虚四元数（实部为0）
  有向量$\boldsymbol{p}=[0,v_1]^T$，四元数$\boldsymbol{q}=[s_2,v_2]$
  $$\begin{aligned}\boldsymbol{qpq}^{-1}=&(\boldsymbol{qp})\frac{\boldsymbol{q}^*}{\Vert\boldsymbol{q}\Vert}=(\boldsymbol{qp})\boldsymbol{q}^*\\=&[-v_1^Tv_2,s_2v_1+v_1\times v_2]^T\cdot[s2,-v2]^T\\=&[-s_2v_1^Tv_2+(s_2v_1+v_1\times v_2)^Tv_2,\cdots]^T\\=&[(v_1\times v_2)^Tv_2,\cdots]\\=&[0,\cdots]\end{aligned}$$

<br>

- 总结旋转矩阵、轴角、欧拉角、四元数的关系
<div align="center">
<img src=../resources/032.png width=100% />
</div>
<br>

- 一般线性方程$\boldsymbol{Ax}=\boldsymbol{b}$有几种解法？
  [参考](https://www.cnblogs.com/newneul/p/8306442.html)
  - 求逆法：仅适合方阵（满秩）的情况
  - QR分解：适合方阵或非方阵，若无解得出近似解
  - 最小二乘法：适合方阵或非方阵，若无解得出最小二乘解
  - 迭代法
<br>