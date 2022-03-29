# 机器学习/深度学习常用数学工具

先做个简单记号， 也许能慢慢总结出点玩意


# 泰勒展开

## 一阶泰勒展开

简单， 且非常之常用


## 二阶泰勒展开

VAT, 


# Dual Norm 对偶范数

$$
\|z\|_{p} = \sup \left\{\sum_{i=1}^{n} z_{i} x_{i}: x=\left(x_{1}, \ldots, x_{n}\right) \in \mathbb{R}^{n},\|x\|_{q} \leq 1\right\} \\\|z\|^p_{p}= \|x\|^q_q
$$

$$
\frac{1}{p} + \frac{1}{q} = 1 => p−p/q=p(1−1/q)=p(1/p)=1
$$

					

[证明过程 英语](https://math.stackexchange.com/questions/265721/proving-that-the-dual-of-the-mathcall-p-norm-is-the-mathcall-q-norm)


## 1 使用

### 1.1 对抗样本 adversarial example/FGSM

$$
L_\theta (x) = \max_{\| r \|_p \le \epsilon} L_\theta( x + r)
$$

### 1.2 Sharpness-Aware Minimization 

$$
L (\theta) = \max_{\| \epsilon \|_p \le \rho } L( \theta + \epsilon)
$$




