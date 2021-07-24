# 论文常见符号的主要含义



| 数学符号                                                     | 一般含义                                                     | 其他说明                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| N, K, k, n                                                   | 一般指代数目、数量                                           | 不排除奇葩用法                                               |
| epoch, iteration                                             | epoch->遍历（随机或不随机）一遍数据集<br> iteration -> 训练一个批次batch | 代码里也可能把若干次iteration算一个epoch                     |
| $D=(\textbf{X}, Y)=(\mathcal{X, Y})= \{\textbf{x}_i, y_i\}$  | D是整个数据集，大写代表数据集，粗体代表向量、矩阵、张量(后面内容我都不用粗体) | 我不知道中间字体的学名叫什么 tex里写作 \matchcal             |
| label, unlabel, $D_l$, $D_{ul}$                              | 有标签、无标签数据集， 应该主要是半监督学习才区分            |                                                              |
| $p(x), p_D$                                                  | 一般是指某分布， x的分布， D(data)数据的“真实”分布           |                                                              |
| $P(y|x), p(y=\text{"car"}|x)$                                | 大小写一般同一个意思，读作给定x，输出y的概率。 多数论文用softmax， 所以一般代表经过softmax输出的概率向量(cifar10就是10维的向量，代表每个类别的) | 不要问我它是行向量还是列向量                                 |
| $w, \theta$                                                  | 都常用作模型参数， weight->w                                 |                                                              |
| $h, z$, logits                                               | h(hidden) 某隐藏层输出，但也常作为 提取的特征(feature->hidden representation)<br>一般分类器 $x\rightarrow f_\theta(x) \rightarrow h $<br>$h \rightarrow 最后一层W \rightarrow z \text(logits) \rightarrow \text{softmax} \rightarrow p(y|x)$ | Contrastive learning(SimCLR/BYOL)里就不是这个特定含义        |
| $f_\theta(x)$                                                | x是输入， f就是带参数$\theta$的模型（比如神经网络）有些输出是概率向量， 有些只代表是特征提取部分$h=f_\theta(x)$， 代表某部分模型， 都可以 |                                                              |
| $g(z), g(x)$                                                 | 生成模型比如GAN里就是生成器(generator)， 如果$f$ 已经用了， 也常用作普通的神经网络（比如SimCLR它们的projector就用$g$) |                                                              |
| $z, p_\theta(x | z)$                                         | 生成模型里， 这个 $z$ 一般就是某个采样的隐藏变量 hidden variable， 生成 $g(z)$，  用 $p_\theta(x|z)$ 算概率分布 |                                                              |
| $q_\theta()$                                                 | 一般变分用的新概率分布， 换一个分布代替原本intractable的$p()$ |                                                              |
| $\| x \|$,$\| x \|_p$                                        | 范数， 默认 Frobenius范数（一般API是这个， 理解时就当成是向量的$L_2$范数， 矩阵范数相对不好理解、一般也都是当向量用）， $p$ -> $L_p$范数<br>p=2 -> 模长、欧氏距离， 平方和开方<br/>p=$\infty$ 元素绝对值中的最大值<br/>p=1, 元素绝对值之和<br>p=0, 非零元素个数 |                                                              |
| $D_{KL}(p\|q), D_{JS}$                                       | 这里是分布间的距离， KL散度距离， JS散度距离， 欧氏距离很少见写成这种符号形式 |                                                              |
| $z\sim U(), \mathcal{N}(), Beta(), Bernouli()$               | 均匀分布、高斯分布及其他分布， 采样                          |                                                              |
| $\mathbb{E}[ f(x)],\mathbb{E}_q, \mathbb{E}_{t\sim q(x)}$    | 期望，  某个分布(q)的 f函数期望                              | 也有写成实体 $E$ 的，总不能是tex模板里不准用 \mathbb 字体吧？ |
| $\mu$ 均值,  $\text{sigma}: \sigma$ 方差 , $\text{Sigma}: \Sigma$ 协方差矩阵 | $\Sigma \sim \sum$   Sigma和求和  没法从外形区分             |                                                              |
| $\langle w , v \rangle, cos(w,v) = \frac{\langle w , v \rangle}{\|w\| \|v\|}$ | 向量内积 和 余弦相似度                                       |                                                              |
| MLE 最大似然估计 maximize likelihood estimation              |                                                              |                                                              |

