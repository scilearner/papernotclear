Title: JEM: Your Classifier is secretely an Energy Based Model
Date: 20220910
Category: paper
tags: paper, generator, EBM

[toc]

# 概述

已知 多类别分类任务概率写作

$$
p(y|x) = \frac{exp(f_{\theta}(x)[y])}{\sum_{y'}exp(f_{\theta}(x)[y'])}
$$
其中 $f_{\theta}(x)[y]$ 为 logits输出

作者假设

$$
p(x, y) = \frac{exp(f_{\theta}(x)[y])}{Z(\theta)} \\
p(x) = \sum_{y} p(x, y) = \frac{\sum_{y} exp(f_{\theta}(x)[y])}{Z(\theta)} \\
$$

可得到一个能量函数 $E_\theta(x) = -\log \sum_{y} exp(f_{\theta}(x)[y])$ 及对应的能量模型

因此， 优化目标函数为

$$
\log p_\theta(x, y) = \log p_\theta(x) + \log p_\theta(y|x) 
$$

