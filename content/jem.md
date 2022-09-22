Title: JEM: Your Classifier is secretely an Energy Based Model
Date: 20220910
Category: paper
tags: paper, generator, EBM

[TOC]

# 概述

已知 多类别分类任务概率写作 Usually, for a multi-class task, the probability $p(y|x)$ is written as

$$
p(y|x) = \frac{exp(f_{\theta}(x)[y])}{\sum_{y'}exp(f_{\theta}(x)[y'])}
$$
其中 $f_{\theta}(x)[y]$ 为 logits输出, where $f_{\theta}(x)[y]$ is the logits output.

作者假设 Following the definition of EBM, define the joint density $p(x, y)$ as:

$$
p(x, y) = \frac{exp(f_{\theta}(x)[y])}{Z(\theta)} 
$$
Hence, it's easily to derive

$$
p(x) = \sum_{y} p(x, y) = \frac{\sum_{y} exp(f_{\theta}(x)[y])}{Z(\theta)}
$$

They derive a new energy function in the multi-class task.
可得到一个能量函数 $E_\theta(x) = -\log \sum_{y} exp(f_{\theta}(x)[y])$ 及对应的能量模型

因此， 优化目标函数为 Combine the energy-based model with the classifier, the objective function is

$$
\log p_\theta(x, y) = \log p_\theta(x) + \log p_\theta(y|x) 
$$

