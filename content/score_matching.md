Title: Score Matching: Estimation of non-normalized statistical models by score matching
Date: 20220919
Category: paper
tags: paper, generator, score matching

[TOC]

# 概述

$p_{\theta}(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)}$, $Z(\theta) = \int_x e^{-E_\theta(x)}$配分函数 是对所有 x 的积分, 不可计算， 但是对x 来说是个常数。

所以 最大对数似然估计 MLE 对 $\theta$求导更新 需要 MCMC采样， 很慢

但 $p_{\theta}(x)$