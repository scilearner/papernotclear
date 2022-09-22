Title: StyleGAN2-ADA: Training Generative Adversarial Networks with Limited Data
Date: 20220922
Category: paper
tags: paper, generator, GAN

[TOC]

# 概述
[[stylegan2]]

1.  RandAugment has 18 data augmentation -> 6 categories: 
	1. They can be unleaking if it is only executed at a probability $p<100\%$.
	2. Explanation.
	3. pixel blitting (x-flips, 90◦ rotations, integer translation), more general geometric transformations, color transforms. They can improve.  Additive noise, cutout can't.
2. Adaptive Discriminant Augmentation for the probability $p$.
	1. Add augmentation to both of Discriminator and Generator.
	2. $r_t = \mathbb{E}[sign (D_{train})]$ indicates the overfitting
3.  Init $p=0$, adjust $p$ every four batches, based on $r_t$.