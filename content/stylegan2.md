Title: StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN
Date: 20220919
Category: paper
tags: paper, generator, GAN

[TOC]

# 概述
[[stylegan]]

1. 改进 AdaIN 造成的气泡点
2. 

## 归一化
原版:  第i层AdaIN $AdaIN(x, y_i) = y_{s,i} \frac{x - \mu(x)}{\sigma(x)} + y_{b, i}$ ,  其中  $y_i = (y_{s, i}, y_{b, i}) = A_i (w),  w = 8MLP(z)$


缺点： 生成 blob有气泡部位 或 corrupted异常 的图片

改进一 移除 AdaIN里的平移项bias $y_b$， 即只保留 scaling项 $y_s$  作用类似 std 
改进二  Demodulation

### Demodulation

移除 normalization操作， 转成 w -> w' = s w -> w'' = w'/ $\sqrt{\sum w'}$


![2_norm](images/stylegan2_normalization.png)

## 正则化

PPL(perceptual path length)跟感知图像质量的关系



   
