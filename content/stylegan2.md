Title: StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN
Date: 20220919
Category: paper
tags: paper, generator, GAN

[TOC]

# 概述
[[stylegan]]

1. 改进 AdaIN,  移除 normalization, 但保持输入与输出方差不变。 不再生成带气泡点的或异常的图片
2. 添加正则化， 避免 隐空间 w 区域的拉伸或挤压， 任意方向梯度相等
	1. 发现 PPL perceptual path length 与 图片质量 相关
3. 改进结构， 判别器使用 残差， 生成器只使用 skip 跳链， 保持 Progressive Growing
4. 可视化分析对分辨率resolution的利用程度， 发现使用更大模型可以提高 对更高分辨率的利用
5. 正则项使得生成器 $x=g(w)$ 求逆结果更准确更唯一， $x=g(w), w'=g^{-1}(x), x'=g(w'), x\approx x'$ 

## 2 归一化改进
原版:  第i层AdaIN $AdaIN(x, y_i) = y_{s,i} \frac{x - \mu(x)}{\sigma(x)} + y_{b, i}$ ,  其中  $y_i = (y_{s, i}, y_{b, i}) = A_i (w),  w = 8MLP(z)$


缺点： 生成 blob有气泡部位 或 corrupted异常 的图片

改进一 移除 AdaIN里的平移项bias $y_b$， 即只保留 scaling项 $y_s$  作用类似 std 
改进二  Demodulation

### Demodulation

移除 normalization操作， 转成 w -> w' = s w -> w'' = w'/ $\sqrt{\sum w'}$


![2_norm](images/stylegan2_normalization.png)

## 3 基于PPL的正则化

## 3.1 感知质量评估
PPL(perceptual path length)跟感知图像质量的关系, 

PPL 越小， 图像质量更好

## 



# 4 改进 progressive growing

## 4.1 


   
