Title:  Score-Based Generative Modeling through Stochastic Differential Equations
Date: 20220921
Category: paper
tags: paper, generator, Diffusion, score matching

[TOC]

# 概述
0. Using SDE and reverse-time SDE in continous domain.
1. Generalize Score matching and Diffusion to inifinite T.
2. Two SDEs to use
3. Train like Denoising Score Matching or DDPM.  Sliced Score matching also works.
4. Inference/Generation: Propose Predictor-Corrector sampler (SDE Solver + MCMC)
5. Propose a Deterministic sampler, call “probability flow ordinary DE (ODE)”.
6. Architecture Improvements
7. class-conditional generation, image imputation and colorization

# Unified Framework

The authors proposed a unified framework generalizes score matching [[NCSN]]  and [[DDPM]]

It uses Stochastic Differential Equation(SDE) [SDE 中文]([https://www.bilibili.com/video/BV1At4y197FG](https://www.bilibili.com/video/BV1At4y197FG)) and reverse-time SDE ([derivation English]([https://ludwigwinkler.github.io/blog/ReverseTimeAnderson/](https://ludwigwinkler.github.io/blog/ReverseTimeAnderson/))) to extend discrete T (>1000) to infinite continuous T.

The general form of SDE is: $d\boldsymbol{x} = f(\boldsymbol{x}, t) dt \;\;\;\; + G(\boldsymbol{x}, t) d\boldsymbol{w}$ 
Compared to the diffusion: $x_t \sim \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \;\; \beta_t \boldsymbol{I})$ 

## Two detailed SDEs for the framework

1.  Variance Exploding (VE) SDE: $dx = \sqrt{\frac{d [\sigma^2(t)]}{dt}}dw$ , derived from the Markov Chain: $x_i = x_{i-1} + \sqrt{\sigma_i^2  - \sigma_{i-1}^2} z_{i-1}$,  where $z_{i-1} \sim \mathcal{N}(0, \boldsymbol{I})$ . 
2. Variance Preserving (VP) SDE: $dx = -1/2 \beta(t) x dt + \sqrt{\beta(t)} dw$, derived from DDPM's discrete Markov chain.


# Training

Train a time-dependent score-based model Eq 7, similar to denoising score matching and also DDPM

# Inference/Generation/Sampling after training

1. Apply general SDE Sampler
2. Propose Predictor-Corrector sampler (SDE Solver + MCMC)

# Deterministic sampler

They proposed a Deterministic sampler, called "probability flow ordinary DE (ODE)"

Advantages (v.s. Stochastic sample):

1. Exact likelihood computation (DDPM has its own computation, iDDPM improves the results)
2. Manipulating latent representations, for image editing, such as interpolation, and temperature scaling.
3. Uniquely identifiable encoding
4. Efficient sampling, reduce T>1000 -> T<100.


# Architecture of U-Net
In Appendix H

5 different improvents

And Exponential Moving Average

# Controllable Generation

By solving a conditional reverse-time SDE.

Tasks: class-conditional generation, image imputation and colorization