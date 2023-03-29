# 我关注的NLP前沿


起因就是发现NLP领域的进展完全没了解

虽然我现在的电脑用不起 大语言模型， 但不影响我看个乐呵

应该主要关注 语言模型 Large Language Model - Prompt - chain of thoughts 这个路线

## LLM

- GPT系列
- BERT
- T5
- BART
- GLM

还有 Transformer-XL, XLNet, UniLM 大差不差

## prompt

由于大模型实在太大了，  反向传播更新prompt 大概成本上也不能接受——只能访问API的话，也得不到梯度

所以更关注 chain-of-thoughts 这种不依赖梯度的

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models  2022.01.28 https://arxiv.org/abs/2201.11903
- Self-Consistency Improves Chain of Thought Reasoning in Language Models 2022.03.21 https://arxiv.org/abs/2203.11171 
- Large Language Models are Zero-Shot Reasoners 2022.05.24 https://arxiv.org/abs/2205.11916
- Ask Me Anything: A simple strategy for prompting language models 2022.10.05 https://arxiv.org/abs/2210.02441
- Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback 2023.02.24 https://arxiv.org/abs/2302.12813
- Context-faithful Prompting for Large Language Models 2023.03.20 https://arxiv.org/abs/2303.11315
- Reflexion an autonomous agent with dynamic memory and self-reflection 2023.03.20 http://arxiv.org/abs/2303.11366