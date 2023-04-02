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
- Large Language Models are Zero-Shot Reasoners 2022.05.24 https://arxiv.org/abs/2205.11916
- Toolformer: Language Models Can Teach Themselves to Use Tools 2023.02.09  https://arxiv.org/abs/2302.04761
- Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback 2023.02.24 https://arxiv.org/abs/2302.12813
- Reflexion an autonomous agent with dynamic memory and self-reflection 2023.03.20 http://arxiv.org/abs/2303.11366



| 方法缩写 | 一句话概述 | 其他(可选) | 
| --------- | -------- | ----------- |
| few-shot CoT             |  标准提示加入中间推理过程 <input, CoT, output>， 手工选定             | |
| self-consistency CoT     |  用CoT多次生成结果进行投票， 少数服从多数                            |   |
| zero-shot CoT            |  用 Let's think step-by-step 生成推理过程的文本， 进行两阶段推理生成  |   |
| Context-faithful Prompt  |  强调给定的上下文， in one's opinion, based on given text, 和 instruction |   |
| Ask Me Anything          |  用 不知道怎么来的claim 生成 问题， 再回答问题， 加上 weak supervision ， 或者使用 summarize 总结 |   |
| WebGPT                   |  类似 InstructGPT， 监督学习训练 上网命令， 再用检索的数据 用人工训练强化学习  |   |
| Toolformer               |  用prompt让LLM自行估计 API接口符号的接入位置， 再用点 启发式条件 进行过滤   | 7亿多参数，能力就能涌现  |
| LLM-Augmenter            |  先从外部知识库获取evidence来构建 提示prompt， 让LLM生成文本， 需要用强化学习训练 Policy模型和知识检验模型  |   |
| Reflexion                |  在ReAct的基础上， 简化reward， 用Reflect-LLM总结历史行动记录成一段话， 就可以清空历史行动记录  |   |
|   |   |   |
|   |   |   |


### 感觉没啥意思的

- Self-Consistency Improves Chain of Thought Reasoning in Language Models 2022.03.21 https://arxiv.org/abs/2203.11171   
    对多次输出的一个投票、少数服从多数
- Ask Me Anything: A simple strategy for prompting language models 2022.10.05 https://arxiv.org/abs/2210.02441   
    不是 0-shot， 感觉局限性比较大， 读起来也不容易。 生成 QA格式的提示prompts， 多组 prompts 的结果 用 weak supervision WS 整合，而不是简单投票。
- Large Language Models Can Self-Improve 2022.10.20 https://arxiv.org/abs/2210.11610    
    只在PaLM 540B上验证， 就相当于 半监督学习， 加入一堆无标签数据、用语言模型赋上伪标签， 再用来微调
- Context-faithful Prompting for Large Language Models 2023.03.20 https://arxiv.org/abs/2303.11315
    针对当前语料与LM语料学到的知识之间的 Knowledge Conflict 和 Prediction with Abstention 提出几个特定prompt提示模板
- More

