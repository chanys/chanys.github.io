---
layout: post
title: Chain-of-Thought (CoT) Prompting
---

Chain-of-Thought is a prompting strategy introduced in the paper "Chain of thought prompting elicits reasoning in large language models", published in January 2022. CoT prompting improves few-shot performance of various language models and was used in the FLAN-PaLM model.

Standard prompting vs CoT prompting (illustrated in the following Figure extracted from the paper):
![_config.yml]({{ site.baseurl }}/images/CoT_1.png)
* In standard few-shot prompting (popularized by GPT-3), a language model is given in-context exemplars of input-output pairs before producing a prediction for a test-time example.
* In CoT prompting, each exemplar in the few-shot prompt is augmented with a chain of thought for an associated answer. The authors usually use 8-shot exemplars in their prompts.

## Evaluation
* The paper evaluated CoT prompting on arthimetic reasoning, commonsense reasoning, and symbolic reasoning evaluation datasets, using different models: GPT-3, LaMDA, PaLM, UL2 20B, and Codex. 
* They found that CoT prompting does not improve performance for small models. It only improves performance of models around 100B parameters.
* The following Figure (from the paper) illustrates CoT prompts for a variety of tasks:
![_config.yml]({{ site.baseurl }}/images/CoT_2.png)
