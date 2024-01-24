---
layout: post
title: Perplexity
---

Perplexity is commonly used to quantify the quality of a language model.

* In terms of language modeling, cross entropy indicates the average number of bits needed to encode one word, and perplexity is the number of words that can be encoded with those bits. Denoting PPL as perplexity and CE as cross-entropy, then for a sequence $X$, $PPL(X)  = 2^{CE(X)}$

* So, if CE = 2, then PPL = 4 (which can be interpreted as the average branching factor). I.e. when trying to guess the next word, the language model is as confused as if it had to pick between 4 different worrds. **So to minimize perplexity, you minimize cross-entropy, which means bringing to modeling distribution $Q$ closer to the real underlying distribution $P$.** 

* Given a sequence $X = \[x_{0}, \ldots, x_{t}\]$, its likelihood is $P(X) = \prod_{0 \le i \le t} p(x_{i}|x_{\lt i})$
  
* Cross entropy $CE(X) = - \frac{1}{t} \text{log} P(X)$
	* you can interpret this as the loss of the language model
   
* Perplexity $PPL(X) = e^{CE(X)} = e^{-\frac{1}{t} \sum_{0 \le i \le t} \text{log } p(x_{i}\|x_{\lt i})}$
	* The above shows that perplexity is closely related to the loss. Since the loss is only a weak proxy for the model's ability to generate quality text, we often also evalute the language model in a downstream task.
