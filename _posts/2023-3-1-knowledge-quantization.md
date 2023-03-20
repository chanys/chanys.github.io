---
layout: post
title: Knowledge Distillation
---

Knowledge distillation is a general purpose method for training a smaller student model to mimic the behaviour of a slower, larger, but better performing teacher. It was popularized in a 2015 paper (Distilling the knowledge in a neural network. G. Hinton et al. 2015) that generalized the method to deep neural networks. **The main idea is to augment the ground truth labels with a distribution of "soft probabilities" from the teacher which provides complementary information for the student to learn from.** We describe knowledge distillation and DistilBERT in this article.

For instance, if a teacher model such as BERT assigns high probabilities to multiple classes, then it could be a sign that these classes lie close to each other in the feature space. Thus, we would like to train a student model to mimic these probabilities, where the goal is to distill some of this "dark knowledge" that the teacher has learned (that is not available from the gold labels alone).

### DistilBERT
One example of a knowledge distilled model is DistilBERT, which learnt from the BERT teacher model. In building DistilBERT, the authors from Hugging Face made the following choices:
* Taking advantage of the common dimensionality between the teacher BERT and student DistilBERT, they initialized DistilBERT from BERT by taking one layer out of two.
* Taking note of best practices used in building the RoBERTa language model, DistilBERT was trained on very large batches (up to 4K examples per batch), used dynamic masking, and without the next sentence prediction objective.

### Knowledge distillation during Pretraining
Instead of performing pre-training of DistilBERT using just the masked language model (MLM) objective, we can also apply knowledge distillation during pretraining using the following aggregated loss:
$$L_{DistilBERT} = \alpha L_{mlm} + \beta L_{KD} + \gamma L_{cos}$$

* Where $L_{mlm}$ is the usual MLM loss, $L_{KD}$ is the knowledge distillation loss (described in the next Section), and $L_{cos} = 1 - \text{cos}(h_s, h_t)$ is the cosine embedding loss. It ensures that the hidden state vectors from teacher ($h_t$) and student ($h_s$) are aligned in direction (both positive, or both negative).

### Knowledge distillation during Fine-tuning of Target Task

Given an input example $x$, the teacher generates logits $z(x) = [z_1(x), \ldots, z_N(x)]$ for the different $N$ classes.
* Instead of doing a plain softmax over the logits (which will assign a high probability to one class, and then the teacher doesn't provide much information beyond the ground truth label), we "soften" the prediction probabilities by scaling the logits with a *temperature* parameter $T \gt 1$:
$$p_i(x) = \frac{\text{exp}(z_i(x)/T)}{\sum_{j} \text{exp}(z_j(x)/T)}$$
* We likewise take the **softened probabilties $q_i(x)$ that the student produces**, and calculate KL divergence vs the **teacher's soften probabilities $p_i(x)$**:
$$D_{KL}(p, q) = \sum_{i} p_i(x) \text{log} \frac{p_i(x)}{q_i(x)}$$
* Then calculate knowledge distillation loss, where $T^{2}$ is a normalization factor to account for the fact that the magnitude of the gradients produced by soft labels scales as $\frac{1}{T^{2}}$:
$$L_{KD} = T^{2}D_{KL}$$
* For classification tasks, the student loss is then a weighted average of distillation loss $L_{KD}$, and the usual cross-entropy loss $L_{CE}$ of the ground truth labels, where $\alpha$ is a tunable parameter:
$$L_{student} = \alpha L_{CE} + (1 - \alpha) L_{KD}$$

Finally, the temperature $T$ is set to 1 at inference for the student, to use the standard softmax probabilities.
