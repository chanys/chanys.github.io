---
layout: post
title: Likelihood based Generative Models
---

We perform function approximate, by learning $\theta$ so that $p_{\theta}(x) \approx p_{\text{data}}(x)$. That is, given any particular sample $x$, the probability estimated by your model for $x$ approximates the real ground truth probability.
We want to fit distribution (the distribution is the data). We want to find a model that can minimize the loss of fidelity in the distribution fit: 
$$\text{arg-min}_{\theta} \text{ loss}(\theta, x^{(1)}, \ldots, x^{(n)}) = \frac{1}{n} \sum_{i=1}^{n} - \text{log } p_{\theta}(x^{(i)})$$
	
I.e. given any sample $x$ that I've drawn from $p_{\text{data}}$, the $\theta$ that I've learnt should be such that $p_{\theta}(x)$ is very close to $p_{\text{data}}(x)$.
When I am training my model, the gradient will tell me which way to shift my parameter (either add to it, or minus from it) so as to reduce the loss. (image that you have a 2-D plot, where x-axis is the different values that a particular parameter $\theta_{j}$ can take, and your y-axis is your loss). You can think of loss as the "gap" between your current (model) distribution vs the (gold/target) distribution.
