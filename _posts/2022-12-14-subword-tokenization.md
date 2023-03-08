---
layout: post
title: Subword Tokenization
---

In this article, we describe Byte pair encoding (BPE), WordPiece tokenization, and Unigram tokenization. One of the first steps in processing a piece of text, is to tokenize it, i.e. split it into "chunks". One simple method is to split on spaces, but (i) some languages like Chinese and Arabic do not come with spaces, (ii) merely splitting on spaces will result in a very large vocabulary size, forcing models to have very large embedding matrices. On the other end of the spectrum, we could split into individual characters. However, it is hard to learn meaningful representations on individual characters. Transformer models use subword tokenization, which splits single words into one or more subwords. 

The principle behind subword tokenization is that frequently used words should not be split into smaller subwords, but rare words should be split into meaningful subwords. Subword tokenization enables smaller vocabulary sizes, while retaining the ability to learn meaningful (subword) representation. Plus, we could also split a previously unseen word into known subwords.

# Byte Pair Encoding (BPE) Subword Tokenization
Byte Pair Encoding (BPE) is a data compression algorithm that had been repurposed for subword tokenization in NLP. 

When training a BPE tokenizer on a corpus of text, we perform the following operations:
1. First perform any required normalization and initial tokenization (e.g. split by white space).
2. Tag on the character `</w>` to end of each word, split words into individual characters, and count the occurrence frequency of individual characters. 
	* The end-of-word `</w>` helps to disambiguate between substrings, e.g. "ox" in "f**ox**" and "**ox**ygen". By keeping track of word delimiters, it also enables us to trivially reconstruct the original text when given a (subword) tokenized string.
3. Iteratively merge pairs of consecutive characters. In each iteration, (i) keep track of the merge rule, (ii) merge the most frequent character pair. 
4. Repeat the above iterative merges, until a predefined number of merges, or until the vocabulary reaches a certain size. When the process stops, we have an ordered list of merging rules.

Applying a trained BPE tokenizer on a new text corpus is simple:
* Apply steps 1 (normalization, initial tokenization) and 2 (tag on `</w>` and split into individual characters).
* Apply the merging rules in order.

Given the string "The big brown fox jumps over the box and ox":
* First split into words and tag on `</w>` at the end of each word like so: `["The</w>", "big</w>", "brown</w>", "fox</w>", "jumps</w>", "over</w>", "the</w>", "box</w>", "and</w>", "ox</w>"]`.
* Break up the words into individual characters and count their occurrence frequency:
   * `[T, h, e, </w>, b, i, g, </w>, b, r, o, w, n, </w>, f, o, x, </w>, j, u, m, p, s, </w>, o, v, e, r, </w>, t, h, e, </w>, b, o, x, </w>, a, n, d, </w>, o, x, </w>]`
   
   | Frequency | Frequency | Frequency | Frequency |
     |:--:|:--:|:--:|:--:|
	 |w:8| T:1 | a:1 | b:3 | 	 
	 | d:1 | e:3 | f:1 | g:1 | 	 
	 | h:2 | i:1 | j:1 | m:1 |
	 | o:5 | p:1 | r:2 | s:1 |
     |t:1|u:1|v:1|	 w:1|
	 |x:3|n:2|||
* Merge the most frequent consecutive character pair. Here, "ox" and "x</w>" are the most frequent. Let's just choose to merge "ox" (which occurs 3 times). We (i) capture this merge rule, (ii) minus off the frequency of "ox" from the frequency of its individual components, resulting in the following updated frequency table and list:
	*  `[T, h, e, </w>, b, i, g, </w>, b, r, o, w, n, </w>, f, ox, </w>, j, u, m, p, s, </w>, o, v, e, r, </w>, t, h, e, </w>, b, ox, </w>, a, n, d, </w>, ox, </w>]`
	
   | Frequency | Frequency | Frequency | Frequency |
   |:--:|:--:|:--:|:--:|
   |w:8| T:1 | a:1 | b:3 | 	 
   |d:1 | e:3 | f:1 | g:1 | 	 
   |h:2 | i:1 | j:1 | m:1 |
   |**o:5-3=2** | p:1 | r:2 | s:1 |
   |t:1|u:1|v:1|	 w:1|
   |**x:3-3=0**|n:2|**ox:3**||
* Next, we merge `("ox", "</w>")` and then merge `(h, e)`, resulting in the following updates:
   *  `[T, he, </w>, b, i, g, </w>, b, r, o, w, n, </w>, f, ox</w>, j, u, m, p, s, </w>, o, v, e, r, </w>, t, he, </w>, b, ox</w>, a, n, d, </w>, ox</w>]`

   | Frequency | Frequency | Frequency | Frequency |
   |:--:|:--:|:--:|:--:|
   |**w:8-3=5**| T:1 | a:1 | b:3 | 	 
   |d:1 | **e:3-2=1** | f:1 | g:1 | 	 
   |**h:2-2=0** | i:1 | j:1 | m:1 |
   |o:2 | p:1 | r:2 | s:1 |
   |t:1|u:1|v:1|	 w:1|
   |x:0|n:2|**ox:3-3=0**|ox\<\/w\>:3|
   |**he:2**|||
   
We stop the iterative pairwise merging process when we have done a pre-defined number of merges, or when we have reached a certain vocabulary size. In general, the vocabulary size will increase at the beginning (as above), before decreasing. In the example above, if we stop now, we will have captured 3 merging rules in order: `(o, x), ("ox", "</w>"), (h, e)`.

# WordPiece Subword Tokenization
WordPiece subword tokenization is used in BERT, DistlBERT, Electra, etc. It is very similar to BPE, except for two main differences.

Recall that when training a BPE tokenizer, it choses the most frequent consecutive byte pair to merge in each iteration. In WordPiece, it choose the highest scoring pair.
* Given elements $i$, $j$, and denoting occurrence frequency by $\text{freq(.)}$, WordPiece choose to merge the highest scoring consecutive pair: $\frac{\text{freq}(i,j)}{\text{freq}(i) * \text{freq}(j)}$

The next main difference is that WordPiece saves the final vocabulary (but not the learned merged rules). Given a trained WordPiece tokenizer and a word to tokenize, WordPiece finds the longest subword that is in the vocabulary and splits on it. For instance, to split the word "networks":
* If the longest subword (matching from the beginning) in the vocabulary is "net", then we split to get `["net", "##works"]`. 
* Then assume "##work" is the longest subword starting at the beginning of "##works" that is in the vocabulary, so we split to get `["net", "##work", "##s"]`. 

# Unigram Subword Tokenization
Compared to BPE and WordPiece, Unigram subword tokenization works in the other direction. We start from a big vocabulary and iteratively remove vocabulary entries until we reach the desired vocabulary size.
To construct the initial vocabulary, we can use, e.g.: 
(i) all plausible substrings of all words in the corpus
(ii) use the most common substrings of the words
(ii) apply BPE on the initial corpus with a large vocabulary size

At each step of the tokenizer training, the Unigram algorithm computes a loss over the corpus given the current vocabulary. Then, for each entry in the vocabulary, we compute how much the overall loss would increase if the symbol was removed. We can chose to remove p% of the vocabulary entries that would increase the loss the least. 
This process is then repeated until the vocabulary has reached the desired size. 
Note that we never remove the base characters, to make sure any word can be tokenized.

We illustrate Unigram subword tokenization, by using the [Unigram tokenization](https://huggingface.co/course/chapter6/7?fw=pt) example from Hugging Face:
* Assume a text corpus of words with frequencies: `("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)`
* Assume that we initialize the vocabulary to be all the possible substrings (with their respective frequency):
`("h", 15) ("u", 36) ("g", 20) ("hu", 15) ("ug", 20) ("p", 17) ("pu", 17) ("n", 16), ("un", 16) ("b", 4) ("bu", 4) ("s", 5) ("hug", 15) ("gs", 5) ("ugs", 5)`
   * The sum of all frequencies in the vocabulary is 210.

### SubWord Tokenization
We first explain how to tokenize a word and calculate its corresponding "tokenization" probability. This will be used as part of the training process later, to train a Unigram subword tokenizer.

To tokenize a word, we calculate the probabilities of all of its possible segmentations. E.g. to segment "pug":
* $P(["p", "u", "g"]) = P("p") * P("u") * P("g") = \frac{17}{210} * \frac{36}{210} * \frac{20}{210} = 0.001321672$
* $P(["p", "ug"]) = P("p") * P("ug") = \frac{17}{210} * \frac{20}{210} = 0.007709751$
* $P(["pu", "g"]) = P("pu") * P("g") = \frac{17}{210} * \frac{20}{210} = 0.007709751$
We chose the segmentation with the highest probability as the tokenization of the word. So, "pug" would be tokenized as ["p", "ug"] or ["pu", "g"] with probability 0.007709751.

### Training the Tokenizer

During each training iteration, we first tokenize each word in the corpus using the current vocabulary. Assume the same given corpus and initial vocabulary as above, we tokenize each word in the corpus `("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)`)
and arrive at the following (one-best) tokenization/segmentation with (max) associated probabilities:
```
'hug': ['hug'] (score 0.071428)
'pug': ['pu', 'g'] (score 0.007710)
'pun': ['pu', 'n'] (score 0.006168)
'bun': ['bu', 'n'] (score 0.001451)
'hugs': ['hug', 's'] (score 0.001701)
```

In practice though, we calculate the log loss over all the plausible segmentations of all words in the corpus. Assume the training data consists of the words $x_1, \ldots, x_N$, and define $S(x_1)$ as the set of all possible tokenizations for a word $x_1$, then the overall loss of the corpus is:
$$L = -\sum_{i=1}^{N} \text{log}(\sum_{x \in S(x_i)} p(x))$$

We can calculate how removing each item in the vocabulary would affect the overall corpus loss. We then choose to remove $p%$ (where $p$ is usually 10% or 20%) of the vocabulary items that will increase the loss the least.
