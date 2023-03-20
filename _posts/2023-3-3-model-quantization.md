---
layout: post
title: Model Quantization
---

To reduce inference runtime, we can also perform quantization, which converts 32-bit floating points to 8-bit integers. This makes inference computation more efficient and reduces memory consumption. When quantizing deep neural models weights, we are distributing the (relatively narrow) range of floating points to a range of integers, clamping any outliers, and then rounding to whole numbers. 

Following is an example code that quantize a particular weight layer of DistilBERT:
```
from transformers import AutoTokenizer, DistilBertModel

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name, output_hidden_states=True)
state_dict = model.state_dict()
weights = state_dict['transformer.layer.0.attention.out_lin.weight']

zero_point = 0
scale = (weights.max() - weights.min()) / (127 - (-128))

# char(): 8-bit signed int
quantized_weights = (weights / scale + zero_point).clamp(-128, 127).round().char()
```

In the above, `scale=0.0053`, and we show the values of the original `weights` matrix, and the `quantized_weights` matrix:
```
tensor([[-0.0283, -0.0414,  0.0004,  ..., -0.0333, -0.0190,  0.0438],
        [ 0.0440,  0.0149,  0.0072,  ..., -0.0220,  0.0383,  0.0030],
        [-0.0457, -0.0289,  0.0271,  ...,  0.0017,  0.0291, -0.0178],
        ...,
        [ 0.0283,  0.0011,  0.0666,  ..., -0.0007,  0.0312, -0.0036],
        [ 0.0002, -0.0118, -0.0648,  ...,  0.0615, -0.0415, -0.0704],
        [-0.0665, -0.0050, -0.0499,  ...,  0.0446,  0.0102, -0.0099]])

tensor([[ -5,  -8,   0,  ...,  -6,  -4,   8],
        [  8,   3,   1,  ...,  -4,   7,   1],
        [ -9,  -5,   5,  ...,   0,   5,  -3],
        ...,
        [  5,   0,  13,  ...,   0,   6,  -1],
        [  0,  -2, -12,  ...,  12,  -8, -13],
        [-13,  -1,  -9,  ...,   8,   2,  -2]], dtype=torch.int8)
```

One reason why deep neural networks such as Transformers are good candidates for quantization is because their weights often take values within a narrow range, thus making it easier to spread across 256 integer numbers. For instance, `weights.max()` and `weights.min()` from above give values of `0.7397` and `-0.6100` respectively.

Besides more efficient computation runtime, since we are now using 8-bit integers rather than 32-bit floating-points, quantization also reduces memory storage by up to a factor of 4:
```
print(sys.getsizeof(weights.storage()) / sys.getsizeof(quantized_weights.storage()))

3.999755879241598
```

## Dynamic Quantization
In particular, PyTorch makes it easy to use 
PyTorch makes it easy to apply quantization on the weights of an existing model using dynamic quantization with a single function call:
`model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`

With dynamic quantization, we are pre-quantizing the weights of neural models to integers, and the activations are dynamically quantized during inference.
