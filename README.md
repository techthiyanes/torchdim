---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

First-class Dimensions for PyTorch
==================================

_The functionality of [einops](http://einops.rocks) (einsum, rearrange), batching ([vmap](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap), [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html)), and tensor indexing with one new concept in PyTorch_

The tensor input to a resnet might have the shape [8, 3, 224, 224] but informally we think of those dimensions as 'batch', 'channel', 'width', and 'height'. Eventhough 'width' and 'height' have the same _size_ we still think of them as separate dimensions, and if we have two _different_ images, we think of both as sharing the _same_ 'channel' dimension.

 Instead of treating this concept informally, first-class dimensions introduce a Python object, a `Dim`, to represent the concept. By expanding the semantics of tensors with dim objects, we can get behavior equivalent to batching transforms (xmap, vmap), einops-style rearragement, and loop-style tensor indexing.

Installation
============

First-class dims are a library that extends PyTorch, so they need to be installed separately.
We may eventually upstream them into PyTorch itself along with `functorch`.

We have to install a nightly build of PyTorch so first set up an environment:

    conda create --name dim
    conda activate dim

First-class dims requires a fairly recent nightly build of PyTorch so that functorch will work. You can install it using one of these commands:

    # For CUDA 10.2
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    # For CUDA 11.3
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    # For CPU-only build
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

Install dim. You will be asked for github credentials to access the fairinternal organization.

    pip install ninja  # Makes the build go faster
    pip install --user "git+https://github.com/fairinternal/torchdim"


Creating and Binding Dims
=========================

Python objects that represent dimension are created using the `dims` operator, which will return as many new dimensions as it is assigned to.[^1]

```{code-cell} ipython3
import torch
from torchdim import dims

batch, channel, width, height = dims()
```

Other representations such as [Named Tensor](https://pytorch.org/docs/stable/named_tensor.html) in PyTorch, or  [JAX's xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html) use strings to name dimensions. We call these dimensions _first class_ because they are Python objects.

In addition to the normal _positional_ dimensions in a tensor, tensors can also have a separate set of first-class dimensions. You can create tensors with first-class dimensions by binding them using indexing:

```{code-cell} ipython3
input = torch.rand(2, 3, 224, 224)
print(input.ndim)
# 4
```

```{code-cell} ipython3
input_fc = input[batch, channel, width, height]
print(input_fc.dims) # first class dimensions
# (batch, channel, width, height)
```

```{code-cell} ipython3
print(input_fc.ndim) # positional dimensions
# 0
```

```{code-cell} ipython3
input_mixed = input[batch, :, :, height]
print(input_mixed.dims)
# (batch, height)
```

```{code-cell} ipython3
print(input_mixed.ndim)
# 2
```

Dimensions will take on the size of the first thing they are bound to:

```{code-cell} ipython3
print(batch.size)
# 2
```

But you can also directly set the size of dimension:

```{code-cell} ipython3
i = dims()
i.size = 5 # ok, i previously did not have a size

i.size = 5 # ok, it already had the size 5
try:
    i.size = 3
except:
    # error! already set to size 3
    pass
j = dims(4) # can also be set on construction
```

[^1]: To implement `dims`, we use a bit of Python introspection using the C API to so that it always returns the number of dimensions it is bound to and sets their debug names to the variable name.

Semantics of Dimensions
=======================
Three rules fully define how dimension objects work with existing tensors APIs. The power of this abstraction arises from the composition of these rules with existing operators.

Rule 1: Implicit Batching
-------------------------
**Tensor operations (e.g. `input + bias`) are implicitly batched over the union of the first-class dimensions in their inputs.**

If `input` has dimensions `batch, channel` and `bias` has dimension `channel`, the output will have the union of those dimensions (`batch, channel`), and the result will computed as if there was a loop over all the first-class dimensions.

```{code-cell} ipython3
input = torch.rand(128, 32)
bias = torch.rand(32)
batch, channels = dims()
result = input[batch, channels] + bias[channels]
print(result.dims)
# (batch, channels)
```

It is helpful think of operators on tensors with first-class dimensions by analogy to code with explicit loops over dimensions, with the first-class dimensions of the inputs acting as implicit `for` loops, and the values in the tensor being scalars within the body of the loop:

    # mental model: loop-level analogy
    for batch in range(batch.size):
        for channel in range(channel.size):
            result[batch, channels] = input[batch, channels] + bias[channels] # arithmetic on scalars

Positional dimensions behave as they did before (e.g. for + they will broadcast), and can be thought of as being a standard tensor _used within the implicit loops_ defined by first-class dimensions.

This rule parallels the rules for named dimensions in xmap, or the implicitly batched dimensions in vmap.

Rule 2: Specifying dimensions
-----------------------------
**Wherever an integer is used to specify a dimension in the existing torch operator, a first-class dimensions can be used instead to tell the operator to work over that dimension.**

```{code-cell} ipython3
avg_pixel_color = input_fc.mean((width, height))
print(avg_pixel_color.dims)
# (batch, channel)
```

Any other ther first-class dimensions (e.g. batch, channel) are still implicitly batched according to Rule #1.

Rule 3: Dims are Tensors
------------------------
**A first-class dimension `d` can be used wherever a Tensor is expected. It will act as if it were a tensor whose only dimension is itself, `d`, and the values along the dimension are the indices of each entry `(0, 1, 2, ..., d.size - 1)`**

```{code-cell} ipython3
print(channel.dims)
# (channel,)
```

```{code-cell} ipython3
print(channel + 1000)
# tensor([1000, 1001, 1002])
```

This means that a dimensions used as a tensor acts as an index into that dimension. Going back to our loop-level analogy, it is analogous to using the loop variable as a value:

    # mental model: loop-level analogy
    for channels in range(batch.size):
        compute channels + 1000

This makes doing complicated indexing arithmetic appear the same as it would in a for loop, but without executing a loop in Python. Here is code that lookups up features in an embedding table given a sequence of ids:

```{code-cell} ipython3
sequence, features = dims()
embeddings = torch.rand(8, 128)
words = torch.tensor([5, 4, 0,])

state = embeddings[words[sequence], features]
print(state.dims)
# (sequence, features)
```

With the following analogy to loops:

    # mental model: loop-level analogy

    for sequence in range(words.size(0)):
        for features in range(embeddings.size(1)):
            state = embeddings[words[sequence], features]

Earlier we showed how to bind tensors dimensions is done with indexing `A[i, j]`. In fact, this binding is just the normal indexing operator. Its behavior follows directly from the behavior of indexing with tensor indices combined with Rule #3 and Rule #1. The expression `A[i + 1, j]` also creates a tensor with dimensions `i` and `j` but with a different indexing math. The implementation knows when simple indexing patterns are used and only actually runs a kernel to do indexing when needed.

Unbinding Dims
-------------
The `order` method converts first-class dimensions in a tensor back to a normal positional dimensions by specifying an order for those dimensions.[^2]

By specifiying a different order from how things were originally bound, it is easy to do transpositions.

```{code-cell} ipython3
i, j = dims()
A = torch.rand(3, 4)
A_T = A[i, j].order(j, i)
assert torch.allclose(A.T, A_T)
```

[^2] `order` is actually just a synonym for the already-existing `permute` method, which takes a list a dimension specifiers and puts the tensor in that order because rule #2 says that first-class dims can be passed as arguments to functions that previousely took only integers as dimension. However, the name `permute` is confusing in this context since it implies dim objects have an original order, so we prefer to use `order` when writing code.

Flattening and Splitting Dims
-----------------------------

**Tuples of dimensions** can be passed to both indexing and `order`. In indexing, this will split the dimension being indexed across the dimensions in the tuple.  In `order` it will flatten the dimensions in a single positional dimension:

```{code-cell} ipython3
i, j, k = dims()
j.size = 2
A = torch.rand(6, 4)
a = A[(i, j), k] # split dim 0 into i,j
print(i.size, j.size, k.size)
# 3 2 4
```

```{code-cell} ipython3
r = a.order(i, (j, k)) # flatten j and k
print(r.shape)
# torch.Size([3, 8])
```

The size of one unsized dimension in a tuple such as `i` can be inferred if the other sizes are known.

Examples
========

The usefulness of dimension objects is best seen through examples. Let's look at some different ways they can be used.

Einsum-style Products
---------------------
Rather than having [einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html) as a custom operator, it is possible to express matrix products directly as a composition of multiplies and summations. The implementation will pattern match any multiplication followed by a sum to the right matrix-multiply operator.

```{code-cell} ipython3
def mm(A, B):
    i, j, k = dims()
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)
mm(torch.rand(3, 4), torch.rand(4, 5)).shape
```

Creating a batched version of multiply is easy, just add a `b` dimensions to everything:

```{code-cell} ipython3
def bmm(A, B):
    b, i, j, k = dims()
    r = (A[b, i, k] * B[b, k, j]).sum(k)
    return r.order(b, i, j)
bmm(torch.rand(3, 4, 5), torch.rand(3, 5, 6)).shape
```

You no longer need to recognize that productions like attension are matrix multiplies, the canonical way of writing them will turn into the right optimized operators:

```{code-cell} ipython3
from torchdim import softmax
def attention(K, Q, V):
    batch, channel, key, query = dims()
    A = (K[batch, channel, key]*Q[batch, channel, query]).sum(channel)
    A = softmax(A * (channel.size ** -0.5), dim=key)
    R = (V[batch, channel, key] * A).sum(key)
    return torch.cat((R.order(batch, channel, query), Q), dim=1)

attention(*(torch.rand(2, 3, 4) for _ in range(3))).shape
```

`einops`
------

[Einops](http://einops.rocks) is an extension to einsum that adds support for the manipulation of dimensions through a few custom operators such as `rearrange` or `repeat`.

First-class dimensions can accomplish the same goal, using PyTorch's existing operator set.

Here are some examples from the [einops tutorial](http://einops.rocks/pytorch-examples.html) showing what the equivalent code looks like with dimension objects:

```{code-cell} ipython3
def pixel_shuffle_einops(img, upscale_factor=2):
    from einops import rearrange
    return rearrange(img, 'b (c h2 w2) h w -> b c (h h2) (w w2)', h2=upscale_factor, w2=upscale_factor)

def pixel_shuffle(img, upscale_factor=2):
    h2, w2, c, b, h, w = dims(upscale_factor, upscale_factor)
    return img[b, (c, h2, w2), h, w].order(b, c, (h, h2), (w, w2))

pixel_shuffle(torch.rand(2, 8, 10, 10)).shape
```

Restyling Gram matrix for style transfer

```{code-cell} ipython3
def gram_matrix_new_einops(y):
    b, ch, h, w = y.shape
    return torch.einsum('bchw,bdhw->bcd', [y, y]) / (h * w)
```

```{code-cell} ipython3
def gram_matrix_new(y):
    b, c, c2, h, w = dims()
    return (y[b, c, h, w] * y[b, c2, h, w]).sum((h, w)).order(b, c, c2) / (h.size * w.size)
gram_matrix_new(torch.rand(1, 2, 3, 4))
```

`vmap`, `xmap`
------------

The implicit batching of Rule #1 means it is easy to created batched versions of existing PyTorch code. The way of specifying how to batch has lighter weight syntax as well.

```{code-cell} ipython3
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size)

def model(feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
batch = dims()
model(examples[batch])
# vs: result = functorch.vmap(model)(examples)
```

Because xmap and vmap are transforms over functions, there is a lot of syntactic distance between the specification of the dimension mappings, and the values where those mappings apply. Dims express the mapping as indexing of the tensor, right at the place where the function is being applied.


[xmap examples](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html):
```
in_axes = [['inputs', 'hidden', ...],
           ['hidden', 'classes', ...],
           ['batch', 'inputs', ...],
           ['batch', ...]]

loss = xmap(named_loss, in_axes=in_axes, out_axes=[...])
print(loss(w1, w2, images, labels))
```

Dimension objects:

```
batch, inputs, hidden, classes = dims()
print(loss(w1[inputs, hidden], w2[hidden, classes], images[batch, inputs], labels[batch]))
```

This pattern also composes well with other code that also uses first class dimensions. For instance, another way to write `bmm` from above is to batch the `mm` operator.
It doesn't matter whether the implementation of the function uses dimension objects, it is also possible to add additional batch dimensions and then call a function:

```{code-cell} ipython3
def bmm_2(A, B):
    i = dims() # note: doesn't matter than mm internally also uses i
    return mm(A[i], B[i])
```

Multi-headed Attention
---------------------

```{code-cell} ipython3
def multiheadattention(q, k, v, num_attention_heads, dropout_prob, use_positional_embedding):
    batch, query_sequence, key_sequence, heads, features = dims()
    heads.size = num_attention_heads

    # binding dimensions, and unflattening the heads from the feature dimension
    q = q[batch, query_sequence, [heads, features]]
    k = k[batch, key_sequence, [heads, features]]
    v = v[batch, key_sequence, [heads, features]]

    # einsum-style operators to calculate scores
    attention_scores = (q*k).sum(features) * (features.size ** -0.5)

    # use first-class dim to specify dimension for softmax
    attention_probs = softmax(attention_scores, dim=key_sequence)

    # dropout work pointwise, following Rule #1
    attention_probs = torch.nn.functional.dropout(attention_probs, p=dropout_prob)


    context_layer = (attention_probs*v).sum(key_sequence)

    # flatten heads back into features
    return context_layer.order(batch, query_sequence, [heads, features])
```

Notice the combination of features: binding dimensions and unflattening heads from the features, using einsum style products, and calling dropout through implicit batching.

Indexing
--------

Rule #3 enables indexing because dimensions act as loop indices when used as a tensor.
It is easy to have computed based on indices, such as the upper triangular operator:

```{code-cell} ipython3
from torch import where
def triu(A):
   i,j = dims()
   a = A[i, j]
   return torch.where(i <= j, a, 0).order(i, j)
triu(torch.rand(3, 4))
```

Embedding bag does an embedding table lookup followed by a sum:

```{code-cell} ipython3
def embedding_bag(input, embedding_weights):
    batch, sequence = dims()
    r = embedding_weights[input[batch, sequence], features].sum(sequence)
    return r.order(batch, features)

input = torch.tensor([[1, 0, 4, 3]])
W = torch.rand(5,)
```

Relative positional embeddings associate an embedding vector with the distance between the query and the key in the sequence.
For instance, a key two elements after query after key will get embedding ID 2. We can use first-class dimensions to do the indexing arithmetic, and embedding lookup:

```{code-cell} ipython3
def relative_positional_embedding(q, k, distance_embedding_weight):
    batch, query_sequence, key_sequence, heads, features = dims()
    q = q[batch, query_sequence, [heads, features]]
    k = k[batch, key_sequence, [heads, features]]

    distance = query_sequence - key_sequence
    n_embeddings = distance_embedding_weight.size(0)
    index_bias = n_embeddings // 2

    assert key_sequence.size + bias <= n_embeddings

    # indexing with dims
    positional_embedding = distance_embedding_weight[distance + index_bias, features]

    # matrix multiplies with dims
    relative_position_scores_query = (q*positional_embedding).sum(features)
    relative_position_scores_key = (k*positional_embedding).sum(features)
    return  (relative_position_scores_query + relative_position_scores_key).order(batch, heads, key_sequence, query_sequence)
```

Tensor Puzzlers
===============

[Tensor Puzzlers](https://github.com/srush/Tensor-Puzzles), created by Sasha Rush, are a good excersize for learning the numpy and torch APIs by figuring out how to define common operations using a small set of primitive tensor operations.

However, the difficulty of many of the puzzlers lies not in how to compute the answer but the awkwardness of the primitives themselves.

**With first class dimensions, these puzzlers are nearly the same as the spec that defines them**


### Puzzle 3 - outer

Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.

```{code-cell} ipython3
def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]

def outer(a, b):
    i, j = dims()
    return (a[i] * b[j]).order(i, j)
```

### Puzzle 4 - diag

Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.

```{code-cell} ipython3
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]

def diag(a):
    # the syntax closely matches the spec
    i = dims()
    return a[i, i].order(i)
```

### Puzzle 5 - eye

Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.

```{code-cell} ipython3
from torch import where
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1

def eye(j: int):
    i,j = dims(j, j)
    return where(i == j, 1, 0).order(i, j)
```

### Puzzle 6 - triu

Compute [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) - the upper triangular matrix.

```{code-cell} ipython3
def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0

def triu(j: int):
    i,j = dims(j, j)
    return where(i <= j, 1, 0).order(i, j)
```

### Puzzle 8 - diff

Compute [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) - the running difference.

```{code-cell} ipython3
def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]
def diff(a, i: int):
    i = dims()
    d = a[i] - a[i - 1]
    return where(i - 1 >= 0, d, a[i]).order(i)
```

### Puzzle 9 - vstack

Compute [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors

```{code-cell} ipython3
def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]

def vstack(a, b):
    v, i = dims(2)
    return where(v == 0,  a[i], b[i]).order(v, i)
```

### Puzzle 10 - roll

Compute [roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) - the vector shifted 1 circular position.

```{code-cell} ipython3
def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 < len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]

def roll(a, i: int):
    i = dims(a.size(0))
    return a[where(i + 1 < i.size, i + 1, 0)].order(i)
```

### Puzzle 11 - flip

Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector

```{code-cell} ipython3
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]

def flip(a, i: int):
    i = dims(a.size(0))
    return a[i.size - i - 1].order(i)
```

### Puzzle 14 - sequence_mask


Compute [sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) - pad out to length per batch.

```{code-cell} ipython3
def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j < length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0

def sequence_mask(values, length):
    j, i = dims()
    v = values[i, j]
    return where(j < length[i], v, 0).order(i, j)
```

Advantages of First-class Dimensions over Named (string) Dimensions
===================================================================

The most prominent difference between first-class dimensions and alternatives such as einops, named tensors (and [tensors considered harmful](https://nlp.seas.harvard.edu/NamedTensor), or xmap is that dimensions are objects rather than strings. Using objects has a number of nice properties.

### Avoiding naming conflicts

Using strings for dimensions introduces the possibility that two unrelated dimensions are given the same name. Using objects instead makes it clear the same names are not the same dimension. It's like the difference between having only global variables, and having the ability to locally bind names in functions.
 For instance, we defined `bmm` by batching a call to `mm`, and even though they both use the name `i` to identify a dimension.  Because each `i` is a different object, there is no naming conflict:

```{code-cell} ipython3
def mm(A, B):
    i, j, k = dims()
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)

def bmm(A, B):
    i = dims() # note: doesn't matter than mm internally also uses i
    return mm(A[i], B[i])
```

Einops avoids conflicts by ensuring names are all introduced and removed in a single expression, but this precludes using long-lived dimensions to present implicit batching similar to xmap. When nested, JAX's xmap seems to consider axes the same if the string name matches. In the above example it would consider the `i` dimension to be the same dimension in both `bmm` and `mm` so the code would error.


### Reuse the same operator set

Having a new object type allows us to extend the existing operator set of PyTorch rather than come up with new operators. For instance, binding dimensions using indexing follows semanticalaly from Rules #1 and #3, so there is no need for a special operator to do binding. Even unbinding is just the `permute` operator which follows from Rule #2, though we call it `order` for clarity. In contrast, using strings requires coming up with new APIs such as `einsum` for matrix multiplies, or `rearrange` for doing permutations.

### Allows dims to act as tensors

Rule #3 is not possible with strings since we cannot make strings behave as tensors. Without this rule, all of the indirect indexing that dims enable would not longer be easy to express.

### Dims can have methods
For instance, as objects, dims can have a size, which allows us to do size inference of dimensions in various places in the API where string based APIs would have to take additional arguments specifying size.


Comparison to tensor compilers
==============================
(unfinished)

TVM, Halide, XLA, Dex
* ability to still have slow dynamically typed execution
* no syntactic overhead switching between the compiled mode and the python mode
* Tensors do not have to be entirely in loop-explicit mode or pytorch mode, tensors with some first-class dims can compute through normal tensor code.
* We still have the oppurtunity to optimize by lazily implementing some operators similar to how multiply and sum work now.
