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

First-class Dimensions
======================

_The functionality of einops (einsum, rearrange), batching (vmap, xmap), and tensor indexing with one new concept_

The tensor input to a resnet might have the shape [8, 3, 224, 224] but informally we think of those dimensions as 'batch', 'channel', 'width', and 'height'. Eventhough 'width' and 'height' have the same _size_ we still think of them as separate dimensions, and if we have two _different_ images, we think of both as sharing the _same_ 'channel' dimension.

 Instead of treating this concept informally, first-class dimensions introduce a Python object, a `Dim`, to represent the concept. By expanding the semantics of tensors with dim objects, we can get behavior equivalent to batching transforms (xmap, vmap), einops-style rearragement, and loop-style tensor indexing.

 Installation
 ============
We have to install a nightly build of PyTorch so first set up an environment:

    conda create --name dim
    conda activate dim

First-class dims requires a fairly recent nightly build of PyTorch so that functorch will work. You can install it using one of these commands:

    # For CUDA 10.2
    pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html --upgrade
    # For CUDA 11.3
    pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html --upgrade
    # For CPU-only build
    pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --upgrade

Install dim. You will be asked for github credentials to access the fairinternal organization.

    pip install ninja  # Makes the build go faster
    pip install --user "git+https://github.com/fairinternal/dynamic_torchscript_experiments#egg=dim&subdirectory=dim"


Creating and Binding Dims
=========================

Python objects that represent dimension are created using the `dims` operator[^1].

```{code-cell} ipython3
import torch
from dim import dims

batch, channel, width, height = dims()
```

Other representations such as [Named Tensor](https://pytorch.org/docs/stable/named_tensor.html) in PyTorch, or  [JAX's xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html) use strings to name dimensions. We call these dimensions _first class_ because they are instead Python objects.

In addition to the normal _positional_ dimensions in a tensor, tensors can also have a separate set of first-class dimensions. You can create tensors with first-class dimensions by binding them using indexing:

```{code-cell} ipython3
input = torch.rand(2, 3, 224, 224)
print(input.ndim)
```

```{code-cell} ipython3
input_fc = input[batch, channel, width, height]
print(input_fc.dims) # first class dimensions
```

```{code-cell} ipython3
print(input_fc.ndim) # positional dimensions
```

```{code-cell} ipython3
input_mixed = input[batch, :, :, height]
print(input_mixed.dims)
```

```{code-cell} ipython3
print(input_mixed.ndim)
```

Dimensions will take on the size of the first thing they are bound to:

```{code-cell} ipython3
print(batch.size)
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

[^1]: We use a bit of Python introspection using the C API to so that `dims` always returns the number of dimensions it is bound to and sets their debug names well.

Semantics of Dimensions
=======================
Three rules fully define how dimension objects work with existing tensors APIs. The power of this abstraction arises from the composition of these rules with existing operators.

Rule 1: Implicit Batching
-------------------------
**Tensor operations (e.g. `input + bias`) are implicitly batched over the union of the first-class dimensions in their inputs.**

If `input` has dimensions `batch, channel` and `bias` has dimension `channel`, the output will have the union of those dimensions (`batch, channel`), and the result will computed as if there was a loop over all the first-class dimensions. It is helpful think of operators on tensors with first-class dimensions by analogy to code with explicit loops over dimensions, with the first-class dimensions of the inputs acting as implicit `for` loops, and the values in the tensor being scalars within the body of the loop:

    # mental model: loop-level analogy
    for batch in range(batch.size):
        for channel in range(channel.size):
            compute input + bias # arithmetic on scalars

Positional dimensions behave as they did before (e.g. for + they will broadcast), and can be thought of as being standard tensor _used within the implicit loops_ defined by first-class dimensions.

 This rule parallels the rules for named dimensions in xmap, or the implicitly batched dimensions in vmap.

Rule 2: Specifying dimensions
-----------------------------
**Wherever an integer is used to specify a dimension in the existing torch operator, a first-class dimensions can be used instead to tell the operator to work over that dimension.**

```{code-cell} ipython3
avg_pixel_color = input_fc.mean((width, height))
print(avg_pixel_color.dims)
```

Any other ther first-class dimensions (e.g. batch, channel) are still implicitly batched according to Rule #1.

Rule 3: Dims are Tensors
------------------------
**A first-class dimension `d` can be used wherever a Tensor is expected. It will act as if it were a tensor whose only dimension is itself, `d`, and the values along the dimension are the indices of each entry `(0, 1, 2, ..., d.size - 1)`**

```{code-cell} ipython3
print(channel.dims)
```

```{code-cell} ipython3
print(channel + 1000)
```

This means that a dimensions used as a tensor acts as an index into that dimension. Going back to our loop-level analogy, it is analogous to using the loop variable as a value:

    # mental model: loop-level analogy
    for channels in range(batch.size):
        compute channels + 1000

This makes doing complicated indexing arithmetic appear the same as it would in a for loop, but without executing a loop in Python. Consider what a lookup from an embedding table would look like:

```{code-cell} ipython3
sequence, features = dims()
embeddings = torch.rand(8, 128)
words = torch.tensor([5, 4, 0,])

state = embeddings[words[sequence], features]
print(state.dims)
```

With the following analogy to loops:

    # mental model: loop-level analogy

    for sequence in range(words.size(0)):
        for features in range(embeddings.size(1)):
            state = embeddings[words[sequence], features]


Removing Dims
-------------
The `positional` method converts a first-class dimensions in a tensor back to a normal positional dimensions.

By specifiying a different order from how things were originally bound, it is easy to do transpositions.

```{code-cell} ipython3
i, j = dims()
A = torch.rand(3, 4)
A_T = A[i, j].positional(j, i)
assert torch.allclose(A.T, A_T)
```

Flattening and Splitting Dims
-----------------------------

**Tuples of dimensions** can be passed to both indexing and `positional`. In indexing, this will split the dimension being indexed across the dimensions in the tuple.  In `positional` it will flatten the dimensions in a single positional dimension:

```{code-cell} ipython3
i, j, k = dims()
j.size = 2
A = torch.rand(6, 4)
a = A[(i, j), k] # split dim 0 into i,j
print(i.size, j.size, k.size)
```

```{code-cell} ipython3
r = a.positional(i, (j, k)) # flatten j and k
print(r.shape)
```

The size of one unsized dimension in a tuple such as `i` can be inferred if the other sizes are known.

Examples
========

einsum-style products
------

```{code-cell} ipython3
def mm(A, B):
    i, j, k = dims()
    r = (A[i, k] * B[k, j]).sum(k)
    return r.positional(i, j)
```

```{code-cell} ipython3
def bmm(A, B):
    b, i, j, k = dims()
    r = (A[b, i, k] * B[b, k, j]).sum(k)
    return r.positional(b, i, j)
```

```{code-cell} ipython3
def bmm_2(A, B):
    i = dims() # note: doesn't matter than mm internally also uses i
    return mm(A[i], B[i])
```

```{code-cell} ipython3
from dim import softmax
def attention(K, Q, V):
    batch, channel, key, query = dims()
    A = (K[batch, channel, key]*Q[batch, channel, query]).sum(channel)
    A = softmax(A * (channel.size ** -0.5), dim=key)
    R = (V[batch, channel, key] * A).sum(key)
    return torch.cat((R.positional(batch, channel, query), Q), dim=1)
```

einops
------
[einops tutorial](http://einops.rocks/pytorch-examples.html)

```{code-cell} ipython3
from einops import rearrange
def pixel_shuffle_einops(img, upscale_factor=2):
    return rearrange(img, 'b (c h2 w2) h w -> b c (h h2) (w w2)', h2=upscale_factor, w2=upscale_factor)

def pixel_shuffle(img, upscale_factor=2):
    h2, w2, c, b, h, w = dims(upscale_factor, upscale_factor)
    return img[b, (c, h2, w2), h, w].positional(b, c, (h, h2), (w, w2))

```


Restyling Gram matrix for style transfer
```{code-cell} ipython3
def gram_matrix_new_einops(y):
    b, ch, h, w = y.shape
    return torch.einsum('bchw,bdhw->bcd', [y, y]) / (h * w)

```


vmap, xmap
----------

Rule #1 means that is easy to implicitly batch things. The way of specifying how to batch has lighter weight syntax as well.

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
```

note the awkward mapping/unmapping apis in xmap


mult-headed attension
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
    return context_layer.positional(batch, query_sequence, [heads, features])
```

indirect indexing
-----------------

```{code-cell} ipython3
from torch import where
def triu(A):
   i,j = dims()
   a = A[i, j]
   return torch.where(i <= j, a, 0).positional(i, j)
```

```{code-cell} ipython3
def embedding_bag(input, embedding_weights):
    batch, sequence = dims()
    r = embedding_weights[input[batch, sequence], features].sum(sequence)
    return r.positional(batch, features)
```

Relative positional embeddings associate an embedding vector with the distance between the query and the key in the sequence.
For instance, a key two elements after query after key will get embedding ID 2. We can use first-class dimensions to do the indexing arithmetic, and embedding lookup.

```{code-cell} ipython3
def relative_positional_embedding(q, k, distance_embedding_weight):
    batch, query_sequence, key_sequence, heads, features = dims()
    q = q[batch, query_sequence, [heads, features]]
    k = k[batch, key_sequence, [heads, features]]

    distance = query_sequence - key_sequence
    n_embeddings = distance_embedding_weight.size(0)
    index_bias = n_embeddings // 2

    assert key_sequence.size + bias <= n_embeddings

    positional_embedding = distance_embedding_weight[distance + index_bias, features]

    relative_position_scores_query = (q*positional_embedding).sum(features)
    relative_position_scores_key = (k*positional_embedding).sum(features)
    return  (relative_position_scores_query + relative_position_scores_key).positional(batch, heads, key_sequence, query_sequence)
```


    upper triangular

+++

tensor puzzlers
============

[Tensor Puzzlers](https://github.com/srush/Tensor-Puzzles) are a good excersize for learning the numpy and torch APIs by figuring out how to define common operations using a small set of primitive tensor operations.

However, the difficulty of many of the puzzlers lies not in how to compute the answer but the awkwardness of the primitives themselves.

**With first class dimensions, these puzzlers are nearly the same as the spec that defines them**

+++

### Puzzle 3 - outer

Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.

```{code-cell} ipython3
def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]

def outer(a, b):
    i, j = dims()
    return (a[i] * b[j]).positional(i, j)
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
    return a[i, i].positional(i)
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
    return where(i.eq(j), 1, 0).positional(i, j)
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
    return where(i <= j, 1, 0).positional(i, j)
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
    return where(i - 1 >= 0, d, a[i]).positional(i)
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
    return where(v.eq(0),  a[i], b[i]).positional(v, i)
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
    return a[where(i + 1 < i.size, i + 1, 0)].positional(i)
```

### Puzzle 11 - flip

Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector

```{code-cell} ipython3
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]

def flip(a, i: int):
    i = dims(a.size(0))
    return a[i.size - i - 1].positional(i)
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
    return where(j < length[i], v, 0).positional(i, j)
```

Advantages of First-class Dimensions over Strings
=================================================

* einops add new operators and lack rule #1
* strings for dimensions prevent having rule #3, which then requries additional operators to bind dimensions because it cannot be a property of indexing.
* Avoiding accidental capture and name collisions with local binding.
* Ability to define dim-specific operators directly on the dimenision
