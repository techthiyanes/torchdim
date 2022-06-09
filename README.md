First-class Dimensions
======================

_The functionality of einops (einsum, rearrange), batching (vmap, xmap), and tensor indexing with one new concept_

The tensor input to a resnet might have the shape [8, 3, 224, 224] but informally we think of those dimensions as 'batch', 'channel', 'width', and 'height'. Eventhough 'width' and 'height' have the same _size_ we still think of them as separate dimensions, and if we have two _different_ images, we think of both as sharing the _same_ 'channel' dimension.

 Instead of treating this concept informally, first-class dimensions introduce a Python object, a `Dim`, to represent the concept. By expanding the semantics of tensors with dim objects, we can get behavior equivalent to batching transforms (xmap, vmap), einops-style rearragement, and loop-style tensor indexing.

Creating and Binding Dims
=========================

Python objects that represent dimension are created using the `dims` operator[^1].

    batch, channel, width, height = dims()

Other representations such as [Named Tensor](https://pytorch.org/docs/stable/named_tensor.html) in PyTorch, or  [JAX's xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html) use strings to name dimensions. We call these dimensions _first class_ because they are instead Python objects.

In addition to the normal _positional_ dimensions in a tensor, tensors can also have a separate set of first-class dimensions. You can create tensors with first-class dimensions by binding them using indexing:

    input = torch.rand(2, 3, 224, 224)
    print(input.ndim)
    > 4
    input_fc = input[batch, channel, width, height]
    print(input_fc.dims) # first class dimensions
    > (batch, channel, width, height)
    print(input_fc.ndim) # positional dimensions
    > 0

    input_mixed = input[batch, :, :, height]
    print(input_mixed.dims)
    > (batch, height)
    print(input_mixed.ndims)
    > 2

Dimensions will take on the size of the first thing they are bound to:

    print(batch.size)
    > 2

But you can also directly set the size of dimension:

    i = dims()
    i.size = 5 # ok, i previously did not have a size

    i.size = 5 # ok, it already had the size 5
    i.size = 3 # error! already set to size 3
    j = dims(4) # can also be set on construction

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


    avg_pixel_color = input_fc.mean(width, height)
    print(avg_color.dims)
    > (batch, channel)

Any other ther first-class dimensions (e.g. batch, channel) are still implicitly batched according to Rule #1.

Rule 3: Dims are Tensors
------------------------
**A first-class dimension `d` can be used wherever a Tensor is expected. It will act as if it were a tensor whose only dimension is itself, `d`, and the values along the dimension are the indices of each entry `(0, 1, 2, ..., d.size - 1)`**

    print(channel.dims)
    > (channel, )
    print(channel + 1000)
    > tensor([1000, 1001, 1002])
    > with dims=(channel,) torch.Size([3])

This means that a dimensions used as a tensor acts as an index into that dimension. Going back to our loop-level analogy, it is analogous to using the loop variable as a value:

    # mental model: loop-level analogy
    for channels in range(batch.size):
        compute channels + 1000

This makes doing complicated indexing arithmetic appear the same as it would in a for loop, but without executing a loop in Python. Consider what a lookup from an embedding table would look like:

    sequence, features = dims()
    embeddings = torch.rand(8, 128)
    words = torch.tensor([5, 4, 0,])

    state = embeddings[words[sequence], features]
    print(state.dims)
    > (sequence, features)

With the following analogy to loops:

    # mental model: loop-level analogy

    for sequence in range(words.size(0)):
        for features in range(embeddings.size(1)):
            state = embeddings[words[sequence], features]


Removing Dims
-------------
The `positional` method converts a first-class dimensions in a tensor back to a normal positional dimensions.

By specifiying a different order from how things were originally bound, it is easy to do transpositions.

    i, j = dims()
    A = torch.rand(3, 4)
    A_T = A[i, j].positional(j, i)
    assert torch.allclose(A.T, A_T)

Flattening and Splitting Dims
-----------------------------

**Tuples of dimensions** can be passed to both indexing and `positional`. In indexing, this will split the dimension being indexed across the dimensions in the tuple.  In `positional` it will flatten the dimensions in a single positional dimension:

    i, j, k = dims()
    j.size = 2
    A = torch.rand(6, 4)
    a = A[(i, j), k] # split dim 0 into i,j
    print(i.size, j.size, k.size)
    > 3, 2, 4
    r = a.positional(i, (j, k)) # flatten j and k
    print(r.shape)
    > [3, 8]

The size of one unsized dimension in a tuple such as `i` can be inferred if the other sizes are known.

Examples
========

einsum
------

    matmul

    batch matmul # note how the concepts compose

    attention

    outer products


einops
------

    pixels shuffle


vmap, xmap
----------

Rule #1 means that is easy to implicitly batch things. The way of specifying how to batch has lighter weight syntax as well.

note the awkward mapping/unmapping apis,



indirect indexing
-----------------

    embeddings
    relative positional embeddings
    upper triangular

mult-headed attension
---------------------

Flatten/Unflatten the heads
einsums of the attension products
indirect indexing of the embeddings

tensor puzzlers
---------------

[Tensor Puzzlers](https://github.com/srush/Tensor-Puzzles) are a good excersize for learning the numpy and torch APIs by figuring out how to define common operations using a small set of primitive tensor operations.

However, the difficulty of many of the puzzlers lies not in how to compute the answer but the awkwardness of the primitives themselves.

**With first class dimensions, these puzzlers are nearly the same as the spec that defines them**

    outer
    diag
    eye
    triu
    diff
    vstack
    roll
    flip
    sequence_mask


Advantages of First-class Dimensions over Strings
=================================================

* einops add new operators and lack rule #1
* strings for dimensions prevent having rule #3, which then requries additional operators to bind dimensions because it cannot be a property of indexing.
* Avoiding accidental capture and name collisions with local binding.
* Ability to define dim-specific operators directly on the dimenision
