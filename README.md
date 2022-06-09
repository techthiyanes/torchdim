First-class Dimensions
======================

_The functionality of batching (vmap, xmap), einops (einsum, rearrange), and tensor indexing with one new object_

The tensor input to a resnet might have the shape [8, 3, 224, 224] but informally we think of those dimensions as 'batch', 'channel', 'width', and 'height'. Eventhough width and height have the same _size_ we still think of them as separate dimensions, and if we have two images, we think of both as sharing the exact same 'channel' dimension.

 Instead of treating this concept informally, first-class dimensions introduce a Python object, a `Dim`, to represent the concept. By expanding the definition of existing tensors and operators on tensors to support dim objects, we can get behavior equivalent to batching transforms (xmap, vmap), einops-style rearragement, and loop-style tensor indexing.

Creating and Binding Dims
=========================

Python objects that represent dimension are created using the `dims` operator.

    batch, channel, width, height = dims()

Other representations such as named tensor in PyTorch, or in JAX's xmap use strings to name dimensions. We call these dimensions _first class_ because they are instead Python objects.

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

Semantics of Dimensions
=======================
Three rules fully define how dimension objects work with existing tensors APIs. The power of this abstraction arises from the composition of these rules with existing operators.

Rule 1: Implicit Batching
-------------------------
**Tensor operations (e.g. `input + bias`) are implicitly batched over the union of the first-class dimensions in their inputs.**

If `input` has dimensions `batch, channel` and `bias` has dimension `channel`, the output will have the union of those dimensions (`batch, channel`), and the result will computed as if there was a loop over all the first-class dimensions:

    forall batch:
        forall channel:
            compute a + b normally on positional dimensions

Positional dimensions behave as they did before (e.g. for + they will broadcast). This rule parallels the rules for named dimensions in xmap, or the implicitly batched dimensions in vmap.

Rule 2: Specifying dimensions
-----------------------------
**Wherever an integer is used to specify a dimension in the existing torch operator, a first-class dimensions can be used instead to tell the operator to work over that dimension.**


    avg_pixel_color = input_fc.mean(width, height)
    print(avg_color.dims)
    > (batch, channel)

Any other ther first-class dimensions (e.g. batch, channel) are still implicitly batched according to Rule #1.

Rule 3: Dims are Tensors
------------------------
**A first-class dimension `d` can be used wherever a Tensor is expected. It will act as if it were a tensor whose only dimension is the first-class dimension `d` and the values along that dimension are the indices of each entry `(0, 1, 2, ..., d.size - 1)`**

    print(channels.dims)
    > (channels, )
    print(channels + 1000)
    > tensor([1000, 1001, 1002])
    > with dims=(channels,) torch.Size([3])

This means that a dimensions used as a tensor acts as an index into that dimension. This makes doing complicated indexing arithmetic appear the same as it would in a for loop, but without executing a loop in Python:

    def roll_loop(x):
        r = torch.empty_like(x)
        for i in x.size(0):
            r[i] = x[i - 1]
        return r

    def roll_fc(x):
        r = torch.empty_like(x)
        i = dims(x.size(0))
        r[i] = x[i - 1]

    x = torch.rand(5)
    assert torch.allclose(roll_loop(x), roll_fc(x)

Additional APIs
===============

For the most part, first-class dims extend the functionality of the existing torch APIs as defined by the rules above. However, we also have a few convience methods for working with dimensions.

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


Advantages of First-class Dimensions over Strings
-------------------------------------------------

* Avoiding accidental capture and name collisions with local binding.
* Ability to define dim-specific operators directly on the dimenision
