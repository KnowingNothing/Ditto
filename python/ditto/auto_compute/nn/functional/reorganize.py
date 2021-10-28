import tvm


def shuffle_channels(inputs, groups):
    '''
    inputs: [batch, channel, height, width]
    groups: int
    return: [batch, channel * height * width]
    '''
    assert len(inputs.shape) == 4
    assert inputs.shape[1] % groups == 0
    batch, channel, height, width = inputs.shape
    channel_per_group = channel // groups

    input_view = tvm.te.compute(
        [batch, groups, channel_per_group, height, width],
        lambda n, g, c, h, w: inputs[n, g * channel_per_group + c, h, w],
        name="input_view"
    )

    shuffle = tvm.te.compute(
        [batch, channel, height, width],
        lambda n, c, h, w: input_view[n, c % groups, c // groups, h, w],
        name="shuffle_channels",
    )

    return shuffle


def batch_flatten(inputs):
    '''
    inputs: [batch, channel, height, width]
    return: [batch, channel * height * width]
    '''
    assert len(inputs.shape) == 4
    batch, channel, height, width = inputs.shape

    return tvm.te.compute(
        [batch, channel * height * width],
        lambda i, j: inputs[i, j//(height * width), (j % (height * width)) // width,
                            j % width],
        name="batch_flatten"
    )


def cat_channel(A, B):
    assert len(A.shape) == 4 and len(B.shape) == 4
    assert A.shape[0] == B.shape[0] and A.shape[2:] == B.shape[2:]

    batch, channel_A, height, width = A.shape
    channel_B = B.shape[1]

    return tvm.te.compute(
        [batch, channel_A + channel_B, height, width],
        lambda n, c, h, w: tvm.tir.Select(
            c < channel_A, A[n, c, h, w], B[n, c - channel_A, h, w]),
        name="cat_channel",
    )
