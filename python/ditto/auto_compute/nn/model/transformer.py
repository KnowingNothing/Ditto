import tvm
from ..module import (
    Module,
    Parameter,
    Linear,
    GELU,
    MultiHeadAttentionMMA,
    LayerNormInfer,
)
from ..functional import softmax
from ...graph import layer


"""
Reference:
https://github.com/codertimo/BERT-pytorch
"""


class PositionwiseFeedForward(Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dtype="float32", out_dtype="float16"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Linear(d_model, d_ff, bias=False, dtype=dtype, out_dtype=out_dtype)
        self.w_2 = Linear(d_ff, d_model, bias=False, dtype=dtype, out_dtype=out_dtype)
        self.activation = GELU()

    def forward(self, x):
        x = self.preprocess(x)
        return self.activation(self.w_2(self.activation(self.w_1(x))))


class TransformerBlock(Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self,
        hidden,
        attn_heads,
        feed_forward_hidden,
        dtype="float32",
        mma_in_dtype="float16",
        mma_acc_dtype="float32",
    ):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        """

        super().__init__()
        self.attention = MultiHeadAttentionMMA(
            num_heads=attn_heads,
            hidden_size=hidden,
            in_dtype=mma_in_dtype,
            acc_dtype=mma_acc_dtype,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dtype=dtype, out_dtype=dtype
        )
        self.layer_norm1 = LayerNormInfer(feature_shape=[hidden], dtype=dtype)
        self.layer_norm2 = LayerNormInfer(feature_shape=[hidden], dtype=dtype)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.attention(x, x, x)
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        return x
