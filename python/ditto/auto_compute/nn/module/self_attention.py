import tvm
from .module import Module, Parameter
from ..functional import BatchGemmSoftmaxGemmMMA, reshape, transpose
from ...graph import LayerTensor, layer, layer_tensor


MUTI_HEAD_ATTENTION_LAYER = "multi_head_attention_mma_layer"
SELF_ATTENTION_MODULE_REG = set([MUTI_HEAD_ATTENTION_LAYER])


def is_self_attention_layer(layer):
    return layer.name in SELF_ATTENTION_MODULE_REG


class MultiHeadAttentionMMA(Module):
    """
    MultiHead Attention with MMA fusion
    """

    def __init__(self, num_heads, hidden_size, in_dtype="float32", acc_dtype="float32", mma_MI=16, mma_NI=16, mma_KI=16):
        super(MultiHeadAttentionMMA, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.model_k = hidden_size // num_heads
        self.in_dtype = in_dtype
        self.acc_dtype = acc_dtype
        self.MI = mma_MI
        self.NI = mma_NI
        self.KI = mma_KI

    def forward(self, query, key, value):
        """
        query: layer_tensor([batch, M, K], name="A", dtype=in_dtype)
        key: layer_tensor([batch, M, K], name="B", dtype=in_dtype)
        value: layer_tensor([batch, M, K], name="C", dtype=in_dtype)
        """
        query, key, value = self.preprocess(query, key, value)
        batch, M, K = query.tensor.shape
        try:
            assert int(K) == self.hidden_size
        except:
            raise RuntimeError(
                "Can't prove the self attention inputs conform to the hidden size."
            )
        query_mh = reshape(query, [batch, M, self.num_heads, self.model_k])
        query_mh = transpose(query_mh, [0, 2, 1, 3])
        query_mh = reshape(query_mh, [batch * self.num_heads, M, self.model_k])
        key_mh = reshape(key, [batch, M, self.num_heads, self.model_k])
        key_mh = transpose(key_mh, [0, 2, 3, 1])
        key_mh = reshape(key_mh, [batch * self.num_heads, self.model_k, M])
        value_mh = reshape(value, [batch, M, self.num_heads, self.model_k])
        value_mh = transpose(value_mh, [0, 2, 1, 3])
        value_mh = reshape(value_mh, [batch * self.num_heads, M, self.model_k])
        outputs_mh = BatchGemmSoftmaxGemmMMA(
            query_mh,
            key_mh,
            value_mh,
            MI=self.MI,
            NI=self.NI,
            KI=self.KI,
            in_dtype=self.in_dtype,
            acc_dtype=self.acc_dtype,
        )
        outputs_mh = reshape(
            outputs_mh, [batch, self.num_heads, M, self.model_k])
        outputs_mh = transpose(outputs_mh, [0, 2, 1, 3])
        outputs = reshape(outputs_mh, [batch, M, K])
        attention_layer = layer(
            outputs.op,
            inputs=[query.tensor, key.tensor, value.tensor],
            weights=[],
            requires_grad=self.training,
            name=MUTI_HEAD_ATTENTION_LAYER,
        )
        return attention_layer(query, key, value)
