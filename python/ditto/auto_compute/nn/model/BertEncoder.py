import tvm
from ..module import (
    Module,
    Parameter,
    Linear,
    Add,
    GELU
)

from ..functional import softmax

from ...graph import layer


class DenseLayer(Module):
    def __init__(self, out_dtype="float32"):
        super(DenseLayer, self).__init__()
        self.out_dtype = out_dtype

    def forward(self, inputs, weight, d_k_const):
        inputs, weight = self.preprocess(inputs, weight)
        N = int(inputs.shape[0])
        d_k = int(inputs.shape[-1])
        T_q = int(inputs.shape[-2])
        T_k = int(weight.shape[-2])
        k = tvm.te.reduce_axis((0, d_k))
        outputs = tvm.te.compute(
            [N, T_q, T_k],
            lambda n, tq, tk: tvm.te.sum(
                (inputs[n, tq, k] * weight[n, tk, k] / tvm.te.sqrt(d_k_const)).astype(self.out_dtype), axis=[k]),
            name="dense"
        )

        d_layer = layer(outputs.op, inputs=[inputs.tensor, weight.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="dense_layer")
        return d_layer(inputs, weight)


class DenseLayer2(Module):
    def __init__(self, out_dtype="float32"):
        super(DenseLayer2, self).__init__()
        self.out_dtype = out_dtype

    def forward(self, inputs, weight):
        inputs, weight = self.preprocess(inputs, weight)
        N, T_q, K = inputs.shape
        _, _, d_v = weight.shape
        k = tvm.te.reduce_axis((0, K))
        outputs = tvm.te.compute(
            [N, T_q, d_v],
            lambda b, m, n: tvm.te.sum(
                (inputs[b, m, k] * weight[b, k, n]).astype(self.out_dtype), axis=[k]),
            name="dense2"
        )

        d_layer = layer(outputs.op, inputs=[inputs.tensor, weight.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="dense2_layer")
        return d_layer(inputs, weight)


class SoftmaxLayer(Module):
    def __init__(self):
        super(SoftmaxLayer, self).__init__()

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = softmax(inputs)
        d_layer = layer(outputs.op, inputs=[inputs.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="softmax_layer")
        return d_layer(inputs)


class ScaledDotProductAttention(Module):
    def __init__(self, causality=False, out_dtype="float32"):
        super(ScaledDotProductAttention, self).__init__()
        self.causality = causality
        self.out_dtype = out_dtype
        self.d1 = DenseLayer(out_dtype)
        self.d2 = DenseLayer2(out_dtype)
        self.softmax = SoftmaxLayer()

    def forward(self, Q, K, V):
        '''See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        # key_masks: A 2d tensor with shape of [N, key_seqlen]
        causality: If True, applies masking for future blinding
        '''
        d_k_const = tvm.tir.const(int(Q.shape[-1]), dtype=self.out_dtype)
        outputs = self.d1(Q, K, d_k_const)
        outputs = self.softmax(outputs)
        outputs = self.d2(outputs, V)
        return outputs


class SplitCatLayer(Module):
    def __init__(self, num_heads):
        super(SplitCatLayer, self).__init__()
        self.num_heads = num_heads

    def forward(self, inputs, N, T, d_model):
        inputs = self.preprocess(inputs)
        h = tvm.tir.const(self.num_heads, "int32")
        outputs = tvm.te.compute(
            [self.num_heads * N, T, d_model // self.num_heads],
            lambda n_h, t, d_h: inputs[n_h // h, t, d_h * h + n_h % h],
            "split_cat"
        )
        d_layer = layer(outputs.op, inputs=[inputs.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="split_cat_layer")
        return d_layer(inputs)


class FuseCatLayer(Module):
    def __init__(self, num_heads):
        super(FuseCatLayer, self).__init__()
        self.num_heads = num_heads

    def forward(self, inputs, N, T, d_model):
        inputs = self.preprocess(inputs)
        h = tvm.tir.const(self.num_heads, "int32")
        outputs = tvm.te.compute(
            [N, T, d_model],
            lambda n, t, d: inputs[n * h + d % h, t, d // h],
            "fuse_cat"
        )
        d_layer = layer(outputs.op, inputs=[inputs.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="fuse_cat_layer")
        return d_layer(inputs)


def layer_normalization(inputs, epsilon=1e-8):
    """
    Layer Norm
    inputs: [N, T, d_model]
    """
    N, T, d_model = inputs.shape
    prefix = inputs.name
    epsilon = tvm.tir.const(epsilon, inputs.dtype)

    rn1 = tvm.te.reduce_axis([0, N], name=prefix + "_rn1")
    rt1 = tvm.te.reduce_axis([0, T], name=prefix + "_rt1")
    mean = tvm.te.compute(
        [d_model],
        lambda d: tvm.te.sum(
            inputs[rn1, rt1, d] / (N*T), axis=[rn1, rt1]),
        name=prefix + "_mean",
    )

    rn2 = tvm.te.reduce_axis([0, N], name=prefix + "_rn2")
    rt2 = tvm.te.reduce_axis([0, T], name=prefix + "_rt2")
    square = tvm.te.compute(
        [d_model],
        lambda d: tvm.te.sum(
            (inputs[rn2, rt2, d] * inputs[rn2, rt2, d]) / (N*T), axis=[rn2, rt2]),
        name=prefix + "_square"
    )

    var = tvm.te.compute(
        [d_model],
        lambda d: square[d] - mean[d] * mean[d],
        name=prefix + "_var"
    )

    return tvm.te.compute(
        [N, T, d_model],
        lambda n, t, d: (
            inputs[n, t, d] - mean[d]) / tvm.te.sqrt(var[d] + epsilon),
        name=prefix + "_ln2d"
    )


class LayerNorm(Module):
    def __init__(self, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = layer_normalization(inputs, self.eps)
        d_layer = layer(outputs.op, inputs=[inputs.tensor],
                        weights=None,
                        const_scalars=None,
                        const_tensors=None,
                        requires_grad=self.training,
                        name="layer_norm_layer")
        return d_layer(inputs)


class MultiHeadAttentionLayer(Module):
    def __init__(self, ma_l1, ma_l2, ma_l3, ma_l4, num_heads=8, causality=False, out_dtype="float32"):
        super(MultiHeadAttentionLayer, self).__init__()
        self.ma_l1 = ma_l1
        self.ma_l2 = ma_l2
        self.ma_l3 = ma_l3
        self.ma_l4 = ma_l4
        self.num_heads = num_heads
        self.causality = causality
        self.out_dtype = out_dtype
        self.split_cat = SplitCatLayer(num_heads)
        self.fuse_cat = FuseCatLayer(num_heads)
        self.add = Add()
        self.ln = LayerNorm()
        self.scaled_dot_product_attention = ScaledDotProductAttention(
            causality, out_dtype)

    def forward(self, queries, keys, values):
        Q = self.ma_l1(queries)
        K = self.ma_l2(keys)
        V = self.ma_l3(values)

        N, T, d_model = queries.shape
        Q_ = self.split_cat(Q, N, T, d_model)
        K_ = self.split_cat(K, N, T, d_model)
        V_ = self.split_cat(V, N, T, d_model)

        outputs = self.scaled_dot_product_attention(Q_, K_, V_)
        outputs = self.fuse_cat(outputs, N, T, d_model)
        outputs = self.ma_l4(outputs)
        outputs = self.add(outputs, queries)
        outputs = self.ln(outputs)
        return outputs


class FastForwardLayer(Module):
    def __init__(self, ff_l1, ff_l2, num_units=None):
        super(FastForwardLayer, self).__init__()
        self.ff_l1 = ff_l1
        self.ff_l2 = ff_l2
        self.gelu = GELU()
        self.add = Add()
        self.ln = LayerNorm()

    def forward(self, inputs):
        outputs = self.ff_l1(inputs)
        outputs = self.gelu(outputs)
        outputs = self.ff_l2(outputs)
        outputs = self.add(outputs, inputs)
        outputs = self.ln(outputs)
        return outputs


class Transformer(Module):
    '''
    xs: tuple of
        x: tensor. (N, T1)
    '''

    def __init__(self, num_blocks, num_heads, d_ff, d_model, dtype="float32", out_dtype="float32"):
        super(Transformer, self).__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_model = d_model
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.ma_l1 = []
        self.ma_l2 = []
        self.ma_l3 = []
        self.ma_l4 = []
        self.ff_l1 = []
        self.ff_l2 = []
        for i in range(num_blocks):
            self.ma_l1.append(
                Linear(d_model, d_model, bias=True, dtype=dtype, out_dtype=out_dtype))
            self.ma_l2.append(
                Linear(d_model, d_model, bias=True, dtype=dtype, out_dtype=out_dtype))
            self.ma_l3.append(
                Linear(d_model, d_model, bias=True, dtype=dtype, out_dtype=out_dtype))
            self.ma_l4.append(
                Linear(d_model, d_model, bias=True, dtype=dtype, out_dtype=out_dtype))
            self.ff_l1.append(
                Linear(d_model, self.d_ff, dtype=dtype, out_dtype=out_dtype))
            self.ff_l2.append(Linear(self.d_ff, self.d_model,
                              dtype=dtype, out_dtype=out_dtype))
        self.multihead_attentions = []
        self.ffs = []
        for i in range(self.num_blocks):
            self.multihead_attentions.append(
                MultiHeadAttentionLayer(
                    self.ma_l1[i], self.ma_l2[i], self.ma_l3[i], self.ma_l4[i],
                    causality=False, out_dtype=self.out_dtype
                )
            )
            self.ffs.append(
                FastForwardLayer(
                    self.ff_l1[i], self.ff_l2[i], num_units=[
                        self.d_ff, self.d_model]
                )
            )

    def forward(self, x):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''

        enc = x

        # Blocks
        for i in range(self.num_blocks):
            # self-attention
            att = self.multihead_attentions[i]
            enc = att(enc, enc, enc)
            # feed forward
            ff = self.ffs[i]
            enc = ff(enc)
        return enc


def BertBaseEncoder(dtype="float32", out_dtype="float32"):
    # https://huggingface.co/bert-base-uncased/blob/main/config.json
    bert_base_config = {
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }

    N = 1  # Batch Size
    T = bert_base_config['max_position_embeddings']
    d_model = bert_base_config['hidden_size']
    d_ff = bert_base_config['intermediate_size']
    num_blocks = bert_base_config['num_hidden_layers']
    num_heads = bert_base_config['num_attention_heads']

    net = Transformer(num_blocks, num_heads, d_ff, d_model,
                      dtype=dtype, out_dtype=out_dtype)
    return net
