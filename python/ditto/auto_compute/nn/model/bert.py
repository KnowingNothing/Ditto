import tvm
from ..module import Module, Sequential
from ..functional import softmax
from ...graph import layer
from .transformer import TransformerBlock


"""
Reference:
https://github.com/codertimo/BERT-pytorch
"""


class BERT(Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(
        self,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dtype="float32",
        mma_in_dtype="float16",
        mma_acc_dtype="float32",
        mma_MI=16,
        mma_NI=16,
        mma_KI=16
    ):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = Sequential(
            *[
                TransformerBlock(
                    hidden,
                    attn_heads,
                    min(hidden * 4, 3072),
                    dtype=dtype,
                    mma_in_dtype=mma_in_dtype,
                    mma_acc_dtype=mma_acc_dtype,
                    mma_MI=mma_MI,
                    mma_NI=mma_NI,
                    mma_KI=mma_KI
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        # x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        return self.transformer_blocks(x)
