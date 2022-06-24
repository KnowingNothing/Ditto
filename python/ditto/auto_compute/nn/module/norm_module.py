from pkg_resources import require
import tvm
from .module import Module, Parameter
from ..functional import batch_norm2d_nchw_v1, batch_norm2d_nchw_v2, layer_norm_infer
from ...graph import LayerTensor, layer


class BatchNorm2d(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        dtype="float32",
        out_dtype="float32",
        layout="NCHW",
        version=2,
    ):
        super(BatchNorm2d, self).__init__()
        assert layout in ["NCHW"]
        assert version in [1, 2]
        self.alpha = Parameter((num_features,), dtype=dtype, name="bn_alpha")
        self.beta = Parameter((num_features,), dtype=dtype, name="bn_beta")
        if version == 2:
            self.mean = Parameter((num_features,), dtype=dtype, name="bn_mean")
            self.variance = Parameter((num_features,), dtype=dtype, name="bn_var")
        self.version = version
        self.layout = layout
        self.eps = eps

        self.dtype = dtype
        self.out_dtype = out_dtype
        assert out_dtype == dtype

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        if self.version == 1:
            outputs = batch_norm2d_nchw_v1(
                inputs.tensor, self.alpha.tensor, self.beta.tensor, self.eps
            )
            bn_layer = layer(
                outputs.op,
                inputs=[inputs.tensor],
                weights=[self.alpha.tensor, self.beta.tensor],
                const_scalars=None,
                const_tensors=None,
                requires_grad=self.training,
                name=f"bn2d_{self.layout}_layer",
            )
        elif self.version == 2:
            outputs = batch_norm2d_nchw_v2(
                inputs.tensor,
                self.mean.tensor,
                self.variance.tensor,
                self.alpha.tensor,
                self.beta.tensor,
                self.eps,
            )
            bn_layer = layer(
                outputs.op,
                inputs=[inputs.tensor],
                weights=[
                    self.mean.tensor,
                    self.variance.tensor,
                    self.alpha.tensor,
                    self.beta.tensor,
                ],
                const_scalars=None,
                const_tensors=None,
                requires_grad=self.training,
                name=f"bn2d_{self.layout}_layer",
            )
        else:
            raise NotImplementedError(f"No batch norm for version: {self.version}.\n")

        return bn_layer(inputs)


class LayerNormInfer(Module):
    "Construct a layernorm module for inference."

    def __init__(self, feature_shape, dtype="float32"):
        super(LayerNormInfer, self).__init__()
        self.alpha = Parameter(feature_shape, dtype=dtype, name="ln_alpha")
        self.beta = Parameter(feature_shape, dtype=dtype, name="ln_beta")
        self.num_feature_dims = len(feature_shape)

    def forward(self, inputs):
        inputs = self.preprocess(inputs)
        outputs = layer_norm_infer(
            inputs.tensor, self.alpha.tensor, self.beta.tensor, self.num_feature_dims
        )
        ln_layer = layer(
            outputs.op,
            inputs=[inputs.tensor],
            weights=[self.alpha.tensor, self.beta.tensor],
            requires_grad=self.training,
            name="layer_norm_infer_layer",
        )
        return ln_layer(inputs)
