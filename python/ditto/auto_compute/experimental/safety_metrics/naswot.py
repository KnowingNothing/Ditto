import numpy as np
import torch
import torch.nn as nn

from safety_evaluator import BaseSafetyEvaluator


class NASWOTSafetyEvaluator(BaseSafetyEvaluator):
    
    """
    Neural Architecture Search without Training
    https://github.com/idstcv/ZenNAS/blob/main/ZeroShotProxy/compute_NASWOT_score.py
    """
    
    def __init__(self):
        super(NASWOTSafetyEvaluator, self).__init__()

    def evaluate(self, module, model: nn.Module, input_shape, **kwargs):

        # register kernel matrix
        batch_size = input_shape[0]
        model.kernel_matrix = np.zeros([batch_size, batch_size])

        # register forward hooks
        def counting_forward_hook(m, inp, out):
            inp = inp if not isinstance(inp, tuple) else inp[0]
            inp = inp.detach().view(inp.size(0), -1)  # [batch-size, flattened]
            relu = (inp > 0).float()
            kern_mat = relu @ relu.T + (1 - relu) @ (1 - relu.T)
            model.kernel_matrix += kern_mat.cpu().numpy()

        handles = list()
        for name, submodule in module.named_modules():
            if not isinstance(submodule, nn.ReLU): continue
            handle = submodule.register_forward_hook(counting_forward_hook)
            handles.append(handle)

        # generate inputs
        inputs = torch.randn(*input_shape)
        model_device = next(model.parameters()).device
        inputs = inputs.to(device=model_device)

        # compute score
        jacob, outputs = self._get_batch_jacobian(model, inputs)
        score = self._logdet(model.kernel_matrix)

        # cleanup
        for handle in handles:
            handle.remove()

        delattr(model, 'kernel_matrix')

        return score

    @staticmethod
    def _logdet(K):
        # compute log-determinant of the kernel matrix K
        sign, logdet = np.linalg.slogdet(K)
        return float(logdet)

    @staticmethod
    def _get_batch_jacobian(model, inputs):
        model.zero_grad()
        inputs.requires_grad_(True)
        outputs = model(inputs)
        outputs.backward(torch.ones_like(outputs))
        jacob = inputs.grad.detach()
        return jacob, outputs.detach()


if __name__ == '__main__':
    import torchvision.models as models

    model = models.resnet18(pretrained=False).cuda()
    module = dict(model.named_modules())['layer4.0']
    print(f"Selected module: {module}")

    evaluator = NASWOTSafetyEvaluator()
    score = evaluator.evaluate(module, model, [32, 3, 224, 224])
    print(f"score: {score:.4f}")
