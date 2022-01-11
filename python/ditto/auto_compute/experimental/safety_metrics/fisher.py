import numpy as np
import torch
import torch.nn as nn

from safety_evaluator import BaseSafetyEvaluator


class FisherSafetyEvaluator(BaseSafetyEvaluator):
    def __init__(self):
        super(FisherSafetyEvaluator, self).__init__()
    
    def evaluate(self, module, model, input_shape, **kwargs):
        module.fisher = 0.
        module.act = 0.

        def counting_forward_hook(m, inp, out):
            m.act = out

        def counting_backward_hook(m, grad_inp, grad_out):
            act = m.act.detach()
            grad = grad_out[0].detach()
            if act.ndim > 2:
                g_nk = torch.sum((act * grad), list(range(2, act.ndim)))
            else:
                g_nk = act * grad
            del_k = g_nk.pow(2).mean(0).mul(0.5).sum()
            m.fisher += del_k
            delattr(module, 'act')

        fp_handle = module.register_forward_hook(counting_forward_hook)
        bp_handle = module.register_backward_hook(counting_backward_hook)

        # generate inputs
        inputs = torch.randn(*input_shape)
        model_device = next(model.parameters()).device
        inputs = inputs.to(device=model_device)

        # run forward-backward
        model.zero_grad()
        outputs = model(inputs)
        outputs.backward(torch.ones_like(outputs))

        # calculate score
        score = module.fisher.abs().item()

        # cleanup
        fp_handle.remove()
        bp_handle.remove()
        assert not hasattr(module, 'act')
        delattr(module, 'fisher')

        return score


if __name__ == '__main__':
    import torchvision.models as models

    model = models.resnet18(pretrained=False).cuda()
    module = dict(model.named_modules())['layer4.0']
    print(f"Selected module: {module}")

    evaluator = FisherSafetyEvaluator()
    score = evaluator.evaluate(module, model, [32, 3, 224, 224])
    print(f"score: {score:.4f}")
