import torch

from safety_evaluator import BaseSafetyEvaluator

class SynflowSafetyEvaluator(BaseSafetyEvaluator):

    """
    Neural Architecture Search without Training
    https://github.com/idstcv/ZenNAS/blob/main/ZeroShotProxy/compute_syncflow_score.py
    https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/synflow.py
    """

    def __init__(self):
        super(SynflowSafetyEvaluator, self).__init__()
    
    def evaluate(self, module, model, input_shape, **kwargs):
        
        # convert parameters to abs + signs
        @torch.no_grad()
        def linearize(net):
            signs = {}
            for name, param in net.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        # convert to original values
        @torch.no_grad()
        def nonlinearize(net, signs):
            for name, param in net.state_dict().items():
                # if 'weight_mask' not in name:
                param.mul_(signs[name])

        # prepare model
        signs = linearize(model)
        model.double()

        # generate inputs
        inputs = torch.ones(*input_shape).double()
        model_device = next(model.parameters()).device
        inputs = inputs.to(device=model_device)

        # run forward-backward
        model.zero_grad()
        outputs = model(inputs)
        outputs.backward(torch.ones_like(outputs))
        
        # compute score
        grads_abs_list = list()
        with torch.no_grad():
            for p in module.parameters():
                if hasattr(p, 'grad') and p.grad is not None:
                    grads_abs_list.append(torch.abs(p * p.grad))

        score = 0.0
        for grad_abs in grads_abs_list:
            # TODO: does this apply to general layers other than CONV and FC?
            reduce_dims = list(range(1, grad_abs.ndim))
            score += torch.mean(torch.sum(grad_abs, dim=reduce_dims)).item()
        
        score = -score

        # restore model
        nonlinearize(model, signs)

        return score


if __name__ == '__main__':
    import torchvision.models as models

    model = models.resnet18(pretrained=False).cuda()
    module = dict(model.named_modules())['layer4.0']
    print(f"Selected module: {module}")

    evaluator = SynflowSafetyEvaluator()
    score = evaluator.evaluate(module, model, [32, 3, 224, 224])
    print(f"score: {score:.4f}")
