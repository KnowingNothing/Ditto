import torch

from safety_evaluator import BaseSafetyEvaluator

class GradNormSafetyEvaluator(BaseSafetyEvaluator):

    """
    Zero-Cost Proxies for Lightweight NAS
    https://github.com/idstcv/ZenNAS/blob/main/ZeroShotProxy/compute_gradnorm_score.py
    """

    def __init__(self):
        super(GradNormSafetyEvaluator, self).__init__()
    
    def evaluate(self, module, model, input_shape, **kwargs):
        # generate inputs
        inputs = torch.randn(*input_shape)
        model_device = next(model.parameters()).device
        inputs = inputs.to(device=model_device)

        # run forward-backward
        model.zero_grad()
        outputs = model(inputs)
        outputs.backward(torch.ones_like(outputs))

        # compute score
        norm2_sum = outputs.new_zeros([])
        with torch.no_grad():
            for p in module.parameters():
                if hasattr(p, 'grad') and p.grad is not None:
                    norm2_sum += torch.norm(p.grad) ** 2

        score = torch.sqrt(norm2_sum).item()

        return score


if __name__ == '__main__':
    import torchvision.models as models

    model = models.resnet18(pretrained=False).cuda()
    module = dict(model.named_modules())['layer4.0']
    print(f"Selected module: {module}")

    evaluator = GradNormSafetyEvaluator()
    score = evaluator.evaluate(module, model, [32, 3, 224, 224])
    print(f"score: {score:.4f}")
